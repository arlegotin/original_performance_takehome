"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def _pack_vliw(self, instrs: list[dict[Engine, list[tuple]]]):
        def slot_rw(engine, slot):
            reads = set()
            writes = set()
            mem_reads = False
            mem_writes = False
            match engine:
                case "alu":
                    _, dest, a1, a2 = slot
                    reads = {a1, a2}
                    writes = {dest}
                case "valu":
                    match slot:
                        case ("vbroadcast", dest, src):
                            reads = {src}
                            writes = {dest + i for i in range(VLEN)}
                        case ("multiply_add", dest, a, b, c):
                            reads = {
                                a + i for i in range(VLEN)
                            } | {b + i for i in range(VLEN)} | {c + i for i in range(VLEN)}
                            writes = {dest + i for i in range(VLEN)}
                        case (op, dest, a1, a2):
                            reads = {a1 + i for i in range(VLEN)} | {
                                a2 + i for i in range(VLEN)
                            }
                            writes = {dest + i for i in range(VLEN)}
                case "load":
                    match slot:
                        case ("const", dest, _val):
                            writes = {dest}
                        case ("load", dest, addr):
                            reads = {addr}
                            writes = {dest}
                            mem_reads = True
                        case ("load_offset", dest, addr, offset):
                            reads = {addr + offset}
                            writes = {dest + offset}
                            mem_reads = True
                        case ("vload", dest, addr):
                            reads = {addr}
                            writes = {dest + i for i in range(VLEN)}
                            mem_reads = True
                case "store":
                    match slot:
                        case ("store", addr, src):
                            reads = {addr, src}
                            mem_writes = True
                        case ("vstore", addr, src):
                            reads = {addr} | {src + i for i in range(VLEN)}
                            mem_writes = True
                case "flow":
                    match slot:
                        case ("select", dest, cond, a, b):
                            reads = {cond, a, b}
                            writes = {dest}
                        case ("add_imm", dest, a, _imm):
                            reads = {a}
                            writes = {dest}
                        case ("vselect", dest, cond, a, b):
                            reads = {cond + i for i in range(VLEN)} | {
                                a + i for i in range(VLEN)
                            } | {b + i for i in range(VLEN)}
                            writes = {dest + i for i in range(VLEN)}
                        case ("pause",):
                            pass
                        case ("halt",):
                            pass
                        case ("cond_jump", cond, addr):
                            reads = {cond, addr}
                        case ("cond_jump_rel", cond, _off):
                            reads = {cond}
                        case ("jump", addr):
                            reads = {addr}
                        case ("jump_indirect", addr):
                            reads = {addr}
                        case ("coreid", dest):
                            writes = {dest}
            return reads, writes, mem_reads, mem_writes

        def flatten(segment):
            ops = []
            pause_index = 0
            for instr in segment:
                for engine, slots in instr.items():
                    for slot in slots:
                        reads, writes, mem_reads, mem_writes = slot_rw(engine, slot)
                        pause_role = None
                        if engine == "flow" and slot == ("pause",):
                            pause_index += 1
                            pause_role = "start" if pause_index == 1 else "end"
                        ops.append(
                            {
                                "engine": engine,
                                "slot": slot,
                                "reads": reads,
                                "writes": writes,
                                "mem_reads": mem_reads,
                                "mem_writes": mem_writes,
                                "pause_role": pause_role,
                            }
                        )
            return ops

        def schedule_segment(segment):
            ops = flatten(segment)
            if not ops:
                return []
            pause_token = -2
            has_start_pause = any(op["pause_role"] == "start" for op in ops)
            last_write = {}
            last_read = {}
            store_ops = []
            deps = [0] * len(ops)
            succs = [set() for _ in range(len(ops))]

            def add_edge(u, v):
                if v not in succs[u]:
                    succs[u].add(v)
                    deps[v] += 1

            for i, op in enumerate(ops):
                reads = set(op["reads"])
                writes = set(op["writes"])
                pause_role = op["pause_role"]
                if pause_role == "start":
                    writes.add(pause_token)
                elif pause_role == "end":
                    reads.add(pause_token)

                if op["engine"] == "store":
                    store_ops.append(i)
                    if has_start_pause:
                        reads.add(pause_token)
                for addr in reads:
                    if addr in last_write:
                        add_edge(last_write[addr], i)
                for addr in writes:
                    if addr in last_write:
                        add_edge(last_write[addr], i)
                    if addr in last_read:
                        add_edge(last_read[addr], i)
                for addr in reads:
                    last_read[addr] = i
                for addr in writes:
                    last_write[addr] = i
                    last_read.pop(addr, None)
                if pause_role == "end" and store_ops:
                    for store_idx in store_ops:
                        add_edge(store_idx, i)

            heights = [1] * len(ops)
            for i in range(len(ops) - 1, -1, -1):
                if succs[i]:
                    heights[i] = 1 + max(heights[j] for j in succs[i])

            ready = [i for i, d in enumerate(deps) if d == 0]
            ready.sort()
            remaining = len(ops)
            remaining_by_engine = defaultdict(int)
            for op in ops:
                remaining_by_engine[op["engine"]] += 1
            scheduled_instrs = []
            while remaining:
                counts = {k: 0 for k in SLOT_LIMITS if k != "debug"}
                bundle = {}
                scheduled = []
                engine_urgency = {
                    eng: remaining_by_engine[eng] / SLOT_LIMITS[eng]
                    for eng in remaining_by_engine
                }
                ready.sort(
                    key=lambda i: (
                        -heights[i],
                        -engine_urgency[ops[i]["engine"]],
                        i,
                    )
                )
                for i in ready:
                    op = ops[i]
                    eng = op["engine"]
                    if counts[eng] >= SLOT_LIMITS[eng]:
                        continue
                    bundle.setdefault(eng, []).append(op["slot"])
                    counts[eng] += 1
                    scheduled.append(i)
                if not scheduled:
                    raise RuntimeError("Scheduler deadlock")
                scheduled_instrs.append(bundle)
                remaining -= len(scheduled)
                for i in scheduled:
                    remaining_by_engine[ops[i]["engine"]] -= 1
                scheduled_set = set(scheduled)
                next_ready = [i for i in ready if i not in scheduled_set]
                for i in scheduled:
                    for v in succs[i]:
                        deps[v] -= 1
                        if deps[v] == 0:
                            next_ready.append(v)
                next_ready.sort()
                ready = next_ready
            return scheduled_instrs

        packed = schedule_segment(instrs)
        merged = []
        for instr in packed:
            if instr.get("flow") == [("pause",)] and merged:
                prev = merged[-1]
                if "flow" not in prev:
                    prev["flow"] = instr["flow"]
                    continue
            merged.append(instr)
        return merged

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        SIMD implementation of reference_kernel2 using VLEN=8.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        def emit(**engine_slots):
            instr = {}
            for name, slots in engine_slots.items():
                if slots:
                    instr[name] = slots
            if instr:
                self.instrs.append(instr)

        def alloc_vec(name):
            return self.alloc_scratch(name, VLEN)

        # Scratch space addresses
        init_vars = [
            ("forest_values_p", 4),
            ("inp_indices_p", 5),
            ("inp_values_p", 6),
        ]
        for v, _ in init_vars:
            self.alloc_scratch(v, 1)
        for v, header_idx in init_vars:
            self.add("load", ("const", tmp1, header_idx))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        four_const = self.scratch_const(4)
        shift19_const = self.scratch_const(HASH_STAGES[1][4])
        shift9_const = self.scratch_const(HASH_STAGES[3][4])
        shift16_const = self.scratch_const(HASH_STAGES[5][4])

        mul12_const = self.scratch_const(1 + (1 << HASH_STAGES[0][4]))
        mul5_const = self.scratch_const(1 + (1 << HASH_STAGES[2][4]))
        mul3_const = self.scratch_const(1 + (1 << HASH_STAGES[4][4]))

        c1_const = self.scratch_const(HASH_STAGES[0][1])
        c2_const = self.scratch_const(HASH_STAGES[1][1])
        c3_const = self.scratch_const(HASH_STAGES[2][1])
        c4_const = self.scratch_const(HASH_STAGES[3][1])
        c5_const = self.scratch_const(HASH_STAGES[4][1])
        c6_const = self.scratch_const(HASH_STAGES[5][1])
        c6_lsb = HASH_STAGES[5][1] & 1

        node0_const = self.alloc_scratch("node0_const")
        node1_const = self.alloc_scratch("node1_const")
        node2_const = self.alloc_scratch("node2_const")
        node3_const = self.alloc_scratch("node3_const")
        node4_const = self.alloc_scratch("node4_const")
        node5_const = self.alloc_scratch("node5_const")
        node6_const = self.alloc_scratch("node6_const")
        node7_const = self.alloc_scratch("node7_const")
        node8_const = self.alloc_scratch("node8_const")
        node9_const = self.alloc_scratch("node9_const")
        node10_const = self.alloc_scratch("node10_const")
        node11_const = self.alloc_scratch("node11_const")
        node12_const = self.alloc_scratch("node12_const")
        node13_const = self.alloc_scratch("node13_const")
        node14_const = self.alloc_scratch("node14_const")
        node_idx_consts = [self.scratch_const(i) for i in range(15)]
        node_consts = [
            node0_const,
            node1_const,
            node2_const,
            node3_const,
            node4_const,
            node5_const,
            node6_const,
            node7_const,
            node8_const,
            node9_const,
            node10_const,
            node11_const,
            node12_const,
            node13_const,
            node14_const,
        ]
        for node_idx_const, node_const in zip(node_idx_consts, node_consts):
            self.add(
                "alu",
                ("+", tmp1, self.scratch["forest_values_p"], node_idx_const),
            )
            self.add("load", ("load", node_const, tmp1))
        one_minus_base = self.alloc_scratch("one_minus_base")
        self.add(
            "alu",
            ("-", one_minus_base, one_const, self.scratch["forest_values_p"]),
        )
        two_minus_base = self.alloc_scratch("two_minus_base")
        self.add("alu", ("+", two_minus_base, one_minus_base, one_const))

        base_offsets = list(range(0, batch_size, VLEN))
        partial_alu_shift9_offsets = set(base_offsets[:16])
        partial_alu_shift16_offsets = set(base_offsets[:16])
        partial_alu_xor_offsets = set(base_offsets[:0])
        partial_valu_shift19_offsets = set(base_offsets[i] for i in (1, 3))
        partial_valu_shift9_offsets = set(base_offsets[:0])
        partial_valu_shift16_offsets = set(base_offsets[:0])
        partial_alu_xor_tmp_offsets = set(base_offsets[:0])
        partial_valu_parity_offsets = set(base_offsets[:0])
        partial_valu_adjust_offsets = set(base_offsets[:0])
        base_consts = [self.scratch_const(i) for i in base_offsets]

        idx_scratch = self.alloc_scratch("idx_scratch", batch_size)
        val_scratch = self.alloc_scratch("val_scratch", batch_size)
        tmp_scratch = self.alloc_scratch("tmp_scratch", batch_size)
        shift_scratch = self.alloc_scratch("shift_scratch", batch_size)

        zero_vec = alloc_vec("zero_vec")
        one_vec = alloc_vec("one_vec")
        two_vec = alloc_vec("two_vec")
        four_vec = alloc_vec("four_vec")
        forest_values_vec = alloc_vec("forest_values_vec")
        one_minus_base_vec = alloc_vec("one_minus_base_vec")
        two_minus_base_vec = alloc_vec("two_minus_base_vec")

        shift9_vec = alloc_vec("shift9_vec")
        shift16_vec = alloc_vec("shift16_vec")
        shift19_vec = alloc_vec("shift19_vec") if partial_valu_shift19_offsets else None
        mul12_vec = alloc_vec("mul12_vec")
        mul5_vec = alloc_vec("mul5_vec")
        mul3_vec = alloc_vec("mul3_vec")

        c1_vec = alloc_vec("c1_vec")
        c2_vec = alloc_vec("c2_vec")
        c3_vec = alloc_vec("c3_vec")
        c4_vec = alloc_vec("c4_vec")
        c5_vec = alloc_vec("c5_vec")
        c6_vec = alloc_vec("c6_vec")
        node0_vec = alloc_vec("node0_vec")
        node1_vec = alloc_vec("node1_vec")
        node2_vec = alloc_vec("node2_vec")
        node3_vec = alloc_vec("node3_vec")
        node4_vec = alloc_vec("node4_vec")
        node5_vec = alloc_vec("node5_vec")
        node6_vec = alloc_vec("node6_vec")
        node7_vec = alloc_vec("node7_vec")
        node8_vec = alloc_vec("node8_vec")
        node9_vec = alloc_vec("node9_vec")
        node10_vec = alloc_vec("node10_vec")
        node11_vec = alloc_vec("node11_vec")
        node12_vec = alloc_vec("node12_vec")
        node13_vec = alloc_vec("node13_vec")
        node14_vec = alloc_vec("node14_vec")
        sel_b0_vec = alloc_vec("sel_b0_vec")
        sel_t0_vec = alloc_vec("sel_t0_vec")
        sel_t1_vec = alloc_vec("sel_t1_vec")
        sel_b0_vec2 = alloc_vec("sel_b0_vec2")
        sel_t0_vec2 = alloc_vec("sel_t0_vec2")
        sel_t1_vec2 = alloc_vec("sel_t1_vec2")
        sel2_t0_vec = alloc_vec("sel2_t0_vec")
        sel2_t1_vec = alloc_vec("sel2_t1_vec")

        vbroadcast_slots = [
            ("vbroadcast", zero_vec, zero_const),
            ("vbroadcast", one_vec, one_const),
            ("vbroadcast", two_vec, two_const),
            ("vbroadcast", four_vec, four_const),
            ("vbroadcast", forest_values_vec, self.scratch["forest_values_p"]),
            ("vbroadcast", one_minus_base_vec, one_minus_base),
            ("vbroadcast", two_minus_base_vec, two_minus_base),
            ("vbroadcast", shift9_vec, shift9_const),
            ("vbroadcast", shift16_vec, shift16_const),
            ("vbroadcast", mul12_vec, mul12_const),
            ("vbroadcast", mul5_vec, mul5_const),
            ("vbroadcast", mul3_vec, mul3_const),
            ("vbroadcast", c1_vec, c1_const),
            ("vbroadcast", c2_vec, c2_const),
            ("vbroadcast", c3_vec, c3_const),
            ("vbroadcast", c4_vec, c4_const),
            ("vbroadcast", c5_vec, c5_const),
            ("vbroadcast", c6_vec, c6_const),
            ("vbroadcast", node0_vec, node0_const),
            ("vbroadcast", node1_vec, node1_const),
            ("vbroadcast", node2_vec, node2_const),
            ("vbroadcast", node3_vec, node3_const),
            ("vbroadcast", node4_vec, node4_const),
            ("vbroadcast", node5_vec, node5_const),
            ("vbroadcast", node6_vec, node6_const),
            ("vbroadcast", node7_vec, node7_const),
            ("vbroadcast", node8_vec, node8_const),
            ("vbroadcast", node9_vec, node9_const),
            ("vbroadcast", node10_vec, node10_const),
            ("vbroadcast", node11_vec, node11_const),
            ("vbroadcast", node12_vec, node12_const),
            ("vbroadcast", node13_vec, node13_const),
            ("vbroadcast", node14_vec, node14_const),
        ]
        if shift19_vec is not None:
            vbroadcast_slots.append(("vbroadcast", shift19_vec, shift19_const))
        for i in range(0, len(vbroadcast_slots), SLOT_LIMITS["valu"]):
            emit(valu=vbroadcast_slots[i : i + SLOT_LIMITS["valu"]])

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))

        # Allocate separate address storage to avoid RAW/WAW hazards on tmp1/tmp2
        load_idx_addrs = self.alloc_scratch("load_idx_addrs", len(base_offsets))
        load_val_addrs = self.alloc_scratch("load_val_addrs", len(base_offsets))

        # Compute all load addresses in parallel
        alu_slots = []
        for i, base_const in enumerate(base_consts):
            alu_slots.append(("+", load_idx_addrs + i, self.scratch["inp_indices_p"], base_const))
            alu_slots.append(("+", load_val_addrs + i, self.scratch["inp_values_p"], base_const))
        for i in range(0, len(alu_slots), SLOT_LIMITS["alu"]):
            emit(alu=alu_slots[i : i + SLOT_LIMITS["alu"]])

        # Emit all loads - scheduler will parallelize across both load units
        for i, base_offset in enumerate(base_offsets):
            idx_block = idx_scratch + base_offset
            val_block = val_scratch + base_offset
            emit(load=[("vload", idx_block, load_idx_addrs + i), ("vload", val_block, load_val_addrs + i)])

        def emit_valu(slots):
            for i in range(0, len(slots), SLOT_LIMITS["valu"]):
                emit(valu=slots[i : i + SLOT_LIMITS["valu"]])

        def emit_alu(slots):
            for i in range(0, len(slots), SLOT_LIMITS["alu"]):
                emit(alu=slots[i : i + SLOT_LIMITS["alu"]])

        def emit_load(slots):
            for i in range(0, len(slots), SLOT_LIMITS["load"]):
                emit(load=slots[i : i + SLOT_LIMITS["load"]])

        # Maintain values in c6-xored form throughout rounds.
        slots = []
        for base_offset in base_offsets:
            val_block = val_scratch + base_offset
            slots.append(("^", val_block, val_block, c6_vec))
        emit_valu(slots)

        # Convert node vectors to node ^ c6 for c6-hoisted hash.
        node_xor_slots = [
            ("^", node0_vec, node0_vec, c6_vec),
            ("^", node1_vec, node1_vec, c6_vec),
            ("^", node2_vec, node2_vec, c6_vec),
            ("^", node3_vec, node3_vec, c6_vec),
            ("^", node4_vec, node4_vec, c6_vec),
            ("^", node5_vec, node5_vec, c6_vec),
            ("^", node6_vec, node6_vec, c6_vec),
            ("^", node7_vec, node7_vec, c6_vec),
            ("^", node8_vec, node8_vec, c6_vec),
            ("^", node9_vec, node9_vec, c6_vec),
            ("^", node10_vec, node10_vec, c6_vec),
            ("^", node11_vec, node11_vec, c6_vec),
            ("^", node12_vec, node12_vec, c6_vec),
            ("^", node13_vec, node13_vec, c6_vec),
            ("^", node14_vec, node14_vec, c6_vec),
        ]
        emit_valu(node_xor_slots)

        wrap_round = forest_height + 1

        def emit_round_two_nodes(use_parity_cond: bool = False):
            if not use_parity_cond:
                slots = []
                for base_offset in base_offsets:
                    idx_block = idx_scratch + base_offset
                    tmp_block = tmp_scratch + base_offset
                    slots.append(("==", tmp_block, idx_block, one_vec))
                emit_valu(slots)
                for base_offset in base_offsets:
                    tmp_block = tmp_scratch + base_offset
                    emit(flow=[("vselect", tmp_block, tmp_block, node1_vec, node2_vec)])
            else:
                for base_offset in base_offsets:
                    tmp_block = tmp_scratch + base_offset
                    if c6_lsb:
                        emit(flow=[("vselect", tmp_block, tmp_block, node1_vec, node2_vec)])
                    else:
                        emit(flow=[("vselect", tmp_block, tmp_block, node2_vec, node1_vec)])
            slots = []
            for base_offset in base_offsets:
                val_block = val_scratch + base_offset
                tmp_block = tmp_scratch + base_offset
                slots.append(("^", val_block, val_block, tmp_block))
            emit_valu(slots)

        def emit_round_four_nodes():
            slots = []
            for base_offset in base_offsets:
                idx_block = idx_scratch + base_offset
                b1_block = shift_scratch + base_offset
                slots.append(("&", b1_block, idx_block, two_vec))
            emit_valu(slots)
            for base_offset in base_offsets:
                b1_block = shift_scratch + base_offset
                b0_block = tmp_scratch + base_offset
                emit(flow=[("vselect", sel2_t0_vec, b1_block, node6_vec, node4_vec)])
                emit(flow=[("vselect", sel2_t1_vec, b1_block, node3_vec, node5_vec)])
                if c6_lsb:
                    emit(flow=[("vselect", b0_block, b0_block, sel2_t1_vec, sel2_t0_vec)])
                else:
                    emit(flow=[("vselect", b0_block, b0_block, sel2_t0_vec, sel2_t1_vec)])
            slots = []
            for base_offset in base_offsets:
                val_block = val_scratch + base_offset
                sel_block = tmp_scratch + base_offset
                slots.append(("^", val_block, val_block, sel_block))
            emit_valu(slots)

        def emit_round_eight_nodes(sel_t0, sel_t1, sel_b0):
            # Bit-tree vselect for depth-3 nodes (7..14) using idx bitmasks.
            for base_offset in base_offsets:
                idx_block = idx_scratch + base_offset
                tmp_block = tmp_scratch + base_offset
                mask_block = shift_scratch + base_offset
                val_block = val_scratch + base_offset
                emit_valu([("&", mask_block, idx_block, two_vec)])
                emit(
                    flow=[
                        ("vselect", sel_t0, tmp_block, node9_vec, node8_vec),
                        ("vselect", sel_t1, tmp_block, node11_vec, node10_vec),
                    ]
                )
                emit(flow=[("vselect", sel_t0, mask_block, sel_t1, sel_t0)])
                emit(
                    flow=[
                        ("vselect", sel_t1, tmp_block, node13_vec, node12_vec),
                        ("vselect", sel_b0, tmp_block, node7_vec, node14_vec),
                    ]
                )
                emit(flow=[("vselect", sel_t1, mask_block, sel_b0, sel_t1)])
                emit_valu([("&", mask_block, idx_block, four_vec)])
                emit(flow=[("vselect", sel_t0, mask_block, sel_t1, sel_t0)])
                emit_valu([("^", val_block, val_block, sel_t0)])

        final_addr = False
        skip_last_addr = False
        if rounds > 0 and wrap_round > 4:
            last_round = rounds - 1
            if last_round >= 4:
                last_phase = (last_round - 4) % wrap_round
                final_addr = last_phase <= wrap_round - 5
                if last_round % wrap_round == wrap_round - 1:
                    final_addr = False
                if final_addr and last_phase == 0:
                    skip_last_addr = True
                    final_addr = False

        for round_idx in range(rounds):
            use_addr = False
            phase = None
            if round_idx >= 4 and wrap_round > 4:
                phase = (round_idx - 4) % wrap_round
                use_addr = phase <= wrap_round - 5
            if skip_last_addr and round_idx == rounds - 1:
                use_addr = False
                phase = None

            if use_addr and phase == 0:
                slots = []
                for base_offset in base_offsets:
                    idx_block = idx_scratch + base_offset
                    slots.append(("+", idx_block, idx_block, forest_values_vec))
                emit_valu(slots)

            if round_idx == 0 or round_idx == wrap_round:
                slots = []
                for base_offset in base_offsets:
                    val_block = val_scratch + base_offset
                    slots.append(("^", val_block, val_block, node0_vec))
                emit_valu(slots)
            elif round_idx == 1 or round_idx == wrap_round + 1:
                emit_round_two_nodes(use_parity_cond=True)
            elif round_idx == 2 or round_idx == wrap_round + 2:
                emit_round_four_nodes()
            elif round_idx == 3:
                emit_round_eight_nodes(sel_t0_vec, sel_t1_vec, sel_b0_vec)
            elif round_idx == wrap_round + 3:
                emit_round_eight_nodes(sel_t0_vec2, sel_t1_vec2, sel_b0_vec2)
            else:
                slots = []
                if not use_addr:
                    for base_offset in base_offsets:
                        idx_block = idx_scratch + base_offset
                        tmp_block = tmp_scratch + base_offset
                        slots.append(("+", tmp_block, idx_block, forest_values_vec))
                    emit_valu(slots)

                load_slots = []
                for base_offset in base_offsets:
                    addr_block = (
                        idx_scratch + base_offset
                        if use_addr
                        else tmp_scratch + base_offset
                    )
                    tmp_block = tmp_scratch + base_offset
                    for offset in range(VLEN):
                        load_slots.append(
                            ("load_offset", tmp_block, addr_block, offset)
                        )
                emit_load(load_slots)

                slots = []
                for base_offset in base_offsets:
                    tmp_block = tmp_scratch + base_offset
                    slots.append(("^", tmp_block, tmp_block, c6_vec))
                emit_valu(slots)

                slots = []
                for base_offset in base_offsets:
                    val_block = val_scratch + base_offset
                    tmp_block = tmp_scratch + base_offset
                    slots.append(("^", val_block, val_block, tmp_block))
                emit_valu(slots)

            slots = []
            for base_offset in base_offsets:
                val_block = val_scratch + base_offset
                slots.append(("multiply_add", val_block, val_block, mul12_vec, c1_vec))
            emit_valu(slots)

            valu_slots = []
            alu_slots = []
            for base_offset in base_offsets:
                val_block = val_scratch + base_offset
                tmp_block = shift_scratch + base_offset
                if base_offset in partial_valu_shift19_offsets:
                    valu_slots.append((">>", tmp_block, val_block, shift19_vec))
                else:
                    for lane in range(VLEN):
                        alu_slots.append(
                            (">>", tmp_block + lane, val_block + lane, shift19_const)
                        )
            if alu_slots:
                emit_alu(alu_slots)
            if valu_slots:
                emit_valu(valu_slots)

            valu_slots = []
            alu_slots = []
            for base_offset in base_offsets:
                val_block = val_scratch + base_offset
                tmp_block = shift_scratch + base_offset
                if base_offset in partial_alu_xor_tmp_offsets:
                    for lane in range(VLEN):
                        alu_slots.append(
                            ("^", val_block + lane, val_block + lane, tmp_block + lane)
                        )
                else:
                    valu_slots.append(("^", val_block, val_block, tmp_block))
            if alu_slots:
                emit_alu(alu_slots)
            emit_valu(valu_slots)

            valu_slots = []
            alu_slots = []
            for base_offset in base_offsets:
                val_block = val_scratch + base_offset
                if base_offset in partial_alu_xor_offsets:
                    for lane in range(VLEN):
                        alu_slots.append(
                            ("^", val_block + lane, val_block + lane, c2_const)
                        )
                else:
                    valu_slots.append(("^", val_block, val_block, c2_vec))
            if alu_slots:
                emit_alu(alu_slots)
            emit_valu(valu_slots)

            slots = []
            for base_offset in base_offsets:
                val_block = val_scratch + base_offset
                slots.append(("multiply_add", val_block, val_block, mul5_vec, c3_vec))
            emit_valu(slots)

            # Use ALU shifts for rounds 3, 10 (wrap-1), and 14 (wrap+3) to reduce VALU pressure
            use_alu_shift = round_idx == 3 or round_idx == wrap_round - 1 or round_idx == wrap_round + 3
            use_alu_shift16 = round_idx == 3 or round_idx == wrap_round - 1
            if use_alu_shift16:
                valu_slots = []
                alu_slots = []
                for base_offset in base_offsets:
                    val_block = val_scratch + base_offset
                    tmp_block = shift_scratch + base_offset
                    if base_offset in partial_valu_shift9_offsets:
                        valu_slots.append(("<<", tmp_block, val_block, shift9_vec))
                    else:
                        for lane in range(VLEN):
                            alu_slots.append(
                                ("<<", tmp_block + lane, val_block + lane, shift9_const)
                            )
                if alu_slots:
                    emit_alu(alu_slots)
                if valu_slots:
                    emit_valu(valu_slots)
            else:
                use_partial_alu_shift = round_idx == rounds - 1
                valu_slots = []
                alu_slots = []
                for base_offset in base_offsets:
                    val_block = val_scratch + base_offset
                    tmp_block = shift_scratch + base_offset
                    if use_partial_alu_shift and base_offset in partial_alu_shift9_offsets:
                        for lane in range(VLEN):
                            alu_slots.append(
                                ("<<", tmp_block + lane, val_block + lane, shift9_const)
                            )
                    else:
                        valu_slots.append(("<<", tmp_block, val_block, shift9_vec))
                if alu_slots:
                    emit_alu(alu_slots)
                emit_valu(valu_slots)

            slots = []
            for base_offset in base_offsets:
                val_block = val_scratch + base_offset
                slots.append(("+", val_block, val_block, c4_vec))
            emit_valu(slots)

            slots = []
            for base_offset in base_offsets:
                val_block = val_scratch + base_offset
                tmp_block = shift_scratch + base_offset
                slots.append(("^", val_block, val_block, tmp_block))
            emit_valu(slots)

            slots = []
            for base_offset in base_offsets:
                val_block = val_scratch + base_offset
                slots.append(("multiply_add", val_block, val_block, mul3_vec, c5_vec))
            emit_valu(slots)

            if use_alu_shift:
                valu_slots = []
                alu_slots = []
                for base_offset in base_offsets:
                    val_block = val_scratch + base_offset
                    tmp_block = shift_scratch + base_offset
                    if base_offset in partial_valu_shift16_offsets:
                        valu_slots.append((">>", tmp_block, val_block, shift16_vec))
                    else:
                        for lane in range(VLEN):
                            alu_slots.append(
                                (">>", tmp_block + lane, val_block + lane, shift16_const)
                            )
                if alu_slots:
                    emit_alu(alu_slots)
                if valu_slots:
                    emit_valu(valu_slots)
            else:
                use_partial_alu_shift = round_idx == rounds - 1
                valu_slots = []
                alu_slots = []
                for base_offset in base_offsets:
                    val_block = val_scratch + base_offset
                    tmp_block = shift_scratch + base_offset
                    if use_partial_alu_shift and base_offset in partial_alu_shift16_offsets:
                        for lane in range(VLEN):
                            alu_slots.append(
                                (">>", tmp_block + lane, val_block + lane, shift16_const)
                            )
                    else:
                        valu_slots.append((">>", tmp_block, val_block, shift16_vec))
                if alu_slots:
                    emit_alu(alu_slots)
                emit_valu(valu_slots)

            slots = []
            for base_offset in base_offsets:
                val_block = val_scratch + base_offset
                tmp_block = shift_scratch + base_offset
                slots.append(("^", val_block, val_block, tmp_block))
            emit_valu(slots)

            if round_idx % wrap_round == wrap_round - 1:
                slots = []
                for base_offset in base_offsets:
                    idx_block = idx_scratch + base_offset
                    slots.append(("+", idx_block, zero_vec, zero_vec))
                emit_valu(slots)
            else:
                # Parity AND: mostly ALU, optionally VALU for selected blocks
                valu_slots = []
                alu_slots = []
                for base_offset in base_offsets:
                    val_block = val_scratch + base_offset
                    tmp_block = tmp_scratch + base_offset
                    if base_offset in partial_valu_parity_offsets:
                        valu_slots.append(("&", tmp_block, val_block, one_vec))
                    else:
                        for lane in range(VLEN):
                            alu_slots.append(
                                ("&", tmp_block + lane, val_block + lane, one_const)
                            )
                if alu_slots:
                    emit_alu(alu_slots)
                if valu_slots:
                    emit_valu(valu_slots)

                slots = []
                for base_offset in base_offsets:
                    idx_block = idx_scratch + base_offset
                    if c6_lsb:
                        if use_addr:
                            slots.append(
                                ("multiply_add", idx_block, idx_block, two_vec, two_minus_base_vec)
                            )
                        else:
                            slots.append(
                                ("multiply_add", idx_block, idx_block, two_vec, two_vec)
                            )
                    else:
                        if use_addr:
                            slots.append(
                                ("multiply_add", idx_block, idx_block, two_vec, one_minus_base_vec)
                            )
                        else:
                            slots.append(
                                ("multiply_add", idx_block, idx_block, two_vec, one_vec)
                            )
                emit_valu(slots)

                if round_idx == rounds - 1:
                    slots = []
                    for base_offset in base_offsets:
                        idx_block = idx_scratch + base_offset
                        tmp_block = tmp_scratch + base_offset
                        if c6_lsb:
                            slots.append(("-", idx_block, idx_block, tmp_block))
                        else:
                            slots.append(("+", idx_block, idx_block, tmp_block))
                    emit_valu(slots)
                else:
                    # Always use ALU for parity adjustment to reduce VALU pressure
                    alu_slots = []
                    valu_slots = []
                    if c6_lsb:
                        for base_offset in base_offsets:
                            idx_block = idx_scratch + base_offset
                            tmp_block = tmp_scratch + base_offset
                            if base_offset in partial_valu_adjust_offsets:
                                valu_slots.append(("-", idx_block, idx_block, tmp_block))
                            else:
                                for lane in range(VLEN):
                                    alu_slots.append(
                                        (
                                            "-",
                                            idx_block + lane,
                                            idx_block + lane,
                                            tmp_block + lane,
                                        )
                                    )
                    else:
                        for base_offset in base_offsets:
                            idx_block = idx_scratch + base_offset
                            tmp_block = tmp_scratch + base_offset
                            if base_offset in partial_valu_adjust_offsets:
                                valu_slots.append(("+", idx_block, idx_block, tmp_block))
                            else:
                                for lane in range(VLEN):
                                    alu_slots.append(
                                        (
                                            "+",
                                            idx_block + lane,
                                            idx_block + lane,
                                            tmp_block + lane,
                                        )
                                    )
                    if alu_slots:
                        emit_alu(alu_slots)
                    if valu_slots:
                        emit_valu(valu_slots)

        if final_addr:
            slots = []
            for base_offset in base_offsets:
                idx_block = idx_scratch + base_offset
                slots.append(("-", idx_block, idx_block, forest_values_vec))
            emit_valu(slots)

        # Convert back from c6-xored form for final output.
        slots = []
        for base_offset in base_offsets:
            val_block = val_scratch + base_offset
            slots.append(("^", val_block, val_block, c6_vec))
        emit_valu(slots)

        # Reuse load addresses for stores (same inp_indices_p/inp_values_p + base_const calculation)
        # This eliminates 64 ALU operations from the critical tail
        # Emit all stores - scheduler will parallelize across both store units
        for i, base_offset in enumerate(base_offsets):
            idx_block = idx_scratch + base_offset
            val_block = val_scratch + base_offset
            emit(store=[("vstore", load_idx_addrs + i, idx_block), ("vstore", load_val_addrs + i, val_block)])

        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})
        self.instrs = self._pack_vliw(self.instrs)

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_indices(self):
        random.seed(123)
        params = [
            (8, 8, 128),
            (8, 13, 192),
            (9, 16, 256),
            (10, 20, 256),
        ]
        for forest_height, rounds, batch_size in params:
            forest = Tree.generate(forest_height)
            inp = Input.generate(forest, batch_size, rounds)
            mem = build_mem_image(forest, inp)

            kb = KernelBuilder()
            kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
            machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
            machine.enable_pause = False
            machine.enable_debug = False
            machine.run()

            for ref_mem in reference_kernel2(mem):
                pass

            inp_indices_p = ref_mem[5]
            inp_values_p = ref_mem[6]
            assert (
                machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)]
                == ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)]
            ), "Incorrect output indices"
            assert (
                machine.mem[inp_values_p : inp_values_p + len(inp.values)]
                == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
            ), "Incorrect output values"

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
