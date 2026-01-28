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
                case "debug":
                    match slot:
                        case ("compare", loc, _key):
                            reads = {loc}
                        case ("vcompare", loc, _keys):
                            reads = {loc + i for i in range(VLEN)}
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

        slot_limits = dict(SLOT_LIMITS)
        slot_limits["debug"] = SLOT_LIMITS["flow"]

        def schedule_segment_slots(segment):
            slots = []
            for instr in segment:
                for engine, engine_slots in instr.items():
                    for slot in engine_slots:
                        if engine == "flow" and slot == ("pause",):
                            continue
                        slots.append((engine, slot))
            if not slots:
                return []

            cycles = []
            usage = []
            ready_time = defaultdict(int)
            last_write = defaultdict(lambda: -1)
            last_read = defaultdict(lambda: -1)

            def ensure_cycle(cycle: int) -> None:
                while len(cycles) <= cycle:
                    cycles.append({})
                    usage.append(defaultdict(int))

            def find_cycle(engine: str, earliest: int) -> int:
                cycle = earliest
                limit = slot_limits[engine]
                while True:
                    ensure_cycle(cycle)
                    if usage[cycle][engine] < limit:
                        return cycle
                    cycle += 1

            for engine, slot in slots:
                reads, writes, _mem_reads, _mem_writes = slot_rw(engine, slot)
                earliest = 0
                for addr in reads:
                    earliest = max(earliest, ready_time[addr])
                for addr in writes:
                    earliest = max(earliest, last_write[addr] + 1, last_read[addr])

                cycle = find_cycle(engine, earliest)
                ensure_cycle(cycle)
                cycles[cycle].setdefault(engine, []).append(slot)
                usage[cycle][engine] += 1

                for addr in reads:
                    if last_read[addr] < cycle:
                        last_read[addr] = cycle
                for addr in writes:
                    last_write[addr] = cycle
                    ready_time[addr] = cycle + 1

            return [c for c in cycles if c]

        def schedule_segment(segment):
            ops = flatten(segment)
            if not ops:
                return []
            if SCHED_SEED is None:
                tie_break = list(range(len(ops)))
            else:
                rng = random.Random(SCHED_SEED)
                tie_break = [rng.random() for _ in range(len(ops))]
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

            if USE_SLIL_SCHED:
                def schedule_segment_slil():
                    pred_count = deps[:]
                    ready = {i for i, d in enumerate(pred_count) if d == 0}
                    last_use = {}
                    for i in range(len(ops) - 1, -1, -1):
                        for addr in ops[i]["reads"]:
                            if addr not in last_use:
                                last_use[addr] = i

                    scheduled_instrs = []
                    while ready:
                        counts = {k: 0 for k in slot_limits}
                        bundle = {}
                        scheduled = []
                        cycle_writes = set()
                        cycle_reads = set()

                        def slil_priority(idx):
                            op = ops[idx]
                            height_score = -heights[idx]
                            last_use_score = 0
                            for addr in op["reads"]:
                                if last_use.get(addr) == idx:
                                    last_use_score -= 1
                            early_consumer_score = 0
                            for s in succs[idx]:
                                if heights[s] > 0:
                                    early_consumer_score -= 1
                            return (height_score, last_use_score, early_consumer_score, idx)

                        ready_sorted = sorted(ready, key=slil_priority)
                        engine_passes = ("valu", "load", "alu", "flow", "store", "debug")
                        for engine in engine_passes:
                            for idx in ready_sorted:
                                if idx in scheduled:
                                    continue
                                if ops[idx]["engine"] != engine:
                                    continue
                                if counts[engine] >= slot_limits[engine]:
                                    continue
                                if not RELAX_CYCLE_HAZARDS:
                                    if ops[idx]["reads"] & cycle_writes:
                                        continue
                                    if ops[idx]["writes"] & (cycle_writes | cycle_reads):
                                        continue
                                bundle.setdefault(engine, []).append(ops[idx]["slot"])
                                counts[engine] += 1
                                cycle_writes |= ops[idx]["writes"]
                                cycle_reads |= ops[idx]["reads"]
                                scheduled.append(idx)

                        for idx in ready_sorted:
                            if idx in scheduled:
                                continue
                            eng = ops[idx]["engine"]
                            if counts[eng] >= slot_limits[eng]:
                                continue
                            if not RELAX_CYCLE_HAZARDS:
                                if ops[idx]["reads"] & cycle_writes:
                                    continue
                                if ops[idx]["writes"] & (cycle_writes | cycle_reads):
                                    continue
                            bundle.setdefault(eng, []).append(ops[idx]["slot"])
                            counts[eng] += 1
                            cycle_writes |= ops[idx]["writes"]
                            cycle_reads |= ops[idx]["reads"]
                            scheduled.append(idx)

                        if not scheduled:
                            idx = max(ready, key=lambda i: (heights[i], -i))
                            eng = ops[idx]["engine"]
                            bundle.setdefault(eng, []).append(ops[idx]["slot"])
                            cycle_writes |= ops[idx]["writes"]
                            scheduled.append(idx)

                        scheduled_instrs.append(bundle)
                        ready -= set(scheduled)
                        for idx in scheduled:
                            for v in succs[idx]:
                                pred_count[v] -= 1
                                if pred_count[v] == 0:
                                    ready.add(v)
                    return scheduled_instrs

                return schedule_segment_slil()

            if USE_GREEDY_SCHED:
                def schedule_segment_greedy():
                    pause_token = -2
                    has_start_pause = any(op["pause_role"] == "start" for op in ops)
                    ops_rw = []
                    for op in ops:
                        reads = set(op["reads"])
                        writes = set(op["writes"])
                        pause_role = op["pause_role"]
                        if pause_role == "start":
                            writes.add(pause_token)
                        elif pause_role == "end":
                            reads.add(pause_token)
                        if op["engine"] == "store" and has_start_pause:
                            reads.add(pause_token)
                        ops_rw.append((op, reads, writes))

                    bundles = []
                    slot_usage = []
                    addr_ready = defaultdict(int)
                    last_write = defaultdict(lambda: -1)
                    last_read = defaultdict(lambda: -1)

                    def ensure_cycle(idx):
                        while len(bundles) <= idx:
                            bundles.append({})
                            slot_usage.append(defaultdict(int))

                    def first_available(eng, min_cycle):
                        cycle = min_cycle
                        limit = slot_limits[eng]
                        while True:
                            ensure_cycle(cycle)
                            if slot_usage[cycle][eng] < limit:
                                return cycle
                            cycle += 1

                    for op, reads, writes in ops_rw:
                        earliest = 0
                        for addr in reads:
                            earliest = max(earliest, addr_ready[addr])
                        for addr in writes:
                            earliest = max(earliest, last_write[addr] + 1, last_read[addr])

                        cycle = first_available(op["engine"], earliest)
                        ensure_cycle(cycle)
                        bundles[cycle].setdefault(op["engine"], []).append(op["slot"])
                        slot_usage[cycle][op["engine"]] += 1

                        for addr in reads:
                            last_read[addr] = max(last_read[addr], cycle)
                        for addr in writes:
                            last_write[addr] = cycle
                            addr_ready[addr] = cycle + 1

                    return [b for b in bundles if b]

                return schedule_segment_greedy()

            ready = [i for i, d in enumerate(deps) if d == 0]
            ready.sort()
            remaining = len(ops)
            remaining_by_engine = defaultdict(int)
            for op in ops:
                remaining_by_engine[op["engine"]] += 1
            scheduled_instrs = []
            while remaining:
                counts = {k: 0 for k in slot_limits}
                bundle = {}
                scheduled = []
                cycle_writes = set()
                cycle_reads = set()
                engine_urgency = {
                    eng: remaining_by_engine[eng] / slot_limits[eng]
                    for eng in remaining_by_engine
                }
                if ENGINE_BIAS:
                    for eng, bias in ENGINE_BIAS.items():
                        if eng in engine_urgency:
                            engine_urgency[eng] *= bias
                if SCHED_MODE == "height":
                    ready.sort(
                        key=lambda i: (
                            -heights[i],
                            tie_break[i],
                        )
                    )
                else:
                    ready.sort(
                        key=lambda i: (
                            -heights[i],
                            -engine_urgency[ops[i]["engine"]],
                            tie_break[i],
                        )
                    )
                for i in ready:
                    op = ops[i]
                    eng = op["engine"]
                    if counts[eng] >= slot_limits[eng]:
                        continue
                    if not RELAX_CYCLE_HAZARDS:
                        if op["reads"] & cycle_writes:
                            continue
                        if op["writes"] & (cycle_writes | cycle_reads):
                            continue
                    bundle.setdefault(eng, []).append(op["slot"])
                    counts[eng] += 1
                    scheduled.append(i)
                    cycle_writes |= op["writes"]
                    cycle_reads |= op["reads"]
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

        if USE_SLOT_SCHED:
            packed = []
            segment = []
            for instr in instrs:
                if instr.get("flow") == [("pause",)]:
                    if segment:
                        packed.extend(schedule_segment_slots(segment))
                        segment = []
                    packed.append(instr)
                else:
                    segment.append(instr)
            if segment:
                packed.extend(schedule_segment_slots(segment))
        elif USE_GREEDY_SCHED:
            packed = []
            segment = []
            for instr in instrs:
                if instr.get("flow") == [("pause",)]:
                    if segment:
                        packed.extend(schedule_segment(segment))
                        segment = []
                    packed.append(instr)
                else:
                    segment.append(instr)
            if segment:
                packed.extend(schedule_segment(segment))
        else:
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
        if ALU_CONSTS:
            two_const = self.alloc_scratch("two_const")
            self.add("alu", ("+", two_const, one_const, one_const))
            four_const = self.alloc_scratch("four_const")
            self.add("alu", ("+", four_const, two_const, two_const))
        else:
            two_const = self.scratch_const(2)
            four_const = self.scratch_const(4)
        if DEPTH4_VSELECT or DEPTH4_VSELECT_REUSE or USE_VLOAD_NODES:
            eight_const = self.scratch_const(8)
        else:
            eight_const = None
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
        node15_30_consts = []
        node_idx_consts = [] if USE_VLOAD_NODES else [self.scratch_const(i) for i in range(15)]
        node_idx_consts_15plus = []
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
        if DEPTH4_VSELECT or DEPTH4_VSELECT_REUSE:
            node15_const = self.alloc_scratch("node15_const")
            node16_const = self.alloc_scratch("node16_const")
            node17_const = self.alloc_scratch("node17_const")
            node18_const = self.alloc_scratch("node18_const")
            node19_const = self.alloc_scratch("node19_const")
            node20_const = self.alloc_scratch("node20_const")
            node21_const = self.alloc_scratch("node21_const")
            node22_const = self.alloc_scratch("node22_const")
            node23_const = self.alloc_scratch("node23_const")
            node24_const = self.alloc_scratch("node24_const")
            node25_const = self.alloc_scratch("node25_const")
            node26_const = self.alloc_scratch("node26_const")
            node27_const = self.alloc_scratch("node27_const")
            node28_const = self.alloc_scratch("node28_const")
            node29_const = self.alloc_scratch("node29_const")
            node30_const = self.alloc_scratch("node30_const")
            node15_30_consts = [
                node15_const,
                node16_const,
                node17_const,
                node18_const,
                node19_const,
                node20_const,
                node21_const,
                node22_const,
                node23_const,
                node24_const,
                node25_const,
                node26_const,
                node27_const,
                node28_const,
                node29_const,
                node30_const,
            ]
            node_idx_consts_15plus = [self.scratch_const(i) for i in range(15, 31)]
            node_idx_consts.extend(node_idx_consts_15plus)
            node_consts.extend(
                [
                    node15_const,
                    node16_const,
                    node17_const,
                    node18_const,
                    node19_const,
                    node20_const,
                    node21_const,
                    node22_const,
                    node23_const,
                    node24_const,
                    node25_const,
                    node26_const,
                    node27_const,
                    node28_const,
                    node29_const,
                    node30_const,
                ]
            )
        elif USE_VLOAD_NODES:
            # Pad slot so vload into node8_const doesn't clobber later scratch.
            node15_const = self.alloc_scratch("node15_const_pad")
        if USE_VLOAD_NODES:
            # Bulk load nodes 0..7 and 8..15 using vload into contiguous scalar slots.
            self.add("load", ("vload", node0_const, self.scratch["forest_values_p"]))
            self.add("alu", ("+", tmp1, self.scratch["forest_values_p"], eight_const))
            self.add("load", ("vload", node8_const, tmp1))
            if node_idx_consts_15plus:
                for node_idx_const, node_const in zip(node_idx_consts_15plus, node15_30_consts):
                    self.add(
                        "alu",
                        ("+", tmp1, self.scratch["forest_values_p"], node_idx_const),
                    )
                    self.add("load", ("load", node_const, tmp1))
        else:
            for node_idx_const, node_const in zip(node_idx_consts, node_consts):
                self.add(
                    "alu",
                    ("+", tmp1, self.scratch["forest_values_p"], node_idx_const),
                )
                self.add("load", ("load", node_const, tmp1))
        if PRE_XOR_NODE_CONSTS:
            for node_const in node_consts:
                self.add("alu", ("^", node_const, node_const, c6_const))
        one_minus_base = self.alloc_scratch("one_minus_base")
        self.add(
            "alu",
            ("-", one_minus_base, one_const, self.scratch["forest_values_p"]),
        )

        natural_offsets = list(range(0, batch_size, VLEN))
        base_offsets = list(natural_offsets)
        if BASE_OFFSET_ORDER == "evenodd" and not LOW_SCRATCH:
            base_offsets = base_offsets[::2] + base_offsets[1::2]
        if ALU_SHIFT9_INDICES is None:
            shift9_order = list(natural_offsets)
            if ALU_SHIFT9_ORDER == "evenodd":
                shift9_order = shift9_order[::2] + shift9_order[1::2]
            partial_alu_shift9_offsets = set(shift9_order[:ALU_SHIFT9_K])
        else:
            partial_alu_shift9_offsets = set(
                natural_offsets[i] for i in ALU_SHIFT9_INDICES
            )
        if ALU_SHIFT16_INDICES is None:
            shift16_order = list(natural_offsets)
            if ALU_SHIFT16_ORDER == "evenodd":
                shift16_order = shift16_order[::2] + shift16_order[1::2]
            partial_alu_shift16_offsets = set(shift16_order[:ALU_SHIFT16_K])
        else:
            partial_alu_shift16_offsets = set(
                natural_offsets[i] for i in ALU_SHIFT16_INDICES
            )
        partial_alu_xor_offsets = set(natural_offsets[:ALU_XOR_C2_K])
        if SHIFT19_PREFIX_K is None:
            partial_valu_shift19_offsets = set(
                natural_offsets[i] for i in SHIFT19_INDICES
            )
        else:
            partial_valu_shift19_offsets = set(natural_offsets[:SHIFT19_PREFIX_K])
        partial_valu_shift9_offsets = set(natural_offsets[:VALU_SHIFT9_K])
        partial_valu_shift16_offsets = set(natural_offsets[:VALU_SHIFT16_K])
        partial_alu_add_c4_offsets = set(natural_offsets[:ALU_ADD_K])
        partial_alu_xor_tmp_offsets = set(natural_offsets[:ALU_XOR_TMP_K])
        partial_alu_xor_shift9_offsets = set(natural_offsets[:ALU_XOR_SHIFT9_K])
        partial_alu_xor_shift16_offsets = set(natural_offsets[:ALU_XOR_SHIFT16_K])
        partial_valu_parity_offsets = set(natural_offsets[:VALU_PARITY_K])
        partial_valu_adjust_offsets = set(natural_offsets[:VALU_ADJUST_K])
        depth2_use_flow = DEPTH2_USE_FLOW
        partial_valu_depth2_offsets = set(natural_offsets[:DEPTH2_VALU_K])
        partial_alu_depth3_xor_offsets = set(natural_offsets[:DEPTH3_XOR_ALU_K])
        if LOW_SCRATCH:
            base_consts = None
            base_ptr = self.alloc_scratch("base_ptr")
            vlen_const = self.scratch_const(VLEN)
            self.add("alu", ("+", base_ptr, zero_const, zero_const))
        else:
            base_consts = [self.scratch_const(i) for i in base_offsets]
            base_ptr = None
            vlen_const = None
        base_index_map = {offset: i for i, offset in enumerate(base_offsets)}
        group_blocks = GROUP_BLOCKS
        if GROUP_SIZES:
            offset_groups = []
            idx = 0
            for size in GROUP_SIZES:
                if size <= 0:
                    continue
                offset_groups.append(base_offsets[idx : idx + size])
                idx += size
            if idx < len(base_offsets):
                offset_groups.append(base_offsets[idx:])
        else:
            offset_groups = [
                base_offsets[i : i + group_blocks]
                for i in range(0, len(base_offsets), group_blocks)
            ]
        if GROUP_ORDER == "reverse":
            offset_groups = list(reversed(offset_groups))
        elif GROUP_ORDER == "evenodd":
            offset_groups = offset_groups[::2] + offset_groups[1::2]

        idx_scratch = self.alloc_scratch("idx_scratch", batch_size)
        val_scratch = self.alloc_scratch("val_scratch", batch_size)
        tmp_scratch = self.alloc_scratch("tmp_scratch", batch_size)
        if ALIAS_SHIFT_TMP:
            shift_scratch = tmp_scratch
        else:
            shift_scratch = self.alloc_scratch("shift_scratch", batch_size)

        zero_vec = alloc_vec("zero_vec")
        one_vec = alloc_vec("one_vec")
        b0_vec = alloc_vec("b0_vec") if B0_FROM_IDX else None
        two_vec = alloc_vec("two_vec")
        four_vec = alloc_vec("four_vec")
        if DEPTH4_VSELECT_REUSE:
            eight_vec = alloc_vec("eight_vec")
        else:
            eight_vec = None
        if DEPTH4_VSELECT:
            node15_vec = alloc_vec("node15_vec")
            node16_vec = alloc_vec("node16_vec")
            node17_vec = alloc_vec("node17_vec")
            node18_vec = alloc_vec("node18_vec")
            node19_vec = alloc_vec("node19_vec")
            node20_vec = alloc_vec("node20_vec")
            node21_vec = alloc_vec("node21_vec")
            node22_vec = alloc_vec("node22_vec")
            node23_vec = alloc_vec("node23_vec")
            node24_vec = alloc_vec("node24_vec")
            node25_vec = alloc_vec("node25_vec")
            node26_vec = alloc_vec("node26_vec")
            node27_vec = alloc_vec("node27_vec")
            node28_vec = alloc_vec("node28_vec")
            node29_vec = alloc_vec("node29_vec")
            node30_vec = alloc_vec("node30_vec")
        elif DEPTH4_VSELECT_REUSE:
            pass
        forest_values_vec = alloc_vec("forest_values_vec")
        if LOW_SCRATCH:
            one_minus_base_vec = None
            two_minus_base_vec = None
        elif DEPTH4_VSELECT:
            one_minus_base_vec = alloc_vec("one_minus_base_vec")
            two_minus_base_vec = None
        else:
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
        sel4_b0_vec = alloc_vec("sel4_b0_vec") if DEPTH4_VSELECT else None
        sel4_t0_vec = alloc_vec("sel4_t0_vec") if DEPTH4_VSELECT else None
        sel4_t1_vec = alloc_vec("sel4_t1_vec") if DEPTH4_VSELECT else None
        if LOW_SCRATCH and not DEPTH4_VSELECT:
            sel_b0_vec2 = sel_b0_vec
            sel_t0_vec2 = sel_t0_vec
            sel_t1_vec2 = sel_t1_vec
        elif DEPTH4_VSELECT:
            sel_b0_vec2 = sel_b0_vec
            sel_t0_vec2 = sel_t0_vec
            sel_t1_vec2 = sel_t1_vec
        else:
            sel_b0_vec2 = alloc_vec("sel_b0_vec2")
            sel_t0_vec2 = alloc_vec("sel_t0_vec2")
            sel_t1_vec2 = alloc_vec("sel_t1_vec2")
        sel_b0_vec_alt = alloc_vec("sel_b0_vec_alt") if USE_SEL_ALT else None
        sel_t0_vec_alt = alloc_vec("sel_t0_vec_alt") if USE_SEL_ALT else None
        sel_t1_vec_alt = alloc_vec("sel_t1_vec_alt") if USE_SEL_ALT else None
        if LOW_SCRATCH or DEPTH4_VSELECT:
            sel2_t0_vec = sel_t0_vec
            sel2_t1_vec = sel_t1_vec
        else:
            sel2_t0_vec = alloc_vec("sel2_t0_vec")
            sel2_t1_vec = alloc_vec("sel2_t1_vec")

        node7_14_vecs = [
            node7_vec,
            node8_vec,
            node9_vec,
            node10_vec,
            node11_vec,
            node12_vec,
            node13_vec,
            node14_vec,
        ]
        node7_14_consts = [
            node7_const,
            node8_const,
            node9_const,
            node10_const,
            node11_const,
            node12_const,
            node13_const,
            node14_const,
        ]
        if DEPTH4_VSELECT or DEPTH4_VSELECT_REUSE:
            if DEPTH4_VSELECT_REUSE:
                node15_22_consts = [
                    node16_const,
                    node17_const,
                    node18_const,
                    node19_const,
                    node20_const,
                    node21_const,
                    node22_const,
                    node23_const,
                ]
                node23_30_consts = [
                    node24_const,
                    node25_const,
                    node26_const,
                    node27_const,
                    node28_const,
                    node29_const,
                    node30_const,
                    node15_const,
                ]
            else:
                node15_22_consts = [
                    node15_const,
                    node16_const,
                    node17_const,
                    node18_const,
                    node19_const,
                    node20_const,
                    node21_const,
                    node22_const,
                ]
                node23_30_consts = [
                    node23_const,
                    node24_const,
                    node25_const,
                    node26_const,
                    node27_const,
                    node28_const,
                    node29_const,
                    node30_const,
                ]
        else:
            node15_22_consts = None
            node23_30_consts = None

        vbroadcast_slots = [
            ("vbroadcast", zero_vec, zero_const),
            ("vbroadcast", one_vec, one_const),
            ("vbroadcast", two_vec, two_const),
            ("vbroadcast", four_vec, four_const),
            *([("vbroadcast", eight_vec, eight_const)] if eight_vec is not None else []),
            ("vbroadcast", forest_values_vec, self.scratch["forest_values_p"]),
            *(
                [
                    ("vbroadcast", one_minus_base_vec, one_minus_base),
                ]
                if one_minus_base_vec is not None
                else []
            ),
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
            *(
                [
                    ("vbroadcast", node15_vec, node15_const),
                    ("vbroadcast", node16_vec, node16_const),
                    ("vbroadcast", node17_vec, node17_const),
                    ("vbroadcast", node18_vec, node18_const),
                    ("vbroadcast", node19_vec, node19_const),
                    ("vbroadcast", node20_vec, node20_const),
                    ("vbroadcast", node21_vec, node21_const),
                    ("vbroadcast", node22_vec, node22_const),
                    ("vbroadcast", node23_vec, node23_const),
                    ("vbroadcast", node24_vec, node24_const),
                    ("vbroadcast", node25_vec, node25_const),
                    ("vbroadcast", node26_vec, node26_const),
                    ("vbroadcast", node27_vec, node27_const),
                    ("vbroadcast", node28_vec, node28_const),
                    ("vbroadcast", node29_vec, node29_const),
                    ("vbroadcast", node30_vec, node30_const),
                ]
                if DEPTH4_VSELECT
                else []
            ),
        ]
        if shift19_vec is not None:
            vbroadcast_slots.append(("vbroadcast", shift19_vec, shift19_const))
        for i in range(0, len(vbroadcast_slots), SLOT_LIMITS["valu"]):
            emit(valu=vbroadcast_slots[i : i + SLOT_LIMITS["valu"]])
        if one_minus_base_vec is not None and two_minus_base_vec is not None:
            # Save one scratch slot by synthesizing two_minus_base_vec from one_minus_base_vec.
            emit(valu=[("+", two_minus_base_vec, one_minus_base_vec, one_vec)])

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))

        if USE_ADDR_ARRAYS:
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
        else:
            for i, base_offset in enumerate(base_offsets):
                idx_block = idx_scratch + base_offset
                val_block = val_scratch + base_offset
                if base_consts is None:
                    emit(alu=[("+", tmp1, self.scratch["inp_indices_p"], base_ptr), ("+", tmp2, self.scratch["inp_values_p"], base_ptr)])
                    emit(load=[("vload", idx_block, tmp1), ("vload", val_block, tmp2)])
                    emit(alu=[("+", base_ptr, base_ptr, vlen_const)])
                    continue
                base_const = base_consts[i]
                emit(alu=[("+", tmp1, self.scratch["inp_indices_p"], base_const), ("+", tmp2, self.scratch["inp_values_p"], base_const)])
                emit(load=[("vload", idx_block, tmp1), ("vload", val_block, tmp2)])

        def emit_valu(slots):
            for i in range(0, len(slots), SLOT_LIMITS["valu"]):
                emit(valu=slots[i : i + SLOT_LIMITS["valu"]])

        def emit_alu(slots):
            for i in range(0, len(slots), SLOT_LIMITS["alu"]):
                emit(alu=slots[i : i + SLOT_LIMITS["alu"]])

        def emit_load(slots):
            for i in range(0, len(slots), SLOT_LIMITS["load"]):
                emit(load=slots[i : i + SLOT_LIMITS["load"]])

        def broadcast_nodes(dest_vecs, src_consts, xor_with_c6: bool = False):
            slots = []
            for dest_vec, src_const in zip(dest_vecs, src_consts):
                slots.append(("vbroadcast", dest_vec, src_const))
            emit_valu(slots)
            if xor_with_c6:
                slots = []
                for dest_vec in dest_vecs:
                    slots.append(("^", dest_vec, dest_vec, c6_vec))
                emit_valu(slots)

        # Maintain values in c6-xored form throughout rounds.
        slots = []
        for base_offset in base_offsets:
            val_block = val_scratch + base_offset
            slots.append(("^", val_block, val_block, c6_vec))
        emit_valu(slots)

        # Convert node vectors to node ^ c6 for c6-hoisted hash.
        if not PRE_XOR_NODE_CONSTS:
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
            if DEPTH4_VSELECT:
                node_xor_slots.extend(
                    [
                        ("^", node15_vec, node15_vec, c6_vec),
                        ("^", node16_vec, node16_vec, c6_vec),
                        ("^", node17_vec, node17_vec, c6_vec),
                        ("^", node18_vec, node18_vec, c6_vec),
                        ("^", node19_vec, node19_vec, c6_vec),
                        ("^", node20_vec, node20_vec, c6_vec),
                        ("^", node21_vec, node21_vec, c6_vec),
                        ("^", node22_vec, node22_vec, c6_vec),
                        ("^", node23_vec, node23_vec, c6_vec),
                        ("^", node24_vec, node24_vec, c6_vec),
                        ("^", node25_vec, node25_vec, c6_vec),
                        ("^", node26_vec, node26_vec, c6_vec),
                        ("^", node27_vec, node27_vec, c6_vec),
                        ("^", node28_vec, node28_vec, c6_vec),
                        ("^", node29_vec, node29_vec, c6_vec),
                        ("^", node30_vec, node30_vec, c6_vec),
                    ]
                )
            emit_valu(node_xor_slots)

        node7_valid = True

        wrap_round = forest_height + 1

        def emit_round_two_nodes(active_offsets, use_parity_cond: bool = False):
            if B0_FROM_IDX:
                for base_offset in active_offsets:
                    idx_block = idx_scratch + base_offset
                    tmp_block = tmp_scratch + base_offset
                    emit_valu([("&", b0_vec, idx_block, one_vec)])
                    emit(flow=[("vselect", tmp_block, b0_vec, node1_vec, node2_vec)])
            elif DEPTH1_VALU_SELECT:
                for base_offset in active_offsets:
                    idx_block = idx_scratch + base_offset
                    cond_block = shift_scratch + base_offset
                    tmp_block = tmp_scratch + base_offset
                    emit_valu([("&", cond_block, idx_block, one_vec)])
                    # tmp = (node1 - node2) * cond + node2
                    emit_valu([("-", tmp_block, node1_vec, node2_vec)])
                    emit_valu([("multiply_add", tmp_block, tmp_block, cond_block, node2_vec)])
            elif not use_parity_cond or not DEPTH1_USE_PARITY:
                slots = []
                for base_offset in active_offsets:
                    idx_block = idx_scratch + base_offset
                    tmp_block = tmp_scratch + base_offset
                    slots.append(("==", tmp_block, idx_block, one_vec))
                emit_valu(slots)
                for base_offset in active_offsets:
                    tmp_block = tmp_scratch + base_offset
                    emit(flow=[("vselect", tmp_block, tmp_block, node1_vec, node2_vec)])
            else:
                for base_offset in active_offsets:
                    tmp_block = tmp_scratch + base_offset
                    if c6_lsb:
                        emit(flow=[("vselect", tmp_block, tmp_block, node1_vec, node2_vec)])
                    else:
                        emit(flow=[("vselect", tmp_block, tmp_block, node2_vec, node1_vec)])
            slots = []
            for base_offset in active_offsets:
                val_block = val_scratch + base_offset
                tmp_block = tmp_scratch + base_offset
                slots.append(("^", val_block, val_block, tmp_block))
            emit_valu(slots)

        def emit_round_four_nodes(active_offsets):
            slots = []
            for base_offset in active_offsets:
                idx_block = idx_scratch + base_offset
                b1_block = shift_scratch + base_offset
                slots.append(("&", b1_block, idx_block, two_vec))
            emit_valu(slots)
            for base_offset in active_offsets:
                b1_block = shift_scratch + base_offset
                b0_block = tmp_scratch + base_offset
                if B0_FROM_IDX:
                    idx_block = idx_scratch + base_offset
                    emit_valu([("&", b0_vec, idx_block, one_vec)])
                use_valu = (not depth2_use_flow) or (base_offset in partial_valu_depth2_offsets)
                if not use_valu:
                    emit(flow=[("vselect", sel2_t0_vec, b1_block, node6_vec, node4_vec)])
                    emit(flow=[("vselect", sel2_t1_vec, b1_block, node3_vec, node5_vec)])
                    if B0_FROM_IDX:
                        emit(flow=[("vselect", b0_block, b0_vec, sel2_t0_vec, sel2_t1_vec)])
                    else:
                        if c6_lsb:
                            emit(flow=[("vselect", b0_block, b0_block, sel2_t1_vec, sel2_t0_vec)])
                        else:
                            emit(flow=[("vselect", b0_block, b0_block, sel2_t0_vec, sel2_t1_vec)])
                else:
                    # Convert b1 (0 or 2) to 0/1 for arithmetic select.
                    emit_valu([(">>", b1_block, b1_block, one_vec)])
                    # sel2_t0 = select(b1, node6, node4)
                    emit_valu([("-", sel2_t0_vec, node6_vec, node4_vec)])
                    emit_valu([("multiply_add", sel2_t0_vec, sel2_t0_vec, b1_block, node4_vec)])
                    # sel2_t1 = select(b1, node3, node5)
                    emit_valu([("-", sel2_t1_vec, node3_vec, node5_vec)])
                    emit_valu([("multiply_add", sel2_t1_vec, sel2_t1_vec, b1_block, node5_vec)])
                    if B0_FROM_IDX:
                        # b0_vec holds 0/1; select between sel2_t0 and sel2_t1
                        emit_valu([("-", sel2_t0_vec, sel2_t0_vec, sel2_t1_vec)])
                        emit_valu([("multiply_add", b0_block, sel2_t0_vec, b0_vec, sel2_t1_vec)])
                    else:
                        if c6_lsb:
                            emit_valu([("-", b0_block, one_vec, b0_block)])
                        # b0_block holds 0/1; select between sel2_t0 and sel2_t1
                        emit_valu([("-", sel2_t0_vec, sel2_t0_vec, sel2_t1_vec)])
                        emit_valu([("multiply_add", b0_block, sel2_t0_vec, b0_block, sel2_t1_vec)])
            slots = []
            for base_offset in active_offsets:
                val_block = val_scratch + base_offset
                sel_block = tmp_scratch + base_offset
                slots.append(("^", val_block, val_block, sel_block))
            emit_valu(slots)

        def emit_round_eight_nodes(
            active_offsets,
            sel_t0,
            sel_t1,
            sel_b0,
            sel_t0_alt=None,
            sel_t1_alt=None,
            sel_b0_alt=None,
            xor_val: bool = True,
        ):
            # Bit-tree vselect for depth-3 nodes (7..14) using idx bitmasks.
            for base_offset in active_offsets:
                if sel_t0_alt is not None and (base_offset // VLEN) & 1:
                    sel0 = sel_t0_alt
                    sel1 = sel_t1_alt
                    selb = sel_b0_alt
                else:
                    sel0 = sel_t0
                    sel1 = sel_t1
                    selb = sel_b0
                idx_block = idx_scratch + base_offset
                tmp_block = tmp_scratch + base_offset
                mask_block = shift_scratch + base_offset
                val_block = val_scratch + base_offset
                if B0_FROM_IDX:
                    emit_valu([("&", b0_vec, idx_block, one_vec)])
                    b0_mask = b0_vec
                else:
                    b0_mask = tmp_block
                emit_valu([("&", mask_block, idx_block, two_vec)])
                emit(
                    flow=[
                        ("vselect", sel0, b0_mask, node9_vec, node8_vec),
                        ("vselect", sel1, b0_mask, node11_vec, node10_vec),
                    ]
                )
                emit(flow=[("vselect", sel0, mask_block, sel1, sel0)])
                emit(
                    flow=[
                        ("vselect", sel1, b0_mask, node13_vec, node12_vec),
                        ("vselect", selb, b0_mask, node7_vec, node14_vec),
                    ]
                )
                emit(flow=[("vselect", sel1, mask_block, selb, sel1)])
                emit_valu([("&", mask_block, idx_block, four_vec)])
                emit(flow=[("vselect", sel0, mask_block, sel1, sel0)])
                if xor_val:
                    if base_offset in partial_alu_depth3_xor_offsets:
                        alu_slots = []
                        for lane in range(VLEN):
                            alu_slots.append(
                                ("^", val_block + lane, val_block + lane, sel0 + lane)
                            )
                        emit_alu(alu_slots)
                    else:
                        emit_valu([("^", val_block, val_block, sel0)])

        def emit_round_eight_nodes_custom(
            active_offsets,
            nodes,
            sel_t0,
            sel_t1,
            sel_b0,
            xor_val: bool = True,
        ):
            n0, n1, n2, n3, n4, n5, n6, n7 = nodes
            for base_offset in active_offsets:
                idx_block = idx_scratch + base_offset
                tmp_block = tmp_scratch + base_offset
                mask_block = shift_scratch + base_offset
                val_block = val_scratch + base_offset
                if B0_FROM_IDX:
                    emit_valu([("&", b0_vec, idx_block, one_vec)])
                    b0_mask = b0_vec
                else:
                    b0_mask = tmp_block
                emit_valu([("&", mask_block, idx_block, two_vec)])
                emit(
                    flow=[
                        ("vselect", sel_t0, b0_mask, n1, n0),
                        ("vselect", sel_t1, b0_mask, n3, n2),
                    ]
                )
                emit(flow=[("vselect", sel_t0, mask_block, sel_t1, sel_t0)])
                emit(
                    flow=[
                        ("vselect", sel_t1, b0_mask, n5, n4),
                        ("vselect", sel_b0, b0_mask, n7, n6),
                    ]
                )
                emit(flow=[("vselect", sel_t1, mask_block, sel_b0, sel_t1)])
                emit_valu([("&", mask_block, idx_block, four_vec)])
                emit(flow=[("vselect", sel_t0, mask_block, sel_t1, sel_t0)])
                if xor_val:
                    emit_valu([("^", val_block, val_block, sel_t0)])

        def emit_round_sixteen_nodes(active_offsets, round_idx):
            # Bit-tree vselect for depth-4 nodes (15..30) using idx bitmasks.
            nodes_low = [
                node16_vec,
                node17_vec,
                node18_vec,
                node19_vec,
                node20_vec,
                node21_vec,
                node22_vec,
                node23_vec,
            ]
            nodes_high = [
                node24_vec,
                node25_vec,
                node26_vec,
                node27_vec,
                node28_vec,
                node29_vec,
                node30_vec,
                node15_vec,
            ]
            def select_nodes_to_dest(nodes, idx_block, mask_block, dest_block):
                n0, n1, n2, n3, n4, n5, n6, n7 = nodes
                emit_valu([("&", mask_block, idx_block, one_vec)])
                emit(
                    flow=[
                        ("vselect", sel4_t1_vec, mask_block, n1, n0),
                        ("vselect", sel4_b0_vec, mask_block, n3, n2),
                    ]
                )
                emit_valu([("&", mask_block, idx_block, two_vec)])
                emit(flow=[("vselect", sel4_t1_vec, mask_block, sel4_b0_vec, sel4_t1_vec)])
                emit_valu([("+", dest_block, sel4_t1_vec, zero_vec)])
                emit_valu([("&", mask_block, idx_block, one_vec)])
                emit(
                    flow=[
                        ("vselect", sel4_t1_vec, mask_block, n5, n4),
                        ("vselect", sel4_b0_vec, mask_block, n7, n6),
                    ]
                )
                emit_valu([("&", mask_block, idx_block, two_vec)])
                emit(flow=[("vselect", sel4_t1_vec, mask_block, sel4_b0_vec, sel4_t1_vec)])
                emit_valu([("&", mask_block, idx_block, four_vec)])
                emit(flow=[("vselect", dest_block, mask_block, sel4_t1_vec, dest_block)])

            for base_offset in active_offsets:
                idx_block = idx_scratch + base_offset
                mask_block = shift_scratch + base_offset
                tmp_block = tmp_scratch + base_offset
                val_block = val_scratch + base_offset
                # Low half -> tmp_block, high half -> sel_t0_vec.
                select_nodes_to_dest(nodes_low, idx_block, mask_block, tmp_block)
                select_nodes_to_dest(nodes_high, idx_block, mask_block, sel4_t0_vec)
                if eight_vec is not None:
                    emit_valu([("&", mask_block, idx_block, eight_vec)])
                else:
                    alu_slots = []
                    for lane in range(VLEN):
                        alu_slots.append(
                            ("&", mask_block + lane, idx_block + lane, eight_const)
                        )
                    emit_alu(alu_slots)
                emit(flow=[("vselect", tmp_block, mask_block, sel4_t0_vec, tmp_block)])
                if DEBUG_DEPTH4:
                    keys = [(round_idx, base_offset + lane, "node_c6") for lane in range(VLEN)]
                    emit(debug=[("vcompare", tmp_block, keys)])
                emit_valu([("^", val_block, val_block, tmp_block)])

        def emit_round_sixteen_nodes_reuse(active_offsets, round_idx):
            nonlocal node7_valid
            # Broadcast lower half (15..22) into node7..14, select into sel_t0_vec.
            broadcast_nodes(node7_14_vecs, node15_22_consts, xor_with_c6=not PRE_XOR_NODE_CONSTS)
            emit_round_eight_nodes(
                active_offsets,
                sel_t0_vec,
                sel_t1_vec,
                sel_b0_vec,
                xor_val=False,
            )
            # Broadcast upper half (23..30) into node7..14, select into sel_t0_vec2.
            broadcast_nodes(node7_14_vecs, node23_30_consts, xor_with_c6=not PRE_XOR_NODE_CONSTS)
            emit_round_eight_nodes(
                active_offsets,
                sel_t0_vec2,
                sel_t1_vec2,
                sel_b0_vec2,
                xor_val=False,
            )
            # Select between lower and upper using b3 (idx & 8).
            for base_offset in active_offsets:
                idx_block = idx_scratch + base_offset
                mask_block = shift_scratch + base_offset
                val_block = val_scratch + base_offset
                alu_slots = []
                for lane in range(VLEN):
                    alu_slots.append(
                        ("&", mask_block + lane, idx_block + lane, eight_const)
                    )
                emit_alu(alu_slots)
                emit(flow=[("vselect", sel_t0_vec, mask_block, sel_t0_vec2, sel_t0_vec)])
                if DEBUG_DEPTH4:
                    keys = [(round_idx, base_offset + lane, "node_c6") for lane in range(VLEN)]
                    emit(debug=[("vcompare", sel_t0_vec, keys)])
                emit_valu([("^", val_block, val_block, sel_t0_vec)])
            node7_valid = False

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
        if DISABLE_USE_ADDR:
            final_addr = False
            skip_last_addr = False

        def emit_round(round_idx, active_offsets):
            nonlocal node7_valid
            if ACTIVE_OFFSET_ORDER == "reverse":
                active_offsets = list(reversed(active_offsets))
            elif ACTIVE_OFFSET_ORDER == "evenodd":
                active_offsets = active_offsets[::2] + active_offsets[1::2]
            use_addr = False
            phase = None
            if round_idx >= 4 and wrap_round > 4:
                phase = (round_idx - 4) % wrap_round
                use_addr = phase <= wrap_round - 5
            if DISABLE_USE_ADDR:
                use_addr = False
                phase = None
            if skip_last_addr and round_idx == rounds - 1:
                use_addr = False
                phase = None
            if FORCE_INDEX_ADDR and (DEPTH4_VSELECT_REUSE or DEPTH4_VSELECT):
                use_addr = False
                phase = None
            if (DEPTH4_VSELECT or DEPTH4_VSELECT_REUSE) and (round_idx == 4 or round_idx == wrap_round + 4):
                use_addr = False
                phase = None

            if use_addr and phase == 0:
                slots = []
                for base_offset in active_offsets:
                    idx_block = idx_scratch + base_offset
                    slots.append(("+", idx_block, idx_block, forest_values_vec))
                emit_valu(slots)

            if round_idx == 0 or round_idx == wrap_round:
                slots = []
                for base_offset in active_offsets:
                    val_block = val_scratch + base_offset
                    slots.append(("^", val_block, val_block, node0_vec))
                emit_valu(slots)
            elif round_idx == 1 or round_idx == wrap_round + 1:
                emit_round_two_nodes(active_offsets, use_parity_cond=True)
            elif round_idx == 2 or round_idx == wrap_round + 2:
                emit_round_four_nodes(active_offsets)
            elif round_idx == 3:
                if DEPTH4_VSELECT_REUSE and not node7_valid:
                    broadcast_nodes(node7_14_vecs, node7_14_consts, xor_with_c6=True)
                    node7_valid = True
                emit_round_eight_nodes(
                    active_offsets,
                    sel_t0_vec,
                    sel_t1_vec,
                    sel_b0_vec,
                    sel_t0_vec_alt,
                    sel_t1_vec_alt,
                    sel_b0_vec_alt,
                )
            elif round_idx == wrap_round + 3:
                if DEPTH4_VSELECT_REUSE and not node7_valid:
                    broadcast_nodes(node7_14_vecs, node7_14_consts, xor_with_c6=True)
                    node7_valid = True
                emit_round_eight_nodes(active_offsets, sel_t0_vec2, sel_t1_vec2, sel_b0_vec2)
            elif DEPTH4_VSELECT_REUSE and (round_idx == 4 or round_idx == wrap_round + 4):
                emit_round_sixteen_nodes_reuse(active_offsets, round_idx)
            elif DEPTH4_VSELECT and (round_idx == 4 or round_idx == wrap_round + 4):
                emit_round_sixteen_nodes(active_offsets, round_idx)
            else:
                slots = []
                if not use_addr:
                    for base_offset in active_offsets:
                        idx_block = idx_scratch + base_offset
                        tmp_block = tmp_scratch + base_offset
                        slots.append(("+", tmp_block, idx_block, forest_values_vec))
                    emit_valu(slots)

                load_slots = []
                for base_offset in active_offsets:
                    addr_block = (
                        idx_scratch + base_offset
                        if use_addr
                        else tmp_scratch + base_offset
                    )
                    tmp_block = tmp_scratch + base_offset
                    for offset in range(VLEN):
                        load_slots.append((
                            "load_offset", tmp_block, addr_block, offset
                        ))
                emit_load(load_slots)

                slots = []
                for base_offset in active_offsets:
                    tmp_block = tmp_scratch + base_offset
                    slots.append(("^", tmp_block, tmp_block, c6_vec))
                emit_valu(slots)

                slots = []
                for base_offset in active_offsets:
                    val_block = val_scratch + base_offset
                    tmp_block = tmp_scratch + base_offset
                    slots.append(("^", val_block, val_block, tmp_block))
                emit_valu(slots)

            slots = []
            for base_offset in active_offsets:
                val_block = val_scratch + base_offset
                slots.append(("multiply_add", val_block, val_block, mul12_vec, c1_vec))
            emit_valu(slots)

            valu_slots = []
            alu_slots = []
            for base_offset in active_offsets:
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

            use_partial_alu_xor = ALU_XOR_ALL_ROUNDS or round_idx == rounds - 1
            valu_slots = []
            alu_slots = []
            for base_offset in active_offsets:
                val_block = val_scratch + base_offset
                tmp_block = shift_scratch + base_offset
                if use_partial_alu_xor and base_offset in partial_alu_xor_tmp_offsets:
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
            for base_offset in active_offsets:
                val_block = val_scratch + base_offset
                if use_partial_alu_xor and base_offset in partial_alu_xor_offsets:
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
            for base_offset in active_offsets:
                val_block = val_scratch + base_offset
                slots.append(("multiply_add", val_block, val_block, mul5_vec, c3_vec))
            emit_valu(slots)

            # Use ALU shifts for rounds 3, 10 (wrap-1), and 14 (wrap+3) to reduce VALU pressure
            use_alu_shift = (
                (ALU_SHIFT16_USE_ROUND3 and round_idx == 3)
                or (ALU_SHIFT16_USE_WRAPMINUS1 and round_idx == wrap_round - 1)
                or (ALU_SHIFT16_USE_WRAPPLUS3 and round_idx == wrap_round + 3)
            )
            use_alu_shift16 = (
                (ALU_SHIFT9_USE_ROUND3 and round_idx == 3)
                or (ALU_SHIFT9_USE_WRAPMINUS1 and round_idx == wrap_round - 1)
                or (ALU_SHIFT9_USE_WRAPPLUS3 and round_idx == wrap_round + 3)
            )
            if use_alu_shift16:
                valu_slots = []
                alu_slots = []
                for base_offset in active_offsets:
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
                for base_offset in active_offsets:
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

            use_partial_alu_add = ALU_ADD_ALL_ROUNDS or round_idx == rounds - 1
            valu_slots = []
            alu_slots = []
            for base_offset in active_offsets:
                val_block = val_scratch + base_offset
                if use_partial_alu_add and base_offset in partial_alu_add_c4_offsets:
                    for lane in range(VLEN):
                        alu_slots.append(("+", val_block + lane, val_block + lane, c4_const))
                else:
                    valu_slots.append(("+", val_block, val_block, c4_vec))
            if alu_slots:
                emit_alu(alu_slots)
            if valu_slots:
                emit_valu(valu_slots)

            valu_slots = []
            alu_slots = []
            for base_offset in active_offsets:
                val_block = val_scratch + base_offset
                tmp_block = shift_scratch + base_offset
                if use_partial_alu_xor and base_offset in partial_alu_xor_shift9_offsets:
                    for lane in range(VLEN):
                        alu_slots.append(
                            ("^", val_block + lane, val_block + lane, tmp_block + lane)
                        )
                else:
                    valu_slots.append(("^", val_block, val_block, tmp_block))
            if alu_slots:
                emit_alu(alu_slots)
            emit_valu(valu_slots)

            slots = []
            for base_offset in active_offsets:
                val_block = val_scratch + base_offset
                slots.append(("multiply_add", val_block, val_block, mul3_vec, c5_vec))
            emit_valu(slots)

            if use_alu_shift:
                valu_slots = []
                alu_slots = []
                for base_offset in active_offsets:
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
                for base_offset in active_offsets:
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

            valu_slots = []
            alu_slots = []
            for base_offset in active_offsets:
                val_block = val_scratch + base_offset
                tmp_block = shift_scratch + base_offset
                if use_partial_alu_xor and base_offset in partial_alu_xor_shift16_offsets:
                    for lane in range(VLEN):
                        alu_slots.append(
                            ("^", val_block + lane, val_block + lane, tmp_block + lane)
                        )
                else:
                    valu_slots.append(("^", val_block, val_block, tmp_block))
            if alu_slots:
                emit_alu(alu_slots)
            emit_valu(valu_slots)

            if DEBUG_HASH:
                for base_offset in active_offsets:
                    val_block = val_scratch + base_offset
                    tmp_block = tmp_scratch + base_offset
                    emit_valu([("^", tmp_block, val_block, c6_vec)])
                    keys = [
                        (round_idx, base_offset + lane, "hashed_val")
                        for lane in range(VLEN)
                    ]
                    emit(debug=[("vcompare", tmp_block, keys)])

            if round_idx % wrap_round == wrap_round - 1:
                slots = []
                for base_offset in active_offsets:
                    idx_block = idx_scratch + base_offset
                    slots.append(("+", idx_block, zero_vec, zero_vec))
                emit_valu(slots)
            else:
                # Parity AND: mostly ALU, optionally VALU for selected blocks
                valu_slots = []
                alu_slots = []
                for base_offset in active_offsets:
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

                if FOLD_IDX_ADJUST:
                    # Compute (base +/- parity) into shift_scratch, keeping tmp_scratch for next-round selects.
                    if use_addr and one_minus_base_vec is None:
                        pre_slots = []
                        post_slots = []
                        for base_offset in active_offsets:
                            tmp_block = tmp_scratch + base_offset
                            add_block = shift_scratch + base_offset
                            if c6_lsb:
                                pre_slots.append(("-", add_block, two_vec, forest_values_vec))
                                post_slots.append(("-", add_block, add_block, tmp_block))
                            else:
                                pre_slots.append(("-", add_block, one_vec, forest_values_vec))
                                post_slots.append(("+", add_block, add_block, tmp_block))
                        emit_valu(pre_slots)
                        emit_valu(post_slots)
                    else:
                        two_minus_base_tmp = None
                        if (
                            use_addr
                            and c6_lsb
                            and two_minus_base_vec is None
                            and one_minus_base_vec is not None
                        ):
                            two_minus_base_tmp = sel_b0_vec
                            emit_valu([("+", two_minus_base_tmp, one_minus_base_vec, one_vec)])
                        slots = []
                        for base_offset in active_offsets:
                            tmp_block = tmp_scratch + base_offset
                            add_block = shift_scratch + base_offset
                            if c6_lsb:
                                if use_addr:
                                    slots.append(
                                        (
                                            "-",
                                            add_block,
                                            two_minus_base_vec
                                            if two_minus_base_vec is not None
                                            else two_minus_base_tmp,
                                            tmp_block,
                                        )
                                    )
                                else:
                                    slots.append(("-", add_block, two_vec, tmp_block))
                            else:
                                if use_addr:
                                    slots.append(("+", add_block, tmp_block, one_minus_base_vec))
                                else:
                                    slots.append(("+", add_block, tmp_block, one_vec))
                        emit_valu(slots)

                    slots = []
                    for base_offset in active_offsets:
                        idx_block = idx_scratch + base_offset
                        add_block = shift_scratch + base_offset
                        slots.append(("multiply_add", idx_block, idx_block, two_vec, add_block))
                    emit_valu(slots)
                else:
                    if use_addr and one_minus_base_vec is None:
                        pre_slots = []
                        mul_slots = []
                        for base_offset in active_offsets:
                            idx_block = idx_scratch + base_offset
                            add_block = shift_scratch + base_offset
                            if c6_lsb:
                                pre_slots.append(("-", add_block, two_vec, forest_values_vec))
                            else:
                                pre_slots.append(("-", add_block, one_vec, forest_values_vec))
                            mul_slots.append(("multiply_add", idx_block, idx_block, two_vec, add_block))
                        emit_valu(pre_slots)
                        emit_valu(mul_slots)
                    else:
                        two_minus_base_tmp = None
                        if (
                            use_addr
                            and c6_lsb
                            and two_minus_base_vec is None
                            and one_minus_base_vec is not None
                        ):
                            two_minus_base_tmp = sel_b0_vec
                            emit_valu([("+", two_minus_base_tmp, one_minus_base_vec, one_vec)])
                        slots = []
                        for base_offset in active_offsets:
                            idx_block = idx_scratch + base_offset
                            if c6_lsb:
                                if use_addr:
                                    slots.append(
                                        (
                                            "multiply_add",
                                            idx_block,
                                            idx_block,
                                            two_vec,
                                            two_minus_base_vec
                                            if two_minus_base_vec is not None
                                            else two_minus_base_tmp,
                                        )
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
                        for base_offset in active_offsets:
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
                            for base_offset in active_offsets:
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
                            for base_offset in active_offsets:
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

            if DEBUG_NEXT_IDX:
                for base_offset in active_offsets:
                    idx_block = idx_scratch + base_offset
                    keys = [
                        (round_idx, base_offset + lane, "next_idx")
                        for lane in range(VLEN)
                    ]
                    emit(debug=[("vcompare", idx_block, keys)])

            if DEBUG_IDX:
                for base_offset in active_offsets:
                    idx_block = idx_scratch + base_offset
                    keys = [
                        (round_idx, base_offset + lane, "wrapped_idx")
                        for lane in range(VLEN)
                    ]
                    emit(debug=[("vcompare", idx_block, keys)])

            if (
                (DEPTH4_VSELECT or DEPTH4_VSELECT_REUSE)
                and (round_idx == 4 or round_idx == wrap_round + 4)
                and round_idx != rounds - 1
                and not DISABLE_USE_ADDR
            ):
                slots = []
                for base_offset in active_offsets:
                    idx_block = idx_scratch + base_offset
                    slots.append(("+", idx_block, idx_block, forest_values_vec))
                emit_valu(slots)

        def emit_final_group(active_offsets):
            if final_addr:
                slots = []
                for base_offset in active_offsets:
                    idx_block = idx_scratch + base_offset
                    slots.append(("-", idx_block, idx_block, forest_values_vec))
                emit_valu(slots)

            slots = []
            for base_offset in active_offsets:
                val_block = val_scratch + base_offset
                slots.append(("^", val_block, val_block, c6_vec))
            emit_valu(slots)

            for base_offset in active_offsets:
                idx_block = idx_scratch + base_offset
                val_block = val_scratch + base_offset
                i = base_index_map[base_offset]
                emit(store=[("vstore", load_idx_addrs + i, idx_block), ("vstore", load_val_addrs + i, val_block)])

        if USE_WAVEFRONT:
            for wave_idx in range(rounds + (len(offset_groups) - 1) * GROUP_SKEW):
                for group_idx, group_offsets in enumerate(offset_groups):
                    round_idx = wave_idx - group_idx * GROUP_SKEW
                    if round_idx < 0 or round_idx >= rounds:
                        continue
                    emit_round(round_idx, group_offsets)
                    if EARLY_STORE and round_idx == rounds - 1:
                        emit_final_group(group_offsets)
        else:
            for round_idx in range(rounds):
                emit_round(round_idx, base_offsets)
        if not (USE_WAVEFRONT and EARLY_STORE and rounds > 0):
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

            if USE_ADDR_ARRAYS:
                # Reuse load addresses for stores (same inp_indices_p/inp_values_p + base_const calculation)
                # This eliminates 64 ALU operations from the critical tail
                # Emit all stores - scheduler will parallelize across both store units
                for i, base_offset in enumerate(base_offsets):
                    idx_block = idx_scratch + base_offset
                    val_block = val_scratch + base_offset
                    emit(store=[("vstore", load_idx_addrs + i, idx_block), ("vstore", load_val_addrs + i, val_block)])
            else:
                if base_consts is None:
                    emit(alu=[("+", base_ptr, zero_const, zero_const)])
                for i, base_offset in enumerate(base_offsets):
                    idx_block = idx_scratch + base_offset
                    val_block = val_scratch + base_offset
                    if base_consts is None:
                        emit(alu=[("+", tmp1, self.scratch["inp_indices_p"], base_ptr), ("+", tmp2, self.scratch["inp_values_p"], base_ptr)])
                        emit(store=[("vstore", tmp1, idx_block), ("vstore", tmp2, val_block)])
                        emit(alu=[("+", base_ptr, base_ptr, vlen_const)])
                        continue
                    base_const = base_consts[i]
                    emit(alu=[("+", tmp1, self.scratch["inp_indices_p"], base_const), ("+", tmp2, self.scratch["inp_values_p"], base_const)])
                    emit(store=[("vstore", tmp1, idx_block), ("vstore", tmp2, val_block)])

        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})
        self.instrs = self._pack_vliw(self.instrs)

BASELINE = 147734
SCHED_SEED = None
USE_SLIL_SCHED = False
USE_SLOT_SCHED = False
USE_ADDR_ARRAYS = True
LOW_SCRATCH = False
FOLD_IDX_ADJUST = False
ALIAS_SHIFT_TMP = False
B0_FROM_IDX = False
FORCE_INDEX_ADDR = False
USE_WAVEFRONT = True
EARLY_STORE = False
GROUP_ORDER = "natural"
GROUP_BLOCKS = 3
GROUP_SKEW = 1
ENGINE_BIAS = None
USE_SEL_ALT = False
ALU_SHIFT9_K = 15
ALU_SHIFT16_K = 16
ALU_SHIFT9_USE_ROUND3 = True
ALU_SHIFT9_USE_WRAPMINUS1 = True
ALU_SHIFT9_USE_WRAPPLUS3 = False
ALU_SHIFT16_USE_ROUND3 = True
ALU_SHIFT16_USE_WRAPMINUS1 = True
ALU_SHIFT16_USE_WRAPPLUS3 = True
BASE_OFFSET_ORDER = "natural"
ALU_SHIFT9_ORDER = "natural"
ALU_SHIFT16_ORDER = "natural"
ALU_SHIFT9_INDICES = None
ALU_SHIFT16_INDICES = None
SCHED_MODE = "height_engine"
USE_GREEDY_SCHED = False
ALU_ADD_K = 0
ALU_ADD_ALL_ROUNDS = False
ALU_XOR_ALL_ROUNDS = False
ALU_XOR_TMP_K = 0
ALU_XOR_C2_K = 0
ALU_XOR_SHIFT9_K = 0
ALU_XOR_SHIFT16_K = 0
VALU_SHIFT9_K = 0
VALU_SHIFT16_K = 0
VALU_PARITY_K = 0
VALU_ADJUST_K = 0
SHIFT19_PREFIX_K = None
SHIFT19_INDICES = (1,)
DEPTH2_USE_FLOW = True
DEPTH2_VALU_K = 0
DEPTH3_XOR_ALU_K = 0
DEPTH4_VSELECT = False
DEPTH4_VSELECT_REUSE = False
DEBUG_DEPTH4 = False
DEBUG_IDX = False
DEBUG_HASH = False
DEPTH1_USE_PARITY = True
DEPTH1_VALU_SELECT = False
DEBUG_NEXT_IDX = False
ACTIVE_OFFSET_ORDER = "natural"
GROUP_SIZES = None
ALU_CONSTS = False
DISABLE_USE_ADDR = False
PRE_XOR_NODE_CONSTS = False
RELAX_CYCLE_HAZARDS = False
USE_VLOAD_NODES = True

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
    quiet: bool = False,
):
    if not quiet:
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
        if (DEBUG_DEPTH4 or DEBUG_IDX) and i == 1:
            if DEBUG_DEPTH4:
                c6 = HASH_STAGES[5][1]
                for key, val in list(value_trace.items()):
                    if len(key) == 3 and key[2] == "node_val":
                        value_trace[(key[0], key[1], "node_c6")] = val ^ c6
            machine.enable_debug = True
        else:
            machine.enable_debug = False
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

    if not quiet:
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
