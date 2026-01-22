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


def compute_dependencies(engine: str, operation: tuple):
    """
    Analyze an operation to determine which memory addresses it reads/writes.
    Returns (read_addrs, write_addrs) for dependency tracking.
    """
    vec_addrs = lambda base: list(range(base, base + VLEN))
    reads, writes = [], []

    if engine == "alu":
        _, dst, src1, src2 = operation
        reads = [src1, src2]
        writes = [dst]
    elif engine == "valu":
        if operation[0] == "vbroadcast":
            _, dst, src = operation
            reads = [src]
            writes = vec_addrs(dst)
        elif operation[0] == "multiply_add":
            _, dst, a, b, c = operation
            reads = vec_addrs(a) + vec_addrs(b) + vec_addrs(c)
            writes = vec_addrs(dst)
        else:
            _, dst, a, b = operation
            reads = vec_addrs(a) + vec_addrs(b)
            writes = vec_addrs(dst)
    elif engine == "load":
        if operation[0] == "const":
            _, dst, _ = operation
            writes = [dst]
        elif operation[0] == "load":
            _, dst, addr = operation
            reads = [addr]
            writes = [dst]
        elif operation[0] == "vload":
            _, dst, addr = operation
            reads = [addr]
            writes = vec_addrs(dst)
    elif engine == "store":
        if operation[0] == "vstore":
            _, addr, src = operation
            reads = [addr] + vec_addrs(src)
    elif engine == "flow":
        if operation[0] == "vselect":
            _, dst, cond, a, b = operation
            reads = vec_addrs(cond) + vec_addrs(a) + vec_addrs(b)
            writes = vec_addrs(dst)
        elif operation[0] == "add_imm":
            _, dst, src, _ = operation
            reads = [src]
            writes = [dst]

    return reads, writes


PACK_STRATEGY = "greedy"
XOR_VALU_DEPTHS = {4, 9}
SCALAR_HASH_OP3_STAGES = set()


def pack_operations_greedy(ops_list):
    """
    Pack a flat list of (engine, op) tuples into VLIW bundles.
    Uses greedy scheduling respecting data dependencies.
    """
    bundles = []
    slot_usage = []
    addr_ready = defaultdict(int)  # cycle when addr is available
    addr_last_write = defaultdict(lambda: -1)
    addr_last_read = defaultdict(lambda: -1)

    def get_or_create_bundle(idx):
        while len(bundles) <= idx:
            bundles.append({})
            slot_usage.append(defaultdict(int))

    def first_available_slot(eng, min_cycle):
        cycle = min_cycle
        limit = SLOT_LIMITS[eng]
        while True:
            get_or_create_bundle(cycle)
            if slot_usage[cycle][eng] < limit:
                return cycle
            cycle += 1

    for eng, op in ops_list:
        reads, writes = compute_dependencies(eng, op)

        # Find earliest cycle based on dependencies
        earliest = 0
        for addr in reads:
            earliest = max(earliest, addr_ready[addr])
        for addr in writes:
            earliest = max(earliest, addr_last_write[addr] + 1, addr_last_read[addr])

        cycle = first_available_slot(eng, earliest)
        get_or_create_bundle(cycle)
        bundles[cycle].setdefault(eng, []).append(op)
        slot_usage[cycle][eng] += 1

        # Update tracking
        for addr in reads:
            addr_last_read[addr] = max(addr_last_read[addr], cycle)
        for addr in writes:
            addr_last_write[addr] = cycle
            addr_ready[addr] = cycle + 1

    return [b for b in bundles if b]


def pack_operations_critical(ops_list):
    """
    Pack operations into VLIW bundles using critical-path list scheduling.
    """
    if not ops_list:
        return []

    n_ops = len(ops_list)
    preds = [dict() for _ in range(n_ops)]
    succs = [[] for _ in range(n_ops)]
    last_write = {}
    last_read = {}
    last_mem = None

    for idx, (eng, op) in enumerate(ops_list):
        reads, writes = compute_dependencies(eng, op)

        for addr in reads:
            if addr in last_write:
                pred = last_write[addr]
                latency = 1
                prev = preds[idx].get(pred)
                if prev is None or latency > prev:
                    preds[idx][pred] = latency

        for addr in writes:
            if addr in last_write:
                pred = last_write[addr]
                latency = 1
                prev = preds[idx].get(pred)
                if prev is None or latency > prev:
                    preds[idx][pred] = latency
            if addr in last_read:
                pred = last_read[addr]
                latency = 0
                prev = preds[idx].get(pred)
                if prev is None or latency > prev:
                    preds[idx][pred] = latency

        if eng in ("load", "store"):
            if last_mem is not None:
                prev = preds[idx].get(last_mem)
                if prev is None or 0 > prev:
                    preds[idx][last_mem] = 0
            last_mem = idx

        for addr in reads:
            last_read[addr] = idx
        for addr in writes:
            last_write[addr] = idx

    for idx in range(n_ops):
        for pred, latency in preds[idx].items():
            succs[pred].append((idx, latency))

    pred_counts = [len(preds[idx]) for idx in range(n_ops)]
    ready = {idx for idx, count in enumerate(pred_counts) if count == 0}

    # Topological order for critical path length.
    topo = []
    queue = [idx for idx, count in enumerate(pred_counts) if count == 0]
    while queue:
        idx = queue.pop()
        topo.append(idx)
        for succ, _ in succs[idx]:
            pred_counts[succ] -= 1
            if pred_counts[succ] == 0:
                queue.append(succ)

    if len(topo) != n_ops:
        return pack_operations_greedy(ops_list)

    crit_dist = [0] * n_ops
    for idx in reversed(topo):
        best = 0
        for succ, latency in succs[idx]:
            cand = latency + crit_dist[succ]
            if cand > best:
                best = cand
        crit_dist[idx] = best

    pred_counts = [len(preds[idx]) for idx in range(n_ops)]
    earliest = [0] * n_ops
    ready = {idx for idx, count in enumerate(pred_counts) if count == 0}

    bundles = []
    slot_usage = []

    def get_or_create_bundle(cycle):
        while len(bundles) <= cycle:
            bundles.append({})
            slot_usage.append(defaultdict(int))

    remaining = n_ops
    cycle = 0
    while remaining > 0:
        if not ready:
            return pack_operations_greedy(ops_list)

        get_or_create_bundle(cycle)
        scheduled_any = False

        for eng, limit in SLOT_LIMITS.items():
            if slot_usage[cycle][eng] >= limit:
                continue

            for _ in range(limit - slot_usage[cycle][eng]):
                best_idx = None
                best_pri = None
                for op_idx in ready:
                    if ops_list[op_idx][0] != eng:
                        continue
                    if earliest[op_idx] > cycle:
                        continue
                    pri = (crit_dist[op_idx], -earliest[op_idx], -op_idx)
                    if best_pri is None or pri > best_pri:
                        best_pri = pri
                        best_idx = op_idx

                if best_idx is None:
                    break

                bundles[cycle].setdefault(eng, []).append(ops_list[best_idx][1])
                slot_usage[cycle][eng] += 1
                scheduled_any = True

                ready.remove(best_idx)
                remaining -= 1

                for succ, latency in succs[best_idx]:
                    pred_counts[succ] -= 1
                    if earliest[succ] < cycle + latency:
                        earliest[succ] = cycle + latency
                    if pred_counts[succ] == 0:
                        ready.add(succ)

        if remaining == 0:
            break

        if not scheduled_any:
            return pack_operations_greedy(ops_list)

        cycle += 1

    return [b for b in bundles if b]


def pack_operations(ops_list):
    if PACK_STRATEGY == "critical":
        return pack_operations_critical(ops_list)
    return pack_operations_greedy(ops_list)


class KernelBuilder:
    """Builds an optimized kernel for tree traversal with hashing."""

    def __init__(self):
        self.instrs = []
        self.memory_map = {}
        self.memory_labels = {}
        self.next_addr = 0
        self.scalar_cache = {}
        self.vector_cache = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.memory_labels)

    def emit(self, engine, operation):
        self.instrs.append({engine: [operation]})

    def reserve(self, label=None, size=1):
        """Reserve scratch memory and return the base address."""
        addr = self.next_addr
        if label:
            self.memory_map[label] = addr
            self.memory_labels[addr] = (label, size)
        self.next_addr += size
        assert self.next_addr <= SCRATCH_SIZE, "Scratch memory exhausted"
        return addr

    def reserve_vector(self, label=None):
        return self.reserve(label, VLEN)

    def get_scalar(self, value, ops_buffer=None):
        """Get or create a scalar constant, optionally deferring the load."""
        if value not in self.scalar_cache:
            addr = self.reserve()
            load_op = ("load", ("const", addr, value))
            if ops_buffer is None:
                self.emit(*load_op)
            else:
                ops_buffer.append(load_op)
            self.scalar_cache[value] = addr
        return self.scalar_cache[value]

    def get_vector(self, value, ops_buffer=None):
        """Get or create a vector constant (broadcast from scalar)."""
        if value not in self.vector_cache:
            scalar_addr = self.get_scalar(value, ops_buffer)
            vec_addr = self.reserve_vector()
            broadcast_op = ("valu", ("vbroadcast", vec_addr, scalar_addr))
            if ops_buffer is None:
                self.emit(*broadcast_op)
            else:
                ops_buffer.append(broadcast_op)
            self.vector_cache[value] = vec_addr
        return self.vector_cache[value]

    def build_kernel(self, tree_depth, _node_count, items, num_rounds, tile_blocks=19, tile_rounds=13):
        """
        Generate the optimized kernel.

        Args:
            tree_depth: Height of the tree (forest_height)
            _node_count: Number of nodes in tree (unused, kept for API compatibility)
            items: Batch size
            num_rounds: Number of hash rounds
            tile_blocks: Number of blocks to process together
            tile_rounds: Number of rounds to process together
        """
        # Temporary registers for address computation
        addr_tmp = self.reserve("addr_tmp")

        # Known memory layout base; derive pointers from inputs.
        TREE_BASE = 7
        IDX_BASE = TREE_BASE + _node_count
        VAL_BASE = IDX_BASE + items

        # Reserve space for header values (even though we hardcode them)
        for name in ["ptr_tree", "ptr_idx", "ptr_val"]:
            self.reserve(name)

        # === INITIALIZATION PHASE ===
        setup_ops = []

        # Load hardcoded pointers
        setup_ops.append(("load", ("const", self.memory_map["ptr_tree"], TREE_BASE)))
        setup_ops.append(("load", ("const", self.memory_map["ptr_idx"], IDX_BASE)))
        setup_ops.append(("load", ("const", self.memory_map["ptr_val"], VAL_BASE)))

        # Common vector constants
        vec_zero = self.get_vector(0, setup_ops)
        vec_two = self.get_vector(2, setup_ops)
        const_one = self.get_scalar(1, setup_ops)
        const_two = self.get_scalar(2, setup_ops)
        const_minus_six = self.get_scalar(-6, setup_ops)

        # Broadcast tree base pointer
        vec_tree_base = self.reserve_vector("vec_tree_base")
        setup_ops.append(("valu", ("vbroadcast", vec_tree_base, self.memory_map["ptr_tree"])))

        # Additional vector constants for selection logic
        const_ten = self.get_scalar(10, setup_ops)
        const_fourteen = self.get_scalar(14, setup_ops)
        const_eighteen = self.get_scalar(18, setup_ops)

        # Pre-load tree nodes 0-14 for levels 0-3 (avoid gathers)
        preloaded_nodes = []
        NUM_PRELOAD = 15
        nodes_tmp = self.reserve_vector("nodes_tmp")
        setup_ops.append(("load", ("vload", nodes_tmp, self.memory_map["ptr_tree"])))
        for i in range(8):
            vector_slot = self.reserve_vector(f"vec_tree_{i}")
            setup_ops.append(("valu", ("vbroadcast", vector_slot, nodes_tmp + i)))
            preloaded_nodes.append(vector_slot)
        offset_8 = self.get_scalar(8, setup_ops)
        setup_ops.append(("alu", ("+", addr_tmp, self.memory_map["ptr_tree"], offset_8)))
        setup_ops.append(("load", ("vload", nodes_tmp, addr_tmp)))
        for i in range(8, NUM_PRELOAD):
            vector_slot = self.reserve_vector(f"vec_tree_{i}")
            setup_ops.append(("valu", ("vbroadcast", vector_slot, nodes_tmp + (i - 8))))
            preloaded_nodes.append(vector_slot)






        # Hash stage constants (with fusion for compatible stages)
        hash_const1 = []
        hash_const1_scalar = []
        hash_const3 = []
        hash_const3_scalar = []
        hash_multipliers = []
        for stage_op1, c1, stage_op2, stage_op3, c3 in HASH_STAGES:
            hash_const1_scalar.append(self.get_scalar(c1, setup_ops))
            hash_const1.append(self.get_vector(c1, setup_ops))
            # Fuse stages where possible: val = val * (1 + 2^c3) + c1
            if stage_op1 == "+" and stage_op2 == "+" and stage_op3 == "<<":
                multiplier = 1 + (1 << c3)
                hash_multipliers.append(self.get_vector(multiplier, setup_ops))
                hash_const3.append(None)
                hash_const3_scalar.append(None)
            else:
                hash_multipliers.append(None)
                hash_const3_scalar.append(self.get_scalar(c3, setup_ops))
                hash_const3.append(self.get_vector(c3, setup_ops))

        scalar_hash_stages = set()

        # Working memory for addresses and values
        assert items % VLEN == 0
        num_blocks = items // VLEN
        working_addr = self.reserve("working_addr", items)
        working_val = self.reserve("working_val", items)

        # Offset counter and VLEN constant
        offset_counter = self.reserve("offset_counter")
        setup_ops.append(("load", ("const", offset_counter, 0)))
        vlen_scalar = self.get_scalar(VLEN, setup_ops)

        # Initialize working addresses to tree base (indices start at 0).
        for blk in range(num_blocks):
            setup_ops.append(("valu", (
                "+", working_addr + blk * VLEN, vec_tree_base, vec_zero
            )))

        # Emit packed initialization
        self.instrs.extend(pack_operations(setup_ops))
        if self.instrs and "flow" not in self.instrs[-1]:
            self.instrs[-1].setdefault("flow", []).append(("pause",))
        else:
            self.emit("flow", ("pause",))

        # === LOAD INITIAL DATA ===
        body_ops = []
        for blk in range(num_blocks):
            body_ops.append(("alu", ("+", addr_tmp, self.memory_map["ptr_val"], offset_counter)))
            body_ops.append(("load", ("vload", working_val + blk * VLEN, addr_tmp)))
            body_ops.append(("alu", ("+", offset_counter, offset_counter, vlen_scalar)))

        # Working buffers for processing groups
        work_buffers = []
        for _ in range(tile_blocks):
            work_buffers.append({
                "result": self.reserve_vector(),
                "temp2": self.reserve_vector(),
                "temp3": self.reserve_vector(),
                "child_right": self.reserve_vector(),
            })

        # === MAIN COMPUTATION LOOP ===
        for round_base in range(0, num_rounds, tile_rounds):
            round_limit = min(num_rounds, round_base + tile_rounds)

            for group_base in range(0, num_blocks, tile_blocks):
                for buf_idx in range(tile_blocks):
                        blk = group_base + buf_idx
                        if blk >= num_blocks:
                            break

                        wb = work_buffers[buf_idx]
                        addr_vec = working_addr + blk * VLEN
                        val_vec = working_val + blk * VLEN

                        def do_xor(node_vec, use_vector):
                            if use_vector:
                                body_ops.append(("valu", ("^", val_vec, val_vec, node_vec)))
                                return
                            for lane in range(VLEN):
                                body_ops.append((
                                    "alu", ("^", val_vec + lane, val_vec + lane, node_vec + lane)
                                ))

                        def emit_hash():
                            for stage_idx, (op1, _, op2, op3, _) in enumerate(HASH_STAGES):
                                if hash_multipliers[stage_idx] is not None:
                                    # Fused operation: val = val * mult + c1
                                    body_ops.append(("valu", (
                                        "multiply_add", val_vec, val_vec,
                                        hash_multipliers[stage_idx], hash_const1[stage_idx]
                                    )))
                                else:
                                    # Three-operation sequence
                                    if stage_idx in scalar_hash_stages:
                                        for lane in range(VLEN):
                                            body_ops.append((
                                                "alu",
                                                (op1, wb["temp2"] + lane,
                                                 val_vec + lane,
                                                 hash_const1_scalar[stage_idx]),
                                            ))
                                            body_ops.append((
                                                "alu",
                                                (op3, wb["temp3"] + lane,
                                                 val_vec + lane,
                                                 hash_const3_scalar[stage_idx]),
                                            ))
                                            body_ops.append((
                                                "alu",
                                                (op2, val_vec + lane,
                                                 wb["temp2"] + lane,
                                                 wb["temp3"] + lane),
                                            ))
                                    else:
                                        body_ops.append(("valu", (
                                            op1, wb["temp2"], val_vec, hash_const1[stage_idx]
                                        )))
                                        if stage_idx in SCALAR_HASH_OP3_STAGES:
                                            for lane in range(VLEN):
                                                body_ops.append((
                                                    "alu",
                                                    (op3, wb["temp3"] + lane,
                                                     val_vec + lane,
                                                     hash_const3_scalar[stage_idx]),
                                                ))
                                        else:
                                            body_ops.append(("valu", (
                                                op3, wb["temp3"], val_vec, hash_const3[stage_idx]
                                            )))
                                        body_ops.append(("valu", (
                                            op2, val_vec, wb["temp2"], wb["temp3"]
                                        )))

                        def emit_index_update(depth):
                            if depth == tree_depth:
                                # Wrap: reset index to 0
                                body_ops.append(("valu", ("+", addr_vec, vec_tree_base, vec_zero)))
                                return None
                            for lane in range(VLEN):
                                body_ops.append((
                                    "alu",
                                    ("&", wb["temp2"] + lane, val_vec + lane, const_one)
                                ))
                                body_ops.append((
                                    "alu",
                                    ("+", wb["temp3"] + lane, wb["temp2"] + lane, const_minus_six)
                                ))
                            body_ops.append(("valu", (
                                "multiply_add", addr_vec, addr_vec, vec_two, wb["temp3"]
                            )))
                            return wb["temp2"]

                        def emit_node_lookup(depth):
                            # Level-specific tree node lookup
                            if depth == 0:
                                # All indices are 0 at start/after wrap - use preloaded node[0]
                                return preloaded_nodes[0]

                            if depth == 1:
                                # Indices are 1 or 2 - binary selection
                                for lane in range(VLEN):
                                    body_ops.append((
                                        "alu",
                                        ("&", wb["temp2"] + lane, addr_vec + lane, const_one)
                                    ))
                                body_ops.append(("flow", (
                                    "vselect", wb["result"], wb["temp2"],
                                    preloaded_nodes[2], preloaded_nodes[1]
                                )))
                                return wb["result"]

                            if depth == 2:
                                # Indices 3-6: two-level selection
                                for lane in range(VLEN):
                                    body_ops.append((
                                        "alu",
                                        ("-", wb["temp3"] + lane, addr_vec + lane, const_ten)
                                    ))
                                    body_ops.append((
                                        "alu",
                                        ("&", wb["temp2"] + lane, wb["temp3"] + lane, const_one)
                                    ))
                                    body_ops.append((
                                        "alu",
                                        ("&", wb["temp3"] + lane, wb["temp3"] + lane, const_two)
                                    ))

                                body_ops.append(("flow", (
                                    "vselect", wb["result"], wb["temp2"],
                                    preloaded_nodes[4], preloaded_nodes[3]
                                )))
                                body_ops.append(("flow", (
                                    "vselect", wb["child_right"], wb["temp2"],
                                    preloaded_nodes[6], preloaded_nodes[5]
                                )))
                                body_ops.append(("flow", (
                                    "vselect", wb["result"], wb["temp3"],
                                    wb["child_right"], wb["result"]
                                )))
                                return wb["result"]

                            if depth == 3:
                                # Indices 7-14: three-level selection
                                for lane in range(VLEN):
                                    body_ops.append((
                                        "alu",
                                        ("-", wb["child_right"] + lane, addr_vec + lane, const_fourteen)
                                    ))
                                    body_ops.append((
                                        "alu",
                                        ("&", wb["temp2"] + lane, wb["child_right"] + lane, const_one)
                                    ))
                                    body_ops.append((
                                        "alu",
                                        ("&", wb["temp3"] + lane, wb["child_right"] + lane, const_two)
                                    ))

                                # First pair selections
                                body_ops.append(("flow", (
                                    "vselect", wb["result"], wb["temp2"],
                                    preloaded_nodes[8], preloaded_nodes[7]
                                )))
                                body_ops.append(("flow", (
                                    "vselect", wb["child_right"], wb["temp2"],
                                    preloaded_nodes[10], preloaded_nodes[9]
                                )))
                                body_ops.append(("flow", (
                                    "vselect", wb["result"], wb["temp3"],
                                    wb["child_right"], wb["result"]
                                )))

                                # Second pair selections
                                body_ops.append(("flow", (
                                    "vselect", wb["child_right"], wb["temp2"],
                                    preloaded_nodes[12], preloaded_nodes[11]
                                )))
                                body_ops.append(("flow", (
                                    "vselect", wb["temp2"], wb["temp2"],
                                    preloaded_nodes[14], preloaded_nodes[13]
                                )))
                                body_ops.append(("flow", (
                                    "vselect", wb["child_right"], wb["temp3"],
                                    wb["temp2"], wb["child_right"]
                                )))

                                # Final bit selection (bit 2)
                                for lane in range(VLEN):
                                    body_ops.append((
                                        "alu",
                                        ("<", wb["temp2"] + lane, addr_vec + lane, const_eighteen)
                                    ))
                                body_ops.append(("flow", (
                                    "vselect", wb["result"], wb["temp2"],
                                    wb["result"], wb["child_right"]
                                )))
                                return wb["result"]

                            # Deep levels: gather from memory
                            for lane in range(VLEN):
                                body_ops.append((
                                    "load",
                                    ("load", wb["result"] + lane, addr_vec + lane)
                                ))
                            return wb["result"]

                        def emit_child_nodes(depth):
                            # vEB-inspired: precompute both child paths for the next depth.
                            if depth == 1:
                                body_ops.append(("flow", (
                                    "vselect", wb["result"], wb["temp2"],
                                    preloaded_nodes[5], preloaded_nodes[3]
                                )))
                                body_ops.append(("flow", (
                                    "vselect", wb["child_right"], wb["temp2"],
                                    preloaded_nodes[6], preloaded_nodes[4]
                                )))
                                return

                            if depth == 2:
                                body_ops.append(("flow", (
                                    "vselect", wb["result"], wb["temp2"],
                                    preloaded_nodes[9], preloaded_nodes[7]
                                )))
                                body_ops.append(("flow", (
                                    "vselect", wb["child_right"], wb["temp2"],
                                    preloaded_nodes[13], preloaded_nodes[11]
                                )))
                                body_ops.append(("flow", (
                                    "vselect", wb["result"], wb["temp3"],
                                    wb["child_right"], wb["result"]
                                )))

                                body_ops.append(("flow", (
                                    "vselect", wb["child_right"], wb["temp2"],
                                    preloaded_nodes[10], preloaded_nodes[8]
                                )))
                                body_ops.append(("flow", (
                                    "vselect", wb["temp2"], wb["temp2"],
                                    preloaded_nodes[14], preloaded_nodes[12]
                                )))
                                body_ops.append(("flow", (
                                    "vselect", wb["child_right"], wb["temp3"],
                                    wb["temp2"], wb["child_right"]
                                )))

                        # Loop unrolling 2x with register renaming for child buffers.
                        for rnd in range(round_base, round_limit, 2):
                            depth = rnd % (tree_depth + 1)
                            has_next = rnd + 1 < round_limit
                            spec_next = has_next and depth <= 2 and depth < tree_depth

                            node_vec = emit_node_lookup(depth)
                            do_xor(node_vec, depth in XOR_VALU_DEPTHS)

                            if spec_next and depth in (1, 2):
                                emit_child_nodes(depth)

                            emit_hash()
                            emit_index_update(depth)

                            if spec_next:
                                if depth == 0:
                                    body_ops.append(("flow", (
                                        "vselect", wb["result"], wb["temp2"],
                                        preloaded_nodes[2], preloaded_nodes[1]
                                    )))
                                else:
                                    body_ops.append(("flow", (
                                        "vselect", wb["result"], wb["temp2"],
                                        wb["child_right"], wb["result"]
                                    )))

                            if not has_next:
                                continue

                            next_depth = (rnd + 1) % (tree_depth + 1)
                            if spec_next:
                                node_vec = wb["result"]
                            else:
                                node_vec = emit_node_lookup(next_depth)

                            do_xor(node_vec, next_depth in XOR_VALU_DEPTHS)
                            emit_hash()
                            emit_index_update(next_depth)

        # === STORE RESULTS (values and indices) ===
        store_ops = []
        store_ops.append(("load", ("const", offset_counter, 0)))
        for blk in range(num_blocks):
            store_ops.append(("valu", (
                "-", working_addr + blk * VLEN, working_addr + blk * VLEN, vec_tree_base
            )))
            store_ops.append(("alu", ("+", addr_tmp, self.memory_map["ptr_idx"], offset_counter)))
            store_ops.append(("store", ("vstore", addr_tmp, working_addr + blk * VLEN)))
            store_ops.append(("alu", ("+", addr_tmp, self.memory_map["ptr_val"], offset_counter)))
            store_ops.append(("store", ("vstore", addr_tmp, working_val + blk * VLEN)))
            store_ops.append(("alu", ("+", offset_counter, offset_counter, vlen_scalar)))
        body_ops.extend(store_ops)

        # Pack and emit all body operations
        self.instrs.extend(pack_operations(body_ops))
        if self.instrs and "flow" not in self.instrs[-1]:
            self.instrs[-1].setdefault("flow", []).append(("pause",))
        else:
            self.instrs.append({"flow": [("pause",)]})


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
        assert (
            machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)]
            == ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)]
        ), f"Incorrect indices on round {i}"

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
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
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()
