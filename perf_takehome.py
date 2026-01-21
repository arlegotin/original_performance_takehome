"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

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

    def add_bundle(self, bundle):
        """Add a packed instruction bundle: {engine: [slots]}."""
        self.instrs.append(bundle)

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

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots


    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        # OPTIMIZED: Use different temp addresses to pack parameter loads
        # Pairs: (rounds,0), (n_nodes,1), (batch_size,2), (forest_height,3), (forest_values_p,4), (inp_indices_p,5), (inp_values_p,6)
        tmp_addrs = [tmp1, tmp2, tmp1, tmp2, tmp1, tmp2, tmp1]  # Alternate between tmp1 and tmp2
        for i in range(0, len(init_vars), 2):
            if i + 1 < len(init_vars):
                # Pack 2 const loads
                self.add_bundle({"load": [("const", tmp_addrs[i], i), ("const", tmp_addrs[i+1], i+1)]})
                # Pack 2 memory loads
                self.add_bundle({"load": [
                    ("load", self.scratch[init_vars[i]], tmp_addrs[i]),
                    ("load", self.scratch[init_vars[i+1]], tmp_addrs[i+1]),
                ]})
            else:
                # Odd one out - defer and combine with const 0,1,2 loads below
                odd_idx = i
                odd_addr = tmp_addrs[i]
                odd_var = init_vars[i]

        # OPTIMIZED: Pack odd parameter with const 0,1,2 loads
        zero_const = self.alloc_scratch("zero_const")
        one_const = self.alloc_scratch("one_const")
        two_const = self.alloc_scratch("two_const")
        self.const_map[0] = zero_const
        self.const_map[1] = one_const
        self.const_map[2] = two_const
        # Preload tree[0] for round 0 optimization
        root_scalar = self.alloc_scratch("root_scalar")
        root_vec = self.alloc_scratch("root_vec", VLEN)
        # Combine: const for odd param + const 0
        self.add_bundle({"load": [("const", odd_addr, odd_idx), ("const", zero_const, 0)]})
        # Combine: load odd param + const 1
        self.add_bundle({"load": [("load", self.scratch[odd_var], odd_addr), ("const", one_const, 1)]})
        # Combine: const 2 + root_scalar load
        self.add_bundle({"load": [("const", two_const, 2), ("load", root_scalar, self.scratch["forest_values_p"])]})

        zero_v = self.alloc_scratch("zero_v", VLEN)
        one_v = self.alloc_scratch("one_v", VLEN)
        two_v = self.alloc_scratch("two_v", VLEN)
        n_nodes_v = self.alloc_scratch("n_nodes_v", VLEN)
        forest_base_v = self.alloc_scratch("forest_base_v", VLEN)
        # OPTIMIZED: Pack 6 vbroadcasts into 1 cycle (root_vec added since root_scalar already loaded)
        self.add_bundle({"valu": [
            ("vbroadcast", zero_v, zero_const),
            ("vbroadcast", one_v, one_const),
            ("vbroadcast", two_v, two_const),
            ("vbroadcast", n_nodes_v, self.scratch["n_nodes"]),
            ("vbroadcast", forest_base_v, self.scratch["forest_values_p"]),
            ("vbroadcast", root_vec, root_scalar),
        ]})

        # Preload tree[1], tree[2] for round 1 optimization (idx in {1,2} after round 0)
        left_scalar = self.alloc_scratch("left_scalar")
        right_scalar = self.alloc_scratch("right_scalar")
        left_vec = self.alloc_scratch("left_vec", VLEN)
        right_vec = self.alloc_scratch("right_vec", VLEN)
        delta_lr_vec = self.alloc_scratch("delta_lr_vec", VLEN)  # tree2 - tree1
        # OPTIMIZED: Allocate const 3-6 early so we can combine ALU ops with loads
        const_three = self.alloc_scratch("const_three")
        const_four = self.alloc_scratch("const_four")
        const_five = self.alloc_scratch("const_five")
        const_six = self.alloc_scratch("const_six")
        self.const_map[3] = const_three
        self.const_map[4] = const_four
        self.const_map[5] = const_five
        self.const_map[6] = const_six
        # OPTIMIZED: Pack tree1/tree2 ALU with const 3,4 loads (saves 1 cycle)
        self.add_bundle({
            "alu": [
                ("+", left_scalar, self.scratch["forest_values_p"], one_const),
                ("+", right_scalar, self.scratch["forest_values_p"], two_const),
            ],
            "load": [("const", const_three, 3), ("const", const_four, 4)],
        })
        # OPTIMIZED: Pack tree1/tree2 loads together
        self.add_bundle({"load": [
            ("load", left_scalar, left_scalar),
            ("load", right_scalar, right_scalar),
        ]})
        # OPTIMIZED: Pack const 5,6 loads with tree1/tree2 broadcasts
        self.add_bundle({
            "load": [("const", const_five, 5), ("const", const_six, 6)],
            "valu": [
                ("vbroadcast", left_vec, left_scalar),
                ("vbroadcast", right_vec, right_scalar),
            ],
        })

        # Preload tree[3..6] for round 2 optimization (idx in {3,4,5,6} after round 1)
        three_vec = self.alloc_scratch("three_vec", VLEN)
        node3_scalar = self.alloc_scratch("node3_scalar")
        node4_scalar = self.alloc_scratch("node4_scalar")
        node5_scalar = self.alloc_scratch("node5_scalar")
        node6_scalar = self.alloc_scratch("node6_scalar")
        node3_vec = self.alloc_scratch("node3_vec", VLEN)
        node4_vec = self.alloc_scratch("node4_vec", VLEN)
        node5_vec = self.alloc_scratch("node5_vec", VLEN)
        node6_vec = self.alloc_scratch("node6_vec", VLEN)
        delta_3_4_vec = self.alloc_scratch("delta_3_4_vec", VLEN)  # tree4 - tree3
        delta_5_6_vec = self.alloc_scratch("delta_5_6_vec", VLEN)  # tree6 - tree5

        # OPTIMIZED: Pack diff_1_2/three_vec with tree3-6 ALU ops (saves 1 cycle)
        self.add_bundle({
            "valu": [
                ("-", delta_lr_vec, right_vec, left_vec),
                ("vbroadcast", three_vec, const_three),
            ],
            "alu": [
                ("+", node3_scalar, self.scratch["forest_values_p"], const_three),
                ("+", node4_scalar, self.scratch["forest_values_p"], const_four),
                ("+", node5_scalar, self.scratch["forest_values_p"], const_five),
                ("+", node6_scalar, self.scratch["forest_values_p"], const_six),
            ],
        })
        # OPTIMIZED: Pack loads 2 per cycle
        # OPTIMIZED: Overlap loads with broadcasts - load tree3,4, then load tree5,6 with broadcast tree3,4
        self.add_bundle({"load": [
            ("load", node3_scalar, node3_scalar),
            ("load", node4_scalar, node4_scalar),
        ]})
        self.add_bundle({
            "load": [
                ("load", node5_scalar, node5_scalar),
                ("load", node6_scalar, node6_scalar),
            ],
            "valu": [
                ("vbroadcast", node3_vec, node3_scalar),
                ("vbroadcast", node4_vec, node4_scalar),
            ],
        })
        # Broadcast tree5,6 and compute diff_3_4 (all VALU)
        self.add_bundle({"valu": [
            ("vbroadcast", node5_vec, node5_scalar),
            ("vbroadcast", node6_vec, node6_scalar),
            ("-", delta_3_4_vec, node4_vec, node3_vec),
        ]})
        # diff_5_6 will be deferred to combine with first hash const load below
        deferred_delta_5_6 = ("-", delta_5_6_vec, node6_vec, node5_vec)

        # OPTIMIZED: Load all hash constants first, then pack vbroadcasts
        hash_c1_vecs = []
        hash_c3_vecs = []
        hash_c3_lane_scalars = []
        hash_mul_vecs = []

        # Phase 1: Allocate and load all scalar constants (pack const loads 2 per cycle)
        hash_c1_scalars = []
        hash_c3_broadcast_scalars = []
        hash_mul_scalars = []

        # Collect all const values and addresses first
        const_load_queue = []  # List of (addr, value)
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            # c1 const
            if val1 not in self.const_map:
                addr = self.alloc_scratch(f"hash_c1_s_{hi}")
                self.const_map[val1] = addr
                const_load_queue.append((addr, val1))
            hash_c1_scalars.append(self.const_map[val1])

            # c3 const
            if val3 not in self.const_map:
                addr = self.alloc_scratch(f"hash_c3_s_{hi}")
                self.const_map[val3] = addr
                const_load_queue.append((addr, val3))
            hash_c3_broadcast_scalars.append(self.const_map[val3])
            hash_c3_lane_scalars.append(self.const_map[val3])

            # mul const for fusible stages
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mul = (1 + (1 << val3)) % (2**32)
                if mul not in self.const_map:
                    addr = self.alloc_scratch(f"hash_mul_s_{hi}")
                    self.const_map[mul] = addr
                    const_load_queue.append((addr, mul))
                hash_mul_scalars.append((hi, self.const_map[mul]))
            else:
                hash_mul_scalars.append((hi, None))

        # Pack const loads 2 per cycle, combine first batch with deferred diff_5_6
        for i in range(0, len(const_load_queue), 2):
            if i + 1 < len(const_load_queue):
                bundle = {"load": [
                    ("const", const_load_queue[i][0], const_load_queue[i][1]),
                    ("const", const_load_queue[i + 1][0], const_load_queue[i + 1][1]),
                ]}
                # Combine diff_5_6 with first batch of const loads (saves 1 cycle)
                if i == 0 and deferred_delta_5_6:
                    bundle["valu"] = [deferred_delta_5_6]
                self.add_bundle(bundle)
            else:
                self.add("load", ("const", const_load_queue[i][0], const_load_queue[i][1]))

        # Phase 2: Allocate vector destinations
        for hi in range(len(HASH_STAGES)):
            c1_v = self.alloc_scratch(f"hash_c1_v_{hi}", VLEN)
            c3_v = self.alloc_scratch(f"hash_c3_v_{hi}", VLEN)
            hash_c1_vecs.append(c1_v)
            hash_c3_vecs.append(c3_v)

        for hi, mul_scalar in hash_mul_scalars:
            if mul_scalar is not None:
                mul_v = self.alloc_scratch(f"hash_mul_v_{hi}", VLEN)
                hash_mul_vecs.append(mul_v)
            else:
                hash_mul_vecs.append(None)

        # Setup for block offset loading
        vec_batch_temp = (batch_size // VLEN) * VLEN
        offset_values = list(range(0, vec_batch_temp, VLEN))  # 0, 8, 16, ..., 248

        # Allocate all block offset addresses upfront
        offset_addrs = []
        for i in range(len(offset_values)):
            addr = self.alloc_scratch(f"block_off_{i}")
            self.const_map[offset_values[i]] = addr
            offset_addrs.append(addr)

        # Load only base offsets (every 4th); compute the rest with ALU adds.
        offset_base_indices = list(range(0, len(offset_values), 4))
        offset_loads = [
            (offset_addrs[i], offset_values[i]) for i in offset_base_indices
        ]
        eight_const = self.alloc_scratch("eight_const")
        sixteen_const = self.alloc_scratch("sixteen_const")
        twentyfour_const = self.alloc_scratch("twentyfour_const")
        self.const_map[8] = eight_const
        self.const_map[16] = sixteen_const
        self.const_map[24] = twentyfour_const
        offset_loads.append((eight_const, 8))
        offset_loads.append((sixteen_const, 16))
        offset_loads.append((twentyfour_const, 24))

        # Phase 3: Interleave vbroadcasts with block offset loads
        # We have 12 vbroadcasts (6 c1 + 6 c3) and 3 mul broadcasts = 15 valu ops
        # Plus 11 block offset const loads (base offsets + 8/16/24 constants)

        # Strategy: pair const loads with vbroadcasts (2 loads + up to 6 valu per cycle)
        broadcast_queue = []
        for i in range(len(HASH_STAGES)):
            broadcast_queue.append(("vbroadcast", hash_c1_vecs[i], hash_c1_scalars[i]))
        for i in range(len(HASH_STAGES)):
            broadcast_queue.append(("vbroadcast", hash_c3_vecs[i], hash_c3_broadcast_scalars[i]))
        for hi in range(len(HASH_STAGES)):
            if hash_mul_vecs[hi] is not None:
                broadcast_queue.append(("vbroadcast", hash_mul_vecs[hi], hash_mul_scalars[hi][1]))

        # Pack: 2 const loads + 6 vbroadcasts per cycle while we have both
        bc_idx = 0
        const_idx = 0
        while bc_idx < len(broadcast_queue) or const_idx < len(offset_loads):
            bundle = {}

            # Add up to 6 vbroadcasts
            valu_ops = []
            while len(valu_ops) < 6 and bc_idx < len(broadcast_queue):
                valu_ops.append(broadcast_queue[bc_idx])
                bc_idx += 1
            if valu_ops:
                bundle["valu"] = valu_ops

            # Add up to 2 const loads
            load_ops = []
            while len(load_ops) < 2 and const_idx < len(offset_loads):
                addr, val = offset_loads[const_idx]
                load_ops.append(("const", addr, val))
                const_idx += 1
            if load_ops:
                bundle["load"] = load_ops

            if bundle:
                self.add_bundle(bundle)

        # Compute remaining block offsets using ALU adds from the base offsets.
        offset_alu_ops = []
        for base_idx in offset_base_indices:
            base_addr = offset_addrs[base_idx]
            offset_alu_ops.append(("+", offset_addrs[base_idx + 1], base_addr, eight_const))
            offset_alu_ops.append(("+", offset_addrs[base_idx + 2], base_addr, sixteen_const))
            offset_alu_ops.append(("+", offset_addrs[base_idx + 3], base_addr, twentyfour_const))
            if len(offset_alu_ops) == SLOT_LIMITS["alu"]:
                self.add_bundle({"alu": offset_alu_ops})
                offset_alu_ops = []
        if offset_alu_ops:
            self.add_bundle({"alu": offset_alu_ops})

        body_instrs = []

        stage_buffers = []
        vec_batch_size = (batch_size // VLEN) * VLEN
        vec_block_count = vec_batch_size // VLEN
        pipeline_buffers = min(13, vec_block_count) if vec_block_count else 0  # optimized from 10
        for bi in range(pipeline_buffers):
            stage_buffers.append({
                "idx": self.alloc_scratch(f"idx_v{bi}", VLEN),
                "val": self.alloc_scratch(f"val_v{bi}", VLEN),
                "node": self.alloc_scratch(f"node_val_v{bi}", VLEN),
                "addr": self.alloc_scratch(f"addr_v{bi}", VLEN),
                "tmp1": self.alloc_scratch(f"tmp1_v{bi}", VLEN),
                "tmp2": self.alloc_scratch(f"tmp2_v{bi}", VLEN),
                "cond": self.alloc_scratch(f"cond_v{bi}", VLEN),
                "idx_addr": self.alloc_scratch(f"idx_addr{bi}"),
                "val_addr": self.alloc_scratch(f"val_addr{bi}"),
            })

        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        def schedule_vector_pipeline():
            if vec_block_count == 0:
                return []
            instrs = []
            active_blocks = []
            free_buffers = list(range(pipeline_buffers))
            next_block_index = 0

            def start_next_block():
                nonlocal next_block_index
                if next_block_index >= vec_block_count or not free_buffers:
                    return False
                buf_idx = free_buffers.pop(0)
                active_blocks.append({
                    "block": next_block_index,
                    "buf_idx": buf_idx,
                    "buf": stage_buffers[buf_idx],
                    "offset": offset_addrs[next_block_index],
                    "phase": "init_addr",
                    "round": 0,
                    "stage": 0,
                    "gather": 0,
                })
                next_block_index += 1
                return True

            while free_buffers and next_block_index < vec_block_count:
                start_next_block()

            while active_blocks or next_block_index < vec_block_count:
                while free_buffers and next_block_index < vec_block_count:
                    start_next_block()

                alu_ops = []
                load_ops = []
                valu_ops = []
                store_ops = []
                flow_ops = []

                alu_slots = SLOT_LIMITS["alu"]
                load_slots = SLOT_LIMITS["load"]
                valu_slots = SLOT_LIMITS["valu"]
                store_slots = SLOT_LIMITS["store"]
                flow_slots = SLOT_LIMITS["flow"]

                scheduled_this_cycle = set()

                # Wrap threshold: only need bounds check after round 9 for n_nodes=2047
                # Max idx after round r is 2^(r+2) - 2
                # After round 9: 2046 < 2047 (no wrap), after round 10: 4094 > 2047 (wrap)
                wrap_round = 10

                def next_round_phase(current_round):
                    """Determine next phase after completing a round."""
                    next_r = current_round + 1
                    if next_r >= rounds:
                        return "store_both"

                    # Calculate effective depth
                    # For rounds 0-wrap_round, depth = next_r
                    # For rounds after wrap (11+), depth restarts from 0
                    # After wrap at round 10, indices reset to 0, so:
                    #   round 11 = depth 0, round 12 = depth 1, etc.
                    if next_r <= wrap_round:
                        depth = next_r
                    else:
                        # After wrap: round 11 = depth 0, round 12 = depth 1, etc.
                        depth = next_r - wrap_round - 1

                    # Use selection for depths 0-2 (both before and after wrap).
                    # Depth 3+ selection adds too much VALU - use gather instead.

                    if depth == 0:
                        # Round 0 never hits this (special-cased in vload)
                        # But round 11 (after wrap) does
                        return "round0_xor"
                    elif depth == 1:
                        return "round1_select"
                    elif depth == 2:
                        return "round2_select1"
                    else:
                        return "addr"  # Gather for depth >= 3

                # Priority 1: Flow operations (vselect for bounds) - only for rounds >= wrap_round
                update4_blocks = [b for b in active_blocks if b["phase"] == "update4" and b["block"] not in scheduled_this_cycle]
                update4_blocks.sort(key=lambda b: b["round"], reverse=True)
                for block in update4_blocks:
                    if flow_slots == 0:
                        break
                    buf = block["buf"]
                    flow_ops.append(("vselect", buf["idx"], buf["cond"], buf["idx"], zero_v))
                    block["round"] += 1
                    block["stage"] = 0
                    block["gather"] = 0
                    block["next_phase"] = next_round_phase(block["round"] - 1)
                    scheduled_this_cycle.add(block["block"])
                    flow_slots -= 1

                # Priority 2: Stores (2 per cycle)
                for block in active_blocks:
                    if store_slots == 0:
                        break
                    if block["phase"] == "store_both" and block["block"] not in scheduled_this_cycle:
                        buf = block["buf"]
                        store_ops.append(("vstore", buf["val_addr"], buf["val"]))
                        block["next_phase"] = "store_idx"
                        scheduled_this_cycle.add(block["block"])
                        store_slots -= 1

                for block in active_blocks:
                    if store_slots == 0:
                        break
                    if block["phase"] == "store_idx" and block["block"] not in scheduled_this_cycle:
                        buf = block["buf"]
                        store_ops.append(("vstore", buf["idx_addr"], buf["idx"]))
                        block["next_phase"] = "done"
                        scheduled_this_cycle.add(block["block"])
                        store_slots -= 1

                # Priority 3: Vloads (need 2 load slots)
                for block in active_blocks:
                    if load_slots < 2:
                        break
                    if block["phase"] == "vload" and block["block"] not in scheduled_this_cycle:
                        buf = block["buf"]
                        load_ops.append(("vload", buf["idx"], buf["idx_addr"]))
                        load_ops.append(("vload", buf["val"], buf["val_addr"]))
                        # For round 0, skip addr and gather phases (all idx=0, use root_vec)
                        if block["round"] == 0:
                            block["next_phase"] = "round0_xor"
                        else:
                            block["next_phase"] = "addr"
                        scheduled_this_cycle.add(block["block"])
                        load_slots -= 2

                # Priority 4: Gather loads (fill remaining load slots from multiple blocks)
                # Skip gather for round 0 - use root_vec directly
                for block in active_blocks:
                    if load_slots == 0:
                        break
                    if block["phase"] == "gather":
                        if block["round"] == 0:
                            # Round 0: all idx=0, use preloaded tree[0]
                            buf = block["buf"]
                            # Copy root_vec to node (this is a valu op, handled below)
                            block["next_phase"] = "round0_xor"
                            scheduled_this_cycle.add(block["block"])
                        else:
                            buf = block["buf"]
                            while load_slots > 0 and block["gather"] < VLEN:
                                lane = block["gather"]
                                load_ops.append(("load_offset", buf["node"], buf["addr"], lane))
                                block["gather"] += 1
                                load_slots -= 1
                            if block["gather"] >= VLEN:
                                block["next_phase"] = "xor"
                                scheduled_this_cycle.add(block["block"])

                # Priority 5: VALU operations - unified scheduling to fill all 6 slots
                # Build a list of all schedulable VALU tasks with their costs and priorities
                valu_tasks = []
                for block in active_blocks:
                    if block["block"] in scheduled_this_cycle:
                        continue
                    phase = block["phase"]
                    buf = block["buf"]
                    # Priority 0 = highest (closer to completion)
                    if phase == "update3":
                        valu_tasks.append((0, 1, block, "update3"))
                    elif phase == "update2":
                        valu_tasks.append((1, 1, block, "update2"))
                    elif phase == "update1":
                        valu_tasks.append((2, 2, block, "update1"))
                    elif phase == "hash_op2":
                        valu_tasks.append((5, 1, block, "hash_op2"))
                    elif phase == "hash_mul":
                        valu_tasks.append((3, 1, block, "hash_mul"))
                    elif phase == "hash_op1":
                        valu_tasks.append((5, 1, block, "hash_op1"))
                    elif phase == "xor":
                        valu_tasks.append((7, 1, block, "xor"))
                    elif phase == "round0_xor":
                        valu_tasks.append((7, 1, block, "round0_xor"))
                    elif phase == "round1_select":
                        valu_tasks.append((7, 1, block, "round1_select"))  # optimized from 6 to 7
                    elif phase == "round2_select1":
                        valu_tasks.append((6, 1, block, "round2_select1"))  # offset, priority 6
                    elif phase == "round2_select2":
                        valu_tasks.append((4, 1, block, "round2_select2"))  # sel1, priority 4 (optimized)
                    elif phase == "round2_select3":
                        valu_tasks.append((6, 2, block, "round2_select3"))  # low, high, priority 6
                    elif phase == "round2_select4":
                        valu_tasks.append((6, 1, block, "round2_select4"))  # diff
                    elif phase == "round2_select5":
                        valu_tasks.append((6, 1, block, "round2_select5"))  # node
                    elif phase == "addr":
                        # Optimized priority (5) for better scheduling
                        valu_tasks.append((5, 1, block, "addr"))

                # Sort by priority (lower = higher priority)
                valu_tasks.sort(key=lambda x: x[0])

                # Schedule tasks greedily
                for priority, cost, block, phase in valu_tasks:
                    if block["block"] in scheduled_this_cycle:
                        continue
                    buf = block["buf"]

                    if phase == "hash_op1":
                        hi = block["stage"]
                        op1 = HASH_STAGES[hi][0]
                        op3 = HASH_STAGES[hi][3]
                        # Prefer offloading op3 shift to scalar ALU (8 lanes) when slots allow.
                        if alu_slots >= VLEN and valu_slots >= 1:
                            valu_ops.append((op1, buf["tmp1"], buf["val"], hash_c1_vecs[hi]))
                            for lane in range(VLEN):
                                alu_ops.append((op3, buf["tmp2"] + lane, buf["val"] + lane, hash_c3_lane_scalars[hi]))
                            alu_slots -= VLEN
                            valu_slots -= 1
                            block["next_phase"] = "hash_op2"
                            scheduled_this_cycle.add(block["block"])
                            continue
                        if valu_slots >= 2:
                            valu_ops.append((op1, buf["tmp1"], buf["val"], hash_c1_vecs[hi]))
                            valu_ops.append((op3, buf["tmp2"], buf["val"], hash_c3_vecs[hi]))
                            valu_slots -= 2
                            block["next_phase"] = "hash_op2"
                            scheduled_this_cycle.add(block["block"])
                            continue
                        continue

                    if phase == "update1":
                        # If ALU slots are available, offload the parity AND to scalar ALU.
                        if alu_slots >= VLEN and valu_slots >= 1:
                            for lane in range(VLEN):
                                alu_ops.append(("&", buf["tmp1"] + lane, buf["val"] + lane, one_const))
                            alu_slots -= VLEN
                            valu_ops.append(("multiply_add", buf["idx"], buf["idx"], two_v, one_v))
                            valu_slots -= 1
                            block["next_phase"] = "update2"
                            scheduled_this_cycle.add(block["block"])
                            continue

                    if valu_slots < cost:
                        continue

                    if phase == "update3":
                        valu_ops.append(("<", buf["cond"], buf["idx"], n_nodes_v))
                        block["next_phase"] = "update4"
                    elif phase == "update2":
                        valu_ops.append(("+", buf["idx"], buf["idx"], buf["tmp1"]))
                        # Skip wrap check for rounds where idx can't exceed n_nodes:
                        # - Rounds 0-9 (before wrap): idx grows but stays < n_nodes
                        # - Rounds 11-15 (after wrap): idx reset to 0 and grows small
                        # Only round 10 (wrap_round) needs the wrap check
                        if block["round"] != wrap_round:
                            block["round"] += 1
                            block["stage"] = 0
                            block["gather"] = 0
                            block["next_phase"] = next_round_phase(block["round"] - 1)
                        else:
                            block["next_phase"] = "update3"
                    elif phase == "update1":
                        valu_ops.append(("&", buf["tmp1"], buf["val"], one_v))
                        valu_ops.append(("multiply_add", buf["idx"], buf["idx"], two_v, one_v))
                        block["next_phase"] = "update2"
                    elif phase == "hash_op2":
                        hi = block["stage"]
                        op2 = HASH_STAGES[hi][2]
                        valu_ops.append((op2, buf["val"], buf["tmp1"], buf["tmp2"]))
                        if hi + 1 == len(HASH_STAGES):
                            block["next_phase"] = "update1"
                        else:
                            block["stage"] = hi + 1
                            block["next_phase"] = "hash_mul" if hash_mul_vecs[hi + 1] is not None else "hash_op1"
                    elif phase == "hash_mul":
                        hi = block["stage"]
                        mul_v = hash_mul_vecs[hi]
                        valu_ops.append(("multiply_add", buf["val"], buf["val"], mul_v, hash_c1_vecs[hi]))
                        if hi + 1 == len(HASH_STAGES):
                            block["next_phase"] = "update1"
                        else:
                            block["stage"] = hi + 1
                            block["next_phase"] = "hash_mul" if hash_mul_vecs[hi + 1] is not None else "hash_op1"
                    elif phase == "xor":
                        valu_ops.append(("^", buf["val"], buf["val"], buf["node"]))
                        block["next_phase"] = "hash_mul" if hash_mul_vecs[0] is not None else "hash_op1"
                    elif phase == "round0_xor":
                        # Round 0: XOR with preloaded tree[0] vector
                        valu_ops.append(("^", buf["val"], buf["val"], root_vec))
                        block["next_phase"] = "hash_mul" if hash_mul_vecs[0] is not None else "hash_op1"
                    elif phase == "round1_select":
                        # Round 1: idx in {1,2}. tmp1 already holds parity == idx - 1.
                        # node = tree1 + tmp1 * (tree2 - tree1)
                        valu_ops.append(("multiply_add", buf["node"], delta_lr_vec, buf["tmp1"], left_vec))
                        block["next_phase"] = "xor"
                    elif phase == "round2_select1":
                        # Round 2: compute offset = idx - 3 into tmp2 (preserve tmp1 parity)
                        # Then compute sel1 = offset >> 1 in next cycle
                        valu_ops.append(("-", buf["tmp2"], buf["idx"], three_vec))
                        block["next_phase"] = "round2_select2"
                    elif phase == "round2_select2":
                        # Compute sel1 = offset >> 1
                        # Try to also do low/high computations if we have slots (they don't depend on cond yet)
                        valu_ops.append((">>", buf["cond"], buf["tmp2"], one_v))  # sel1
                        block["next_phase"] = "round2_select3"
                    elif phase == "round2_select3":
                        # Compute low/high; sel0 == parity in tmp1
                        valu_ops.append(("multiply_add", buf["tmp2"], delta_3_4_vec, buf["tmp1"], node3_vec))  # low
                        valu_ops.append(("multiply_add", buf["node"], delta_5_6_vec, buf["tmp1"], node5_vec))  # high
                        block["next_phase"] = "round2_select4"
                    elif phase == "round2_select4":
                        # Compute diff = high - low, then final selection if we have 2 slots
                        # node = node - tmp2, then node = node * cond + tmp2
                        # These have RAW dependency (second reads node from first), so can't combine
                        valu_ops.append(("-", buf["node"], buf["node"], buf["tmp2"]))
                        block["next_phase"] = "round2_select5"
                    elif phase == "round2_select5":
                        # Final selection: node = low + sel1 * diff
                        valu_ops.append(("multiply_add", buf["node"], buf["node"], buf["cond"], buf["tmp2"]))
                        block["next_phase"] = "xor"
                    elif phase == "addr":
                        valu_ops.append(("+", buf["addr"], buf["idx"], forest_base_v))
                        block["next_phase"] = "gather"

                    scheduled_this_cycle.add(block["block"])
                    valu_slots -= cost

                # Priority 6: ALU for init_addr (2 slots each)
                for block in active_blocks:
                    if alu_slots < 2:
                        break
                    if block["phase"] == "init_addr" and block["block"] not in scheduled_this_cycle:
                        buf = block["buf"]
                        alu_ops.append(("+", buf["idx_addr"], self.scratch["inp_indices_p"], block["offset"]))
                        alu_ops.append(("+", buf["val_addr"], self.scratch["inp_values_p"], block["offset"]))
                        block["next_phase"] = "vload"
                        scheduled_this_cycle.add(block["block"])
                        alu_slots -= 2

                if not (alu_ops or load_ops or valu_ops or store_ops or flow_ops):
                    # Check if any block is in gather phase but wasn't fully scheduled
                    stuck = False
                    for block in active_blocks:
                        if block["phase"] == "gather" and block["gather"] < VLEN:
                            stuck = True
                            break
                    if not stuck:
                        raise RuntimeError("scheduler made no progress")
                    # Otherwise we need another cycle to continue gather
                    continue

                instr = {}
                if alu_ops:
                    instr["alu"] = alu_ops
                if load_ops:
                    instr["load"] = load_ops
                if valu_ops:
                    instr["valu"] = valu_ops
                if store_ops:
                    instr["store"] = store_ops
                if flow_ops:
                    instr["flow"] = flow_ops
                instrs.append(instr)

                # Apply state transitions
                new_active = []
                for block in active_blocks:
                    next_phase = block.pop("next_phase", None)
                    if next_phase:
                        block["phase"] = next_phase
                    if block["phase"] == "done":
                        free_buffers.append(block["buf_idx"])
                    else:
                        new_active.append(block)
                active_blocks = new_active

            return instrs

        body_instrs.extend(schedule_vector_pipeline())

        for round_i in range(rounds):
            for i in range(vec_batch_size, batch_size):
                tail_slots = []
                i_const = self.scratch_const(i)
                tail_slots.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                tail_slots.append(("load", ("load", tmp_idx, tmp_addr)))
                tail_slots.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                tail_slots.append(("load", ("load", tmp_val, tmp_addr)))
                tail_slots.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                tail_slots.append(("load", ("load", tmp_node_val, tmp_addr)))
                tail_slots.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                tail_slots.extend(self.build_hash(tmp_val, tmp1, tmp2, round_i, i))
                tail_slots.append(("alu", ("%", tmp1, tmp_val, two_const)))
                tail_slots.append(("alu", ("==", tmp1, tmp1, zero_const)))
                tail_slots.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                tail_slots.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                tail_slots.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                tail_slots.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                tail_slots.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                tail_slots.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                tail_slots.append(("store", ("store", tmp_addr, tmp_idx)))
                tail_slots.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                tail_slots.append(("store", ("store", tmp_addr, tmp_val)))
                body_instrs.extend(self.build(tail_slots))

        self.instrs.extend(body_instrs)
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
    for ref_mem in reference_kernel2(mem, value_trace):
        pass
    machine.run()
    inp_values_p = ref_mem[6]
    if prints:
        print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
        print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
    assert (
        machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    ), "Incorrect result on final values"
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
