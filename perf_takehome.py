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

import heapq
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
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vconst_map = {}
        self.engines = []
        self.slots = []
        self.succs = []
        self.indeg = []
        self.last_writer = {}
        self.last_reader = {}
        self.war_addrs = set()
        self.instrs = []

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def _vec_range(self, base):
        return list(range(base, base + VLEN))

    def rw(self, engine, slot):
        if engine == "alu":
            _, dest, a1, a2 = slot
            return [a1, a2], [dest]
        if engine == "load":
            match slot:
                case ("const", dest, _val):
                    return [], [dest]
                case ("load", dest, addr):
                    return [addr], [dest]
                case ("vload", dest, addr):
                    return [addr], self._vec_range(dest)
                case ("load_offset", dest, addr, offset):
                    return [addr + offset], [dest + offset]
                case _:
                    raise NotImplementedError(f"Unknown load op {slot}")
        if engine == "valu":
            match slot:
                case ("vbroadcast", dest, src):
                    return [src], self._vec_range(dest)
                case ("multiply_add", dest, a, b, c):
                    reads = self._vec_range(a) + self._vec_range(b) + self._vec_range(c)
                    return reads, self._vec_range(dest)
                case (op, dest, a1, a2):
                    reads = self._vec_range(a1) + self._vec_range(a2)
                    return reads, self._vec_range(dest)
                case _:
                    raise NotImplementedError(f"Unknown valu op {slot}")
        if engine == "store":
            match slot:
                case ("vstore", addr, src):
                    reads = [addr] + self._vec_range(src)
                    return reads, []
                case ("store", addr, src):
                    return [addr, src], []
                case _:
                    raise NotImplementedError(f"Unknown store op {slot}")
        raise NotImplementedError(f"Unknown engine {engine}")

    def emit_op(self, engine, slot):
        reads, writes = self.rw(engine, slot)
        deps = set()
        for r in reads:
            if r in self.last_writer:
                deps.add(self.last_writer[r])
            if r in self.war_addrs and r in self.last_reader:
                deps.add(self.last_reader[r])
        for w in writes:
            if w in self.last_writer:
                deps.add(self.last_writer[w])
            if w in self.war_addrs and w in self.last_reader:
                deps.add(self.last_reader[w])
        op_id = len(self.engines)
        self.engines.append(engine)
        self.slots.append(slot)
        self.succs.append([])
        self.indeg.append(len(deps))
        for dep in deps:
            self.succs[dep].append(op_id)
        for r in reads:
            if r in self.war_addrs:
                self.last_reader[r] = op_id
        for w in writes:
            self.last_writer[w] = op_id
            if w in self.war_addrs and w in self.last_reader:
                del self.last_reader[w]
        return op_id

    def const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.emit_op("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def vconst(self, val, name=None):
        if val in self.vconst_map:
            return self.vconst_map[val]
        scalar = self.const(val, name=name)
        vec = self.alloc_scratch(f"{name}_vec" if name else None, VLEN)
        self.emit_op("valu", ("vbroadcast", vec, scalar))
        self.vconst_map[val] = vec
        return vec

    def vbroadcast(self, src, name=None):
        vec = self.alloc_scratch(f"{name}_vec" if name else None, VLEN)
        self.emit_op("valu", ("vbroadcast", vec, src))
        return vec

    def schedule(self):
        n_ops = len(self.engines)
        inf = n_ops + 1
        load_dist = [inf] * n_ops
        for op_id in range(n_ops - 1, -1, -1):
            if self.engines[op_id] == "load":
                load_dist[op_id] = 0
            for succ in self.succs[op_id]:
                cand = load_dist[succ] + 1
                if cand < load_dist[op_id]:
                    load_dist[op_id] = cand

        indeg = self.indeg.copy()
        ready = {}
        for engine in SLOT_LIMITS:
            if engine == "debug":
                continue
            ready[engine] = []

        def perm(engine, op_id):
            if engine == "load":
                return (op_id ^ (op_id >> 1)) & 0xFFFF
            if engine == "alu":
                return op_id
            if engine == "valu":
                return -op_id
            return op_id

        for op_id, deg in enumerate(indeg):
            if deg == 0:
                engine = self.engines[op_id]
                heapq.heappush(
                    ready[engine],
                    (load_dist[op_id], perm(engine, op_id), op_id),
                )

        instrs = []
        while any(ready[engine] for engine in ready):
            bundle = {}
            next_ready = {engine: [] for engine in ready}
            for engine, limit in SLOT_LIMITS.items():
                if engine == "debug":
                    continue
                slots = []
                for _ in range(limit):
                    if not ready[engine]:
                        break
                    _, _, op_id = heapq.heappop(ready[engine])
                    slots.append(op_id)
                if slots:
                    bundle[engine] = [self.slots[op_id] for op_id in slots]
                    for op_id in slots:
                        for succ in self.succs[op_id]:
                            indeg[succ] -= 1
                            if indeg[succ] == 0:
                                succ_engine = self.engines[succ]
                                heapq.heappush(
                                    next_ready[succ_engine],
                                    (
                                        load_dist[succ],
                                        perm(succ_engine, succ),
                                        succ,
                                    ),
                                )
            if bundle:
                instrs.append(bundle)
            for engine in ready:
                if next_ready[engine]:
                    for item in next_ready[engine]:
                        heapq.heappush(ready[engine], item)
        self.instrs = instrs

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized kernel with DAG scheduling and multiply_add folding.
        """
        n_groups = batch_size // VLEN

        idx_base = self.alloc_scratch("idx", batch_size)
        val_base = self.alloc_scratch("val", batch_size)
        node_base = self.alloc_scratch("node", batch_size)
        tmp_base = self.alloc_scratch("tmp", batch_size)
        aux_base = self.alloc_scratch("aux", batch_size)
        idx_addr_base = self.alloc_scratch("idx_addr", n_groups)
        val_addr_base = self.alloc_scratch("val_addr", n_groups)

        self.war_addrs = set(range(node_base, node_base + batch_size))
        self.war_addrs.update(range(tmp_base, tmp_base + batch_size))

        forest_base = self.const(7, "forest_base")
        zero = self.const(0, "zero")
        forest_base_vec = self.vconst(7, "forest_base")
        shift1 = self.const(1, "shift1")
        shift9 = self.const(9, "shift9")
        shift16 = self.const(16, "shift16")
        shift19 = self.const(19, "shift19")
        vlen = self.const(VLEN, "vlen")
        one_vec = self.vconst(1, "one")
        two_vec = self.vconst(2, "two")
        three_vec = self.vconst(3, "three")

        mul4097_vec = self.vconst(1 + (1 << 12), "mul4097")
        mul33_vec = self.vconst(1 + (1 << 5), "mul33")
        mul9_vec = self.vconst(1 + (1 << 3), "mul9")
        add1_vec = self.vconst(0x7ED55D16, "add1")
        xor2_vec = self.vconst(0xC761C23C, "xor2")
        add3_vec = self.vconst(0x165667B1, "add3")
        add4_vec = self.vconst(0xD3A2646C, "add4")
        add5_vec = self.vconst(0xFD7046C5, "add5")
        xor6_vec = self.vconst(0xB55A4F09, "xor6")

        def emit_shift(op, dest_vec, src_vec, shift_scalar):
            for lane in range(VLEN):
                self.emit_op(
                    "alu",
                    (op, dest_vec + lane, src_vec + lane, shift_scalar),
                )


        top_nodes = self.alloc_scratch("top_nodes", VLEN)
        self.emit_op("load", ("vload", top_nodes, forest_base))

        d12 = self.alloc_scratch("d12")
        d34 = self.alloc_scratch("d34")
        d56 = self.alloc_scratch("d56")
        self.emit_op("alu", ("^", d12, top_nodes + 1, top_nodes + 2))
        self.emit_op("alu", ("^", d34, top_nodes + 3, top_nodes + 4))
        self.emit_op("alu", ("^", d56, top_nodes + 5, top_nodes + 6))

        node0_vec = self.vbroadcast(top_nodes + 0, "node0")
        node2_vec = self.vbroadcast(top_nodes + 2, "node2")
        node3_vec = self.vbroadcast(top_nodes + 3, "node3")
        node5_vec = self.vbroadcast(top_nodes + 5, "node5")
        d12_vec = self.vbroadcast(d12, "d12")
        d34_vec = self.vbroadcast(d34, "d34")
        d56_vec = self.vbroadcast(d56, "d56")

        inp_indices_p = 7 + n_nodes
        inp_values_p = inp_indices_p + batch_size

        self.emit_op("load", ("const", idx_addr_base + 0, inp_indices_p))
        self.emit_op("load", ("const", val_addr_base + 0, inp_values_p))
        for g in range(1, n_groups):
            self.emit_op("alu", ("+", idx_addr_base + g, idx_addr_base + g - 1, vlen))
            self.emit_op("alu", ("+", val_addr_base + g, val_addr_base + g - 1, vlen))

        for g in range(n_groups):
            idx_addr = idx_addr_base + g
            val_addr = val_addr_base + g
            idx_vec = idx_base + g * VLEN
            val_vec = val_base + g * VLEN
            self.emit_op("valu", ("vbroadcast", idx_vec, zero))
            self.emit_op("load", ("vload", val_vec, val_addr))

        for r in range(rounds):
            depth = r % (forest_height + 1)
            for g in range(n_groups):
                idx_vec = idx_base + g * VLEN
                val_vec = val_base + g * VLEN
                node_vec = node_base + g * VLEN
                tmp_vec = tmp_base + g * VLEN
                aux_vec = aux_base + g * VLEN

                if depth >= 3:
                    self.emit_op("valu", ("+", aux_vec, idx_vec, forest_base_vec))
                    for off in range(VLEN):
                        self.emit_op(
                            "load",
                            ("load_offset", node_vec, aux_vec, off),
                        )
                    self.emit_op("valu", ("^", val_vec, val_vec, node_vec))
                elif depth == 0:
                    self.emit_op("valu", ("^", val_vec, val_vec, node0_vec))
                elif depth == 1:
                    self.emit_op("valu", ("&", tmp_vec, idx_vec, one_vec))
                    self.emit_op("valu", ("*", node_vec, d12_vec, tmp_vec))
                    self.emit_op("valu", ("^", val_vec, val_vec, node2_vec))
                    self.emit_op("valu", ("^", val_vec, val_vec, node_vec))
                else:
                    self.emit_op("valu", ("-", aux_vec, idx_vec, three_vec))
                    self.emit_op("valu", ("&", node_vec, aux_vec, one_vec))
                    self.emit_op("valu", ("*", tmp_vec, d34_vec, node_vec))
                    self.emit_op("valu", ("^", tmp_vec, tmp_vec, node3_vec))
                    self.emit_op("valu", ("*", node_vec, d56_vec, node_vec))
                    self.emit_op("valu", ("^", node_vec, node_vec, node5_vec))
                    self.emit_op("valu", ("^", val_vec, val_vec, tmp_vec))
                    self.emit_op("valu", ("^", node_vec, node_vec, tmp_vec))
                    emit_shift(">>", tmp_vec, aux_vec, shift1)
                    self.emit_op("valu", ("*", node_vec, node_vec, tmp_vec))
                    self.emit_op("valu", ("^", val_vec, val_vec, node_vec))

                self.emit_op(
                    "valu",
                    ("multiply_add", val_vec, val_vec, mul4097_vec, add1_vec),
                )
                emit_shift(">>", tmp_vec, val_vec, shift19)
                self.emit_op("valu", ("^", tmp_vec, tmp_vec, xor2_vec))
                self.emit_op("valu", ("^", val_vec, val_vec, tmp_vec))
                self.emit_op(
                    "valu",
                    ("multiply_add", val_vec, val_vec, mul33_vec, add3_vec),
                )
                emit_shift("<<", tmp_vec, val_vec, shift9)
                self.emit_op("valu", ("+", node_vec, val_vec, add4_vec))
                self.emit_op("valu", ("^", val_vec, node_vec, tmp_vec))
                self.emit_op(
                    "valu",
                    ("multiply_add", val_vec, val_vec, mul9_vec, add5_vec),
                )
                emit_shift(">>", tmp_vec, val_vec, shift16)
                self.emit_op("valu", ("^", tmp_vec, tmp_vec, xor6_vec))
                self.emit_op("valu", ("^", val_vec, val_vec, tmp_vec))

                if depth == forest_height:
                    self.emit_op(
                        "valu", ("^", idx_vec, idx_vec, idx_vec)
                    )
                else:
                    self.emit_op("valu", ("&", tmp_vec, val_vec, one_vec))
                    self.emit_op("valu", ("+", tmp_vec, tmp_vec, one_vec))
                    self.emit_op(
                        "valu", ("multiply_add", idx_vec, idx_vec, two_vec, tmp_vec)
                    )

        for g in range(n_groups):
            idx_vec = idx_base + g * VLEN
            val_vec = val_base + g * VLEN
            self.emit_op("store", ("vstore", idx_addr_base + g, idx_vec))
            self.emit_op("store", ("vstore", val_addr_base + g, val_vec))

        self.schedule()

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
    machine.run()
    for ref_mem in reference_kernel2(mem, value_trace):
        pass
    inp_values_p = ref_mem[6]
    if prints:
        print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
        print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
    assert (
        machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    ), "Incorrect output values"
    inp_indices_p = ref_mem[5]
    if prints:
        print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
    assert (
        machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)]
        == ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)]
    ), "Incorrect output indices"

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
