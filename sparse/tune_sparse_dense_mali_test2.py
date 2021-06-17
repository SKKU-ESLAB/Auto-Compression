# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Auto-scheduling Sparse Matrix Multiplication on CPU with Custom Sketch Rule
===========================================================================
**Author**: `Chengfan Jia <https://github.com/jcf94/>`_

This is a tutorial on how to use the auto-scheduler to tune a sparse matrix multiplication for
CPUs.

Auto-scheduler is designed to explore the schedule with best performance for a given computation
declaration automatically. While sometimes, we may have a demand to try some special ops which may
not been well-supported by auto-scheduler's default sketch rules and result in poor performance.
Fortunately, auto-scheduler currently allows user to provide a CustomSketch to cover these cases.

We use sparse matrix multiplication as an example in this tutorial to demonstrate how to implement
and plug a custom sketch rule to the auto-scheduler's search policy.

Note that this tutorial will not run on Windows or recent versions of macOS. To
get it to run, you will need to wrap the body of this tutorial in a :code:`if
__name__ == "__main__":` block.
"""

import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, runtime, topi
from tvm.auto_scheduler import _ffi_api
from tvm.contrib import utils
from tvm.topi.utils import get_const_tuple
from tvm.topi.sparse.utils import random_bsr_matrix

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hw', default=32, type=int)
parser.add_argument('--ci', default=1024, type=int)
parser.add_argument('--co', default=1024, type=int)
parser.add_argument('--density', default=0.1, type=float)
parser.add_argument('--bso', default=4, type=int)
parser.add_argument('--bsi', default=1, type=int)
parser.add_argument('--data-layout', default='hwc', type=str)
parser.add_argument('--weight-layout', default='oi', type=str)
parser.add_argument('--output-layout', default='hwc', type=str)

ARGS = parser.parse_args()

@auto_scheduler.register_task_input_check_func
def try_get_sparse_dense_v2_input(args):
    sparse_prefix = sparse_data = sparse_indices = sparse_indptr = None

    def _process_inputs(input_tensors, m, n, prefix_init):
        nonlocal sparse_prefix
        nonlocal sparse_data
        nonlocal sparse_indices
        nonlocal sparse_indptr

        assert len(input_tensors) == 4
        unsure_tensors = list(input_tensors)
        # Get the Dense data
        dense_data = None
        for tensor in unsure_tensors:
            if len(tensor.shape) == 2:
                assert dense_data is None
                dense_data = tensor
                #assert m == dense_data.shape[0]
                #k = dense_data.shape[1]
                if m == dense_data.shape[0]:
                    k = dense_data.shape[1]
                else:
                    k = dense_data.shape[0]
        unsure_tensors.remove(dense_data)

        # Get the Sparse data
        sparse_data = None
        for tensor in unsure_tensors:
            if len(tensor.shape) == 3:
                assert sparse_data is None
                sparse_data = tensor
                block_size, bs_r, bs_c = sparse_data.shape
        unsure_tensors.remove(sparse_data)

        # Get the Sparse indptr & indices
        sparse_indices = None
        for tensor in unsure_tensors:
            assert len(tensor.shape) == 1
            if tensor.shape[0] == block_size:
                assert sparse_indices is None
                sparse_indices = tensor
        unsure_tensors.remove(sparse_indices)
        assert len(unsure_tensors) == 1
        sparse_indptr = unsure_tensors[0]

        # Generate the sparse_prefix
        density = 1.0
        for i in sparse_data.shape:
            density *= i
        density /= k * n
        density = density.value
        sparse_prefix = "%s_%d_%d_%d_%d_%.2f_" % (prefix_init, n, k, bs_r, bs_c, density)

    visited = set()

    def _traverse(t):
        # We cannot directly add tensors to the set, because the comparison of
        # two tensors with ndim=0 is ambiguous.
        assert t.handle is not None
        if t.handle.value in visited:
            return

        if isinstance(t.op, te.ComputeOp):
            # TODO(jcf94): Currently only support to one sparse op, add more support here
            if t.op.tag[:-4] == "sparse_dense_v2":
                assert len(t.op.input_tensors) == 1
                block_tensor = t.op.input_tensors[0]
                if block_tensor.op.tag == "sparse_dense_v2_block_hwc":
                    m, n = t.shape
                elif block_tensor.op.tag == "sparse_dense_v2_block_chw":
                    n, m = t.shape
                _process_inputs(block_tensor.op.input_tensors, m, n, "sparse_dense_v2_bsr")
            if sparse_prefix is not None:
                # Early stop if we find a sparse_prefix
                # Notice: If any workload has more than one sparse input, this may get problem
                return
            for x in t.op.input_tensors:
                _traverse(x)
        visited.add(t.handle.value)

    try:
        for arg in args:
            _traverse(arg)
    # pylint: disable=broad-except
    except Exception:
        return {}

    if sparse_data is None or sparse_indices is None or sparse_indptr is None:
        return {}

    sparse_input_map = {}
    sparse_input_map[sparse_data] = sparse_prefix + "W_data"
    sparse_input_map[sparse_indices] = sparse_prefix + "W_indices"
    sparse_input_map[sparse_indptr] = sparse_prefix + "W_indptr"

    return sparse_input_map

def _sparse_dense_sp_rhs_bsrmm(data, weight_data, weight_indices, weight_indptr,
                               data_layout, weight_layout, output_layout):
    if data_layout == 'hwc':
        (m, k) = get_const_tuple(data.shape)
    elif data_layout == 'chw':
        (k, m) = get_const_tuple(data.shape)
    
    if weight_layout == 'oi':
        (nnz, bs_o, bs_i) = get_const_tuple(weight_data.shape)
    elif weight_layout == 'io':
        (nnz, bs_i, bs_o) = get_const_tuple(weight_data.shape)

    (num_blocks_plus_1,) = get_const_tuple(weight_indptr.shape)
    num_blocks = num_blocks_plus_1 - 1

    def _compute_block_hwc(i, nb_j, j):
        row_start = weight_indptr[nb_j]
        row_end = weight_indptr[nb_j + 1]
        row_elems = row_end - row_start
        elem_idx = te.reduce_axis((0, row_elems), name="elem_idx")
        block_offset = row_start + elem_idx
        c = te.reduce_axis((0, bs_i), name="c")
        block_j = weight_indices[block_offset]
        if weight_layout == 'oi':
            block_ij_val = weight_data[block_offset][j][c]
        elif weight_layout == 'io':
            block_ij_val = weight_data[block_offset][c][j]
        if data_layout == 'hwc':
            x_val = data[i, bs_i * block_j + c]
        elif data_layout == 'chw':
            x_val = data[bs_i * block_j + c, i]
        return te.sum(block_ij_val * x_val, axis=[elem_idx, c])

    def _compute_block_chw(nb_j, j, i):
        row_start = weight_indptr[nb_j]
        row_end = weight_indptr[nb_j + 1]
        row_elems = row_end - row_start
        elem_idx = te.reduce_axis((0, row_elems), name="elem_idx")
        block_offset = row_start + elem_idx
        c = te.reduce_axis((0, bs_i), name="c")
        block_j = weight_indices[block_offset]
        if weight_layout == 'oi':
            block_ij_val = weight_data[block_offset][j][c]
        elif weight_layout == 'io':
            block_ij_val = weight_data[block_offset][c][j]
        if data_layout == 'hwc':
            x_val = data[i, bs_i * block_j + c]
        elif data_layout == 'chw':
            x_val = data[bs_i * block_j + c, i]
        return te.sum(block_ij_val * x_val, axis=[elem_idx, c])

    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    if output_layout == 'hwc':
        bsrmm_block = te.compute(
            (m, num_blocks, bs_o),
            _compute_block_hwc,
            tag="sparse_dense_v2_block_hwc",
            attrs={"FLOP": 2 * m * nnz * bs_o * bs_i},
        )
        return te.compute(
            (m, num_blocks * bs_o),
            lambda m, n: bsrmm_block[m, idxd(n, bs_o), idxm(n, bs_o)],
            tag="sparse_dense_v2_hwc",
        )
    elif output_layout == 'chw':
        bsrmm_block = te.compute(
            (num_blocks, bs_o, m),
            _compute_block_chw,
            tag="sparse_dense_v2_block_chw",
            attrs={"FLOP": 2 * m * nnz * bs_o * bs_i},
        )
        return te.compute(
            (num_blocks * bs_o, m),
            lambda n, m: bsrmm_block[idxd(n, bs_o), idxm(n, bs_o), m],
            tag="sparse_dense_v2_chw",
        )

######################################################################
# Define the computation
# ^^^^^^^^^^^^^^^^^^^^^^
# To begin with, let us define the computation of a sparse matmul with several relu and bias add.
# The function should return the list of input/output tensors.
# From these tensors, the auto-scheduler can get the whole computational graph.
@auto_scheduler.register_workload
def sparse_dense_v2(data_shape, w_data_shape, w_indices_shape, w_indptr_shape, dtype, data_layout, weight_layout, output_layout):
    X = te.placeholder(shape=data_shape, dtype=dtype)
    W_data = te.placeholder(shape=w_data_shape, dtype=dtype)
    W_indices = te.placeholder(shape=w_indices_shape, dtype="int32")
    W_indptr = te.placeholder(shape=w_indptr_shape, dtype="int32")

    out = _sparse_dense_sp_rhs_bsrmm(X, W_data, W_indices, W_indptr, data_layout, weight_layout, output_layout)

    return [X, W_data, W_indices, W_indptr, out]


######################################################################
# Special step for sparse workload
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# During schedule tuning, auto-scheduler will use random inputs to measure the performance of a
# generated schedule. While we cannot directly use a random array as the input of a sparse op, for
# the "indices" and "indptr" array are meaningful for the computation.
#
# To solve this problem, we register these as special buffers, and load them when process program
# measuring.
# See the `tvm.auto_scheduler.measure.py` for more details.

# Define the basic shapes of this sparse computation
M = ARGS.hw * ARGS.hw
K = ARGS.ci
N = ARGS.co
BS_O = ARGS.bso
BS_I = ARGS.bsi
density = ARGS.density

# Default
# data: hwc
# weight: oi
# output: hwc

# Generate the test data with numpy
X_np = np.random.randn(M, K).astype("float32")
W_sp_np = random_bsr_matrix(N, K, BS_O, BS_I, density=density, dtype="float32")
W_np = W_sp_np.todense()
Y_np = X_np @ W_np.T  # Process the matrix multiplication

W_sp_np_data = W_sp_np.data
W_sp_np_indices = W_sp_np.indices
W_sp_np_indptr = W_sp_np.indptr

if ARGS.data_layout == 'chw':
    X_np = X_np.transpose(1, 0)
if ARGS.weight_layout == 'io':
    W_np = W_np.transpose(1, 0)
    W_sp_np_data = W_sp_np_data.transpose(0, 2, 1)
if ARGS.output_layout == 'chw':
    Y_np = Y_np.transpose(1, 0)

######################################################################
# Create the search task
# ^^^^^^^^^^^^^^^^^^^^^^
# We then create a search task with M=N=K=512 and dtype="float32"
# If your machine supports avx instructions, you can
#
#   - replace "llvm" below with "llvm -mcpu=core-avx2" to enable AVX2
#   - replace "llvm" below with "llvm -mcpu=skylake-avx512" to enable AVX-512

#target = tvm.target.Target("llvm")
#target = tvm.target.Target("llvm -mtriple=armv7l-linux-gnueabihf -mattr=+neon")
target = tvm.target.Target("opencl -device=mali", host="llvm -mtriple=armv7l-linux-gnueabihf")

device_key = 'xu4'
rpc_host = '115.145.178.78'
rpc_port = 8109

# Register the sparse data to task inputs
#prefix = "sparse_dense_bsr_%d_%d_%d_%d_%.2f_%s_" % (N, K, BS_O, BS_I, density, ARGS.weight_layout)
prefix = "sparse_dense_v2_bsr_%d_%d_%d_%d_%.2f_" % (W_np.shape[0], W_np.shape[1], W_sp_np_data.shape[1], W_sp_np_data.shape[2], density)
task = tvm.auto_scheduler.SearchTask(
    func=sparse_dense_v2,
    args=(X_np.shape, W_sp_np_data.shape, W_sp_np_indices.shape, W_sp_np_indptr.shape, "float32", ARGS.data_layout, ARGS.weight_layout, ARGS.output_layout),
    target=target,
    task_inputs={
        prefix + "W_data": runtime.ndarray.array(W_sp_np_data),
        prefix + "W_indices": runtime.ndarray.array(W_sp_np_indices),
        prefix + "W_indptr": runtime.ndarray.array(W_sp_np_indptr),
    },
    task_inputs_save_to_file=True,
    #layout_rewrite_option=auto_scheduler.LayoutRewriteOption.NO_REWRITE,
)

# Inspect the computational graph
print("Computational DAG:")
print(task.compute_dag)

######################################################################
# Write the custom sketch for sparse dense op
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Before tuning, we will need to define the CustomSketchRule for the sparse dense op.
#
# CustomSketchRule consists of two parts: the condition function and the apply function.
#
#   - condition function: describe when to apply this sketch rule. For example, we can only apply
#     the rule to the sparse ops by matching their name and tag.
#   - apply function: describe how to generate the initial sketch. You can implement it using
#     auto-scheduler provided loop state APIs.


def meet_condition_func(search_policy, state, stage_id):
    state = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
    if state.stages[stage_id].op.tag in [
        "sparse_dense_v2_hwc",
        "sparse_dense_v2_chw",
        "sparse_dense_v2_block_hwc",
        "sparse_dense_v2_block_chw"
    ]:
        return auto_scheduler.PreloadCustomSketchRule.APPLY_AND_SKIP_REST
    else:
        return auto_scheduler.PreloadCustomSketchRule.PASS


def apply_func(search_policy, state, stage_id):
    ret = []
    s0 = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
    if s0.stages[stage_id].op.tag[:-4] == "sparse_dense_v2_block":
        return [s0.state_object, stage_id - 1]

    sparse_dense = s0.stages[stage_id].op
    sparse_dense_block = s0.stages[stage_id - 1].op
    assert sparse_dense.tag[:-4] == "sparse_dense_v2"
    assert sparse_dense_block.tag[:-4] == "sparse_dense_v2_block"

    # Set the default consumer of compute block
    consumer = sparse_dense

    # If sparse dense has a single elementwise consumer
    # We can compute inline the sparse_dense output stage
    consumers = _ffi_api.SearchPolicyUtilsGetConsumers(
        search_policy.search_task, s0.state_object, stage_id
    )
    if len(consumers) == 1:
        consumer_id = int(consumers.items()[0][0])
        if _ffi_api.SearchPolicyUtilsIsElementwiseMatch(
            search_policy.search_task, s0.state_object, stage_id, consumer_id
        ):
            consumer = s0.stages[consumer_id].op
            s0.compute_inline(sparse_dense)

    if ARGS.output_layout == 'hwc':
        i, nb_j, j, row_offset, c = s0[sparse_dense_block].iters
        m, n = s0[consumer].iters

        i0, i1, i2, i3, i4 = s0.split(sparse_dense_block, i, [None, None, None, None])
        m0, m1, m2, m3 = s0.follow_split(consumer, m, len(s0.transform_steps) - 1, 3)

        nb_n, n = s0.split(consumer, n, [j.range.extent])
        #j0, j1, j2 = s0.split(sparse_dense_block, nb_j, [1, 4])
        j0, j1, j2 = s0.split(sparse_dense_block, nb_j, [None, None])
        n0, n1, n2 = s0.follow_split(consumer, nb_n, len(s0.transform_steps) - 1, 2)

        c0, c1, c2 = s0.split(sparse_dense_block, c, [None, None])

        jj0, jj1 = s0.split(sparse_dense_block, j, [4])
        nn0, nn1 = s0.split(consumer, n, [4])

        #s0.reorder(sparse_dense_block, [i0, j0, i1, j1, i2, j2, c0, row_offset, c1, i3, c2, i4, j])
        #s0.reorder(sparse_dense_block, [i0, j0, i1, j1, i2, j2, c0, i3, c1, row_offset, c2, i4, j])
        s0.reorder(sparse_dense_block, [i0, j0, i1, j1, i2, j2, c0, i3, c1, row_offset, c2, i4, jj0, jj1])
        #s0.reorder(consumer, [m0, n0, m1, n1, m2, n2, m3, n])
        s0.reorder(consumer, [m0, n0, m1, n1, m2, n2, m3, nn0, nn1])
        s0.compute_at(sparse_dense_block, consumer, n2)
    elif ARGS.output_layout == 'chw':
        nb_j, j, i, row_offset, c = s0[sparse_dense_block].iters
        n, m = s0[consumer].iters

        nb_n, n = s0.split(consumer, n, [j.range.extent])
        i0, i1, i2, i3, i4 = s0.split(sparse_dense_block, i, [None, None, None, None])
        m0, m1, m2, m3 = s0.follow_split(consumer, m, len(s0.transform_steps) - 1, 3)

        j0, j1, j2 = s0.split(sparse_dense_block, nb_j, [None, None])
        n0, n1, n2 = s0.follow_split(consumer, nb_n, len(s0.transform_steps) - 1, 2)

        c0, c1, c2 = s0.split(sparse_dense_block, c, [None, None])

        s0.reorder(sparse_dense_block, [j0, i0, j1, i1, j2, i2, c0, row_offset, c1, i3, c2, j, i4])
        s0.reorder(consumer, [n0, m0, n1, m1, n2, m2, n, m3])
        s0.compute_at(sparse_dense_block, consumer, m2)

    print(s0)
    ret.append([s0.state_object, stage_id - 2])

    return ret


######################################################################
# Next, we set parameters for the auto-scheduler with the custom sketch plugged in.
#
# * :code:`num_measure_trials` is the number of measurement trials we can use during the search.
#   We only make 10 trials in this tutorial for a fast demonstration. In practice, 1000 is a
#   good value for the search to converge. You can do more trials according to your time budget.
# * In addition, we use :code:`RecordToFile` to dump measurement records into a file
#   `sparse_dense.json`.
#   The measurement records can be used to query the history best, resume the search,
#   and do more analyses later.
# * see :any:`auto_scheduler.TuningOptions` for more parameters
# * Here, we need to create a :code:`auto_scheduler.SketchPolicy` object, and add the custom sketch
#   rule as a `init_search_callbacks`.

log_file = "spmm_mali_hw%d_ci%d_co%d_d%s_bsi%d_bso%d_%s_%s_%s.json" % (ARGS.hw, ARGS.ci, ARGS.co, str(ARGS.density), ARGS.bso, ARGS.bsi, ARGS.data_layout, ARGS.weight_layout, ARGS.output_layout)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=1000,
    builder=auto_scheduler.LocalBuilder(build_func="default"),
    runner=auto_scheduler.RPCRunner(
        device_key,
        host=rpc_host,
        port=rpc_port,
        timeout=30,
        repeat=1,
        min_repeat_ms=200,
        enable_cpu_cache_flush=True,
    ),
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
    num_measures_per_round=32,
)

search_policy = auto_scheduler.SketchPolicy(
    task,
    program_cost_model=auto_scheduler.XGBModel(),
    init_search_callbacks=[
        auto_scheduler.PreloadCustomSketchRule(meet_condition_func, apply_func, "SparseDense")
    ],
)

######################################################################
# Run the search
# ^^^^^^^^^^^^^^
# Now we get all inputs ready.
# We can kick off the search and let the auto-scheduler do its magic.
# After some measurement trials, we can load the best schedule from the log
# file and apply it.

# Run auto-tuning (search)
# Notice: We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.
task.tune(tune_option, search_policy)

# Apply the best schedule
sch, args = task.apply_best(log_file)

######################################################################
# We can lower the schedule to see the IR after auto-scheduling.
# The auto-scheduler correctly performs optimizations including multi-level tiling,
# layout transformation, parallelization, vectorization, unrolling, and operator fusion.

print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))

######################################################################
# Check correctness and evaluate performance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We build the binary and check its correctness and performance.

func = tvm.build(sch, args, target)

tmp = utils.tempdir()
filename = 'net.tar'
func.export_library(tmp.relpath(filename))

print("Upload...")
remote = auto_scheduler.utils.request_remote(device_key, rpc_host, rpc_port, timeout=10000)
remote.upload(tmp.relpath(filename))
func = remote.load_module(filename)


#dev = remote.cpu()
dev = remote.cl()

X_tvm = tvm.nd.array(X_np, device=dev)
W_data_tvm = tvm.nd.array(W_sp_np_data, device=dev)
W_indices_tvm = tvm.nd.array(W_sp_np_indices, device=dev)
W_indptr_tvm = tvm.nd.array(W_sp_np_indptr, device=dev)
Y_tvm = tvm.nd.empty(Y_np.shape, device=dev)

func(X_tvm, W_data_tvm, W_indices_tvm, W_indptr_tvm, Y_tvm)

# Check results
#tvm.testing.assert_allclose(Y_np, Y_tvm.asnumpy(), atol=1e-4, rtol=1e-4)
np.testing.assert_allclose(Y_np, Y_tvm.asnumpy(), atol=1e-4, rtol=1e-4)

# Evaluate execution time.
evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
print(
    "Execution time of this operator: %.3f ms"
    % (
        np.median(evaluator(X_tvm, W_data_tvm, W_indices_tvm, W_indptr_tvm, Y_tvm).results)
        * 1000
    )
)

######################################################################
# .. note:: Tuning result example
#
#   .. code-block:: c
#
#    ----------------------------------------------------------------------
#    Lowered TIR:
#    primfn(placeholder_5: handle, placeholder_6: handle, placeholder_7: handle, placeholder_8: handle, placeholder_9: handle, compute_1: handle) -> ()
#      attr = {"global_symbol": "main", "tir.noalias": True}
#      buffers = {placeholder_2: Buffer(placeholder_10: Pointer(float32), float32, [9831, 16, 1], []),
#                 placeholder_4: Buffer(placeholder_11: Pointer(int32), int32, [33], []),
#                 placeholder_3: Buffer(placeholder_12: Pointer(float32), float32, [512, 512], []),
#                 compute: Buffer(compute_2: Pointer(float32), float32, [512, 512], []),
#                 placeholder_1: Buffer(placeholder_13: Pointer(float32), float32, [512, 512], []),
#                 placeholder: Buffer(placeholder_14: Pointer(int32), int32, [9831], [])}
#      buffer_map = {placeholder_7: placeholder, placeholder_9: placeholder_1, placeholder_6: placeholder_2, compute_1: compute, placeholder_5: placeholder_3, placeholder_8: placeholder_4} {
#      for (i0.outer.i1.outer.fused: int32, 0, 1024) "parallel" {
#        attr [compute_3: Pointer(float32)] "storage_scope" = "global";
#        allocate(compute_3, float32, [256]) {
#          for (nb_j.inner: int32, 0, 2) {
#            for (i.inner.init: int32, 0, 8) {
#              for (j.init: int32, 0, 16) {
#                compute_3[(((i.inner.init*32) + (nb_j.inner*16)) + j.init)] = 0f32
#              }
#            }
#            for (elem_idx: int32, 0, ((int32*)placeholder_11[(((floormod(i0.outer.i1.outer.fused, 16)*2) + nb_j.inner) + 1)] - (int32*)placeholder_11[((floormod(i0.outer.i1.outer.fused, 16)*2) + nb_j.inner)])) {
#              for (i.inner: int32, 0, 8) {
#                for (j: int32, 0, 16) {
#                  compute_3[(((i.inner*32) + (nb_j.inner*16)) + j)] = ((float32*)compute_3[(((i.inner*32) + (nb_j.inner*16)) + j)] + ((float32*)placeholder_10[((((int32*)placeholder_11[((floormod(i0.outer.i1.outer.fused, 16)*2) + nb_j.inner)]*16) + (elem_idx*16)) + j)]*max((float32*)placeholder_12[(((floordiv(i0.outer.i1.outer.fused, 16)*4096) + (i.inner*512)) + (int32*)placeholder_14[((int32*)placeholder_11[((floormod(i0.outer.i1.outer.fused, 16)*2) + nb_j.inner)] + elem_idx)])], 0f32)))
#                }
#              }
#            }
#          }
#          for (i0.inner: int32, 0, 8) {
#            compute_2[ramp((((floordiv(i0.outer.i1.outer.fused, 16)*4096) + (i0.inner*512)) + (floormod(i0.outer.i1.outer.fused, 16)*32)), 1, 32)] = max(((float32x32*)compute_3[ramp((i0.inner*32), 1, 32)] + (float32x32*)placeholder_13[ramp((((floordiv(i0.outer.i1.outer.fused, 16)*4096) + (i0.inner*512)) + (floormod(i0.outer.i1.outer.fused, 16)*32)), 1, 32)]), broadcast(0f32, 32))
#          }
#        }
#      }
#    }
