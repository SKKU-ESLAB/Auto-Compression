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

######################################################################
# Define the computation
# ^^^^^^^^^^^^^^^^^^^^^^
# To begin with, let us define the computation of a sparse matmul with several relu and bias add.
# The function should return the list of input/output tensors.
# From these tensors, the auto-scheduler can get the whole computational graph.


@auto_scheduler.register_workload
def sparse_dense(M, N, K, w_data_shape, w_indices_shape, w_indptr_shape, dtype):
    X = te.placeholder(shape=(M, K), dtype=dtype)
    W_data = te.placeholder(shape=w_data_shape, dtype=dtype)
    W_indices = te.placeholder(shape=w_indices_shape, dtype="int32")
    W_indptr = te.placeholder(shape=w_indptr_shape, dtype="int32")

    out = topi.nn.sparse_dense(X, W_data, W_indices, W_indptr)

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
M = 1024
K = 1024
N = 1024
BS_R = 1
BS_C = 4
density = 0.1

# Generate the test data with numpy
X_np = np.random.randn(M, K).astype("float32")
W_sp_np = random_bsr_matrix(N, K, BS_R, BS_C, density=density, dtype="float32")
W_np = W_sp_np.todense()
Y_np = X_np @ W_np.T  # Process the matrix multiplication

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
prefix = "sparse_dense_bsr_%d_%d_%d_%d_%.2f_" % (N, K, BS_R, BS_C, density)
task = tvm.auto_scheduler.SearchTask(
    func=sparse_dense,
    args=(M, N, K, W_sp_np.data.shape, W_sp_np.indices.shape, W_sp_np.indptr.shape, "float32"),
    target=target,
    task_inputs={
        prefix + "W_data": runtime.ndarray.array(W_sp_np.data),
        prefix + "W_indices": runtime.ndarray.array(W_sp_np.indices),
        prefix + "W_indptr": runtime.ndarray.array(W_sp_np.indptr),
    },
    task_inputs_save_to_file=True,
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
        "sparse_dense_sp_rhs_bsrmm",
        "sparse_dense_sp_rhs_bsrmm_block",
    ]:
        return auto_scheduler.PreloadCustomSketchRule.APPLY_AND_SKIP_REST
    else:
        return auto_scheduler.PreloadCustomSketchRule.PASS


def apply_func(search_policy, state, stage_id):
    ret = []
    s0 = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
    if s0.stages[stage_id].op.tag == "sparse_dense_sp_rhs_bsrmm_block":
        return [s0.state_object, stage_id - 1]

    sparse_dense = s0.stages[stage_id].op
    sparse_dense_block = s0.stages[stage_id - 1].op
    assert sparse_dense.tag == "sparse_dense_sp_rhs_bsrmm"
    assert sparse_dense_block.tag == "sparse_dense_sp_rhs_bsrmm_block"

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

    i, nb_j, j, row_offset, c = s0[sparse_dense_block].iters
    m, n = s0[consumer].iters
    nb_n, n = s0.split(consumer, n, [j.range.extent])
    i0, i1, i2, i3, i4 = s0.split(sparse_dense_block, i, [None, None, None, None])
    #i0, i1, i2, i3, i4 = s0.split(sparse_dense_block, i, [2, 4, 2, 4])
    m0, m1, m2, m3 = s0.follow_split(consumer, m, len(s0.transform_steps) - 1, 3)
    j0, j1, j2, = s0.split(sparse_dense_block, nb_j, [None, None])
    #j0, j1, j2, j3 = s0.split(sparse_dense_block, nb_j, [4, 2, 16])
    n0, n1, n2, = s0.follow_split(consumer, nb_n, len(s0.transform_steps) - 1, 2)

    c0, c1, c2 = s0.split(sparse_dense_block, c, [None, None])

    ro0, ro1 = s0.split(sparse_dense_block, row_offset, [None])

    #s0.reorder(sparse_dense_block, [i0, j0, i1, j1, i2, j2, c0, row_offset, c1, i3, c2, i4, j])
    s0.reorder(sparse_dense_block, [i0, j0, i1, j1, i2, j2, c0, ro0, ro1, c1, i3, c2, i4, j])
    s0.reorder(consumer, [m0, n0, m1, n1, m2, n2, m3, n])
    s0.compute_at(sparse_dense_block, consumer, n2)

    s0.rfactor(sparse_dense_block, ro0, 7)

    #s0.bind(consumer, m0, 'blockIdx.x')
    #s0.bind(consumer, n0, 'blockIdx.y')
    #s0.bind(consumer, m1, 'threadIdx.x')
    #s0.bind(consumer, n1, 'threadIdx.y')
    print(s0)
    exit()

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

log_file = "spmm_mali.json"
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

print(task.print_best(log_file, print_mode='schedule'))
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
print(func.imported_modules[0].get_source())

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
W_data_tvm = tvm.nd.array(W_sp_np.data, device=dev)
W_indices_tvm = tvm.nd.array(W_sp_np.indices, device=dev)
W_indptr_tvm = tvm.nd.array(W_sp_np.indptr, device=dev)
Y_tvm = tvm.nd.empty(Y_np.shape, device=dev)

func(X_tvm, W_data_tvm, W_indices_tvm, W_indptr_tvm, Y_tvm)

# Check results
#tvm.testing.assert_allclose(Y_np, Y_tvm.asnumpy(), atol=1e-4, rtol=1e-4)

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
