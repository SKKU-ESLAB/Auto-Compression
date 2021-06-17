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

######################################################################
# Define the computation
# ^^^^^^^^^^^^^^^^^^^^^^
# To begin with, let us define the computation of a sparse matmul with several relu and bias add.
# The function should return the list of input/output tensors.
# From these tensors, the auto-scheduler can get the whole computational graph.
@auto_scheduler.register_workload
def conv2d(data_shape, w_shape, dtype, data_layout, weight_layout, output_layout):
    X = te.placeholder(shape=data_shape, dtype=dtype)
    W = te.placeholder(shape=w_shape, dtype=dtype)

    if data_layout == 'hwc':
        N, IH, IW, CI = data_shape
    else:
        N, CI, IH, IW = data_shape
    CO, CI, KH, KW = w_shape
    OH = IH
    OW = IW

    rc = te.reduce_axis((0, CI), name="rc")
    ry = te.reduce_axis((0, KH), name="ry")
    rx = te.reduce_axis((0, KW), name="rx")

    if data_layout == 'hwc':
        if output_layout == 'hwc':
            out = te.compute(
                    (N, OH, OW, CO),
                    lambda nn, yy, xx, ff: te.sum(
                        X[nn, yy+ry, xx+rx, rc] * W[ff, rc, ry, rx],
                        axis=[rc, ry, rx],
                    ),
                    tag="conv2d")
        else:
            out = te.compute(
                    (N, CO, OH, OW),
                    lambda nn, ff, yy, xx: te.sum(
                        X[nn, yy+ry, xx+rx, rc] * W[ff, rc, ry, rx],
                        axis=[rc, ry, rx],
                    ),
                    tag="conv2d")
    else:
        if output_layout == 'hwc':
            out = te.compute(
                    (N, OH, OW, CO),
                    lambda nn, yy, xx, ff: te.sum(
                        X[nn, rc, yy+ry, xx+rx] * W[ff, rc, ry, rx],
                        axis=[rc, ry, rx],
                    ),
                    tag="conv2d")
        else:
            out = te.compute(
                    (N, CO, OH, OW),
                    lambda nn, ff, yy, xx: te.sum(
                        X[nn, rc, yy+ry, xx+rx] * W[ff, rc, ry, rx],
                        axis=[rc, ry, rx],
                    ),
                    tag="conv2d")

    return [X, W, out]


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

# Default
# data: hwc
# weight: oi
# output: hwc

# Generate the test data with numpy
X_np = np.random.randn(1, ARGS.hw, ARGS.hw, ARGS.ci).astype("float32")
W_np = np.random.randn(ARGS.co, ARGS.ci, 1, 1).astype("float32")
Y_np = np.random.randn(1, ARGS.hw, ARGS.hw, ARGS.co).astype("float32")

if ARGS.data_layout == 'chw':
    X_np = X_np.transpose(0, 3, 1, 2)
if ARGS.weight_layout == 'io':
    W_np = W_np.transpose(1, 0, 2, 3)
if ARGS.output_layout == 'chw':
    Y_np = Y_np.transpose(0, 3, 1, 2)

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
task = tvm.auto_scheduler.SearchTask(
    func=conv2d,
    args=(X_np.shape, W_np.shape, "float32", ARGS.data_layout, ARGS.weight_layout, ARGS.output_layout),
    target=target,
    #task_inputs_save_to_file=True,
    #layout_rewrite_option=auto_scheduler.LayoutRewriteOption.NO_REWRITE,
)

# Inspect the computational graph
print("Computational DAG:")
print(task.compute_dag)

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

log_file = "conv2d_mali_hw%d_ci%d_co%d_%s_%s_%s.json" % (ARGS.hw, ARGS.ci, ARGS.co, ARGS.data_layout, ARGS.weight_layout, ARGS.output_layout)
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
    #init_search_callbacks=[
    #    auto_scheduler.PreloadCustomSketchRule(meet_condition_func, apply_func, "SparseDense")
    #],
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
W_tvm = tvm.nd.array(W_np, device=dev)
Y_tvm = tvm.nd.empty(Y_np.shape, device=dev)

func(X_tvm, W_tvm, Y_tvm)

# Check results
#tvm.testing.assert_allclose(Y_np, Y_tvm.asnumpy(), atol=1e-4, rtol=1e-4)
#np.testing.assert_allclose(Y_np, Y_tvm.asnumpy(), atol=1e-4, rtol=1e-4)

# Evaluate execution time.
evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
print(
    "Execution time of this operator: %.3f ms"
    % (
        np.median(evaluator(X_tvm, W_tvm, Y_tvm).results)
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
