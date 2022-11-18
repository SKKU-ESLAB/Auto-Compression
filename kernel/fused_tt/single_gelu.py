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

import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, relay
from tvm import topi

################################################################################
# Defining the Matrix Multiplication
# ----------------------------------
# To start, we define a matrix multiplication with a bias addition.  Note that
# this uses standard operations available in TVMs Tensor Expression language.
# The major difference is the use of the `auto_sceduler` decorator at the top
# of the function definition.  The function should return a list of
# input/output tensors.  From these tensors, the auto-scheduler can get the
# whole computational graph.

os.environ["TVM_NUM_THREADS"] = str(4)

@auto_scheduler.register_workload
def tt_linear(B, I, O, R, dtype):
    #_I = I[0] * I[1] * I[2] * I[3]
    _I = I[0] * I[1] * I[2]
    #_O = O[0] * O[1] * O[2] * O[3]
    _O = O[0] * O[1] * O[2]
    A = te.placeholder((B, _I), name="input", dtype=dtype)

    out = te.compute((B, _I), lambda b, i: A[b, i] * (0.5 + tvm.tir.erf(A[b, i] * 0.5**0.5) * 0.5))
    return [A, out]


################################################################################
# Create the search task
# ----------------------
# With the function defined, we can now create the task for the auto_scheduler
# to search against. We specify the particular parameters for this matrix
# multiplication, in this case a multiplication of to square matricies of size
# 1024x1024. We then create a search task with N=L=M=1024 and dtype="float32"
#
# .. note:: Improve performance with custom targets
#   In order for TVM to take full advantage of specific hardware platforms,
#   you will want to manuall specify your CPU capabilities. For example:
#   - replace "llvm" below with "llvm -mcpu=core-avx2" to enable AVX2
#   - replace "llvm" below with "llvm -mcpu=skylake-avx512" to enable AVX-512

#target = tvm.target.Target("llvm")
target = tvm.target.Target("llvm -mtriple=arm-linux-gnueabihf -mattr=+neon")
device_key = "xu4"
rpc_host = "10.201.135.165"
rpc_port = 8109
#N = L = M = 1024
B = 196
I = [12, 16, 16]
O = [12, 16, 16]
R = [1, 32, 32, 1]
task = tvm.auto_scheduler.SearchTask(func=tt_linear, args=(B, I, O, R, "float32"), target=target)

# Inspect the computational graph
print("Computational DAG:")
print(task.compute_dag)

################################################################################
# Set Parameters for Auto-Scheduler
# ---------------------------------
# Next, we set parameters for the auto-scheduler.
#
# * :code:`num_measure_trials` is the number of measurement trials we can use
#   during the search.  We only make 10 trials in this tutorial for a fast
#   demonstration. In practice, 1000 is a good value for the search to converge.
#   You can do more trials according to your time budget.
# * In addition, we use :code:`RecordToFile` to log measurement records into a
#   file `matmul.json`.  The measurement records can be used to query the history
#   best, resume the search, and do more analyses later.
# * see :any:`auto_scheduler.TuningOptions` for more parameters

log_file = "gelu.json"
#tune_option = auto_scheduler.TuningOptions(
#    num_measure_trials=1000,
#    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
#    verbose=2,
#)
tuner = auto_scheduler.TaskScheduler([task,], load_log_file=log_file)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=2000,  # change this to 20000 to achieve the best performance
    builder=auto_scheduler.LocalBuilder(build_func="default"),
    runner=auto_scheduler.RPCRunner(
        device_key,
        host=rpc_host,
        port=rpc_port,
        timeout=300,
        n_parallel=64,
        repeat=1,
        min_repeat_ms=200,
        enable_cpu_cache_flush=True,
    ),
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
)
tuner.tune(tune_option)

################################################################################
# Run the search
# --------------
# Now we get all inputs ready. Pretty simple, isn't it?  We can kick off the
# search and let the auto-scheduler do its magic.  After some measurement
# trials, we can load the best schedule from the log file and apply it.

# Run auto-tuning (search)
#task.tune(tune_option)
# Apply the best schedule
#sch, args = task.apply_best(log_file)

# Compile with the history best
print("Compile...")
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(
        opt_level=3, config={"relay.backend.use_auto_scheduler": True}
    ):
        lib = relay.build(mod, target=target, params=params)

# Export library
tmp = tempdir()
filename = "net.tar"
lib.export_library(tmp.relpath(filename))

# Upload module to device
print("Upload...")
remote = auto_scheduler.utils.request_remote(device_key, rpc_host, rpc_port, timeout=10000)
remote.upload(tmp.relpath(filename))
rlib = remote.load_module(filename)

# Create graph executor
dev = remote.cpu()
module = graph_executor.GraphModule(rlib["default"](dev))
data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
module.set_input("data", data_tvm)

# Evaluate
print("Evaluate inference time cost...")
print(module.benchmark(dev, repeat=3, min_repeat_ms=500))

################################################################################
# Inspecting the Optimized Schedule
# ---------------------------------
# We can lower the schedule to see the IR after auto-scheduling.  The
# auto-scheduler correctly performs optimizations including multi-level tiling,
# layout transformation, parallelization, vectorization, unrolling, and
# operator fusion.

print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))

exit()

################################################################################
# Check correctness and evaluate performance
# ------------------------------------------
# We build the binary and check its correctness and performance.

_I = I[0] * I[1] * I[2] * I[3]
_O = O[0] * O[1] * O[2] * O[3]

func = tvm.build(sch, args, target)
a_np = np.random.uniform(size=(B, _I)).astype(np.float32)
w1_np = np.random.uniform(size=(R[0], O[0], I[0], R[1])).astype(np.float32)
w2_np = np.random.uniform(size=(R[1], O[1], I[1], R[2])).astype(np.float32)
w3_np = np.random.uniform(size=(R[2], O[2], I[2], R[3])).astype(np.float32)
w4_np = np.random.uniform(size=(R[3], O[3], I[3], R[4])).astype(np.float32)
out_np = np.random.uniform(size=(B, _O)).astype(np.float32)
#out_np = a_np.dot(b_np) + c_np

dev = tvm.cpu()
a_tvm = tvm.nd.array(a_np, device=dev)
w1_tvm = tvm.nd.array(w1_np, device=dev)
w2_tvm = tvm.nd.array(w2_np, device=dev)
w3_tvm = tvm.nd.array(w3_np, device=dev)
w4_tvm = tvm.nd.array(w4_np, device=dev)
out_tvm = tvm.nd.empty(out_np.shape, device=dev)
func(a_tvm, w1_tvm, w2_tvm, w3_tvm, w4_tvm, out_tvm)

# Check results
#np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)

# Evaluate execution time.
evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
print(
    "Execution time of this operator: %.3f ms"
    % (np.median(evaluator(a_tvm, w1_tvm, w2_tvm, w3_tvm, w4_tvm, out_tvm).results) * 1000)
)


################################################################################
# Using the record file
# ---------------------
# During the search, all measurement records are logged into the record file
# "matmul.json". The measurement records can be used to re-apply search
# results, resume the search, and perform other analyses.
#
# Here is an example where we load the best schedule from a file, and print the
# equivalent python schedule API. This can be used for debugging and learning
# the behavior of the auto-scheduler.

print("Equivalent python schedule:")
print(task.print_best(log_file))

################################################################################
# A more complicated example is to resume the search.  In this case, we need to
# create the search policy and cost model by ourselves and resume the status of
# search policy and cost model with the log file.  In the example below we
# resume the status and do more 5 trials.


def resume_search(task, log_file):
    print("Resume search:")
    cost_model = auto_scheduler.XGBModel()
    cost_model.update_from_file(log_file)
    search_policy = auto_scheduler.SketchPolicy(
        task, cost_model, init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(log_file)]
    )
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=5, measure_callbacks=[auto_scheduler.RecordToFile(log_file)]
    )
    task.tune(tune_option, search_policy=search_policy)


#resume_search(task, log_file)

################################################################################
# Final Notes and Summary
# -----------------------
# In this tutorial, we have shown how to use the TVM Auto-Scheduler to
# automatically optimize a matrix multiplication, without the need to specify a
# search template.  It ends a series of examples that starts from the Tensor
# Expression (TE) language that demonstrates how TVM can optimize computational
# operations.
