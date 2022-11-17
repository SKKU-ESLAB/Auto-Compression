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
Optimizing Operators with Auto-scheduling
=========================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, \
            `Chengfan Jia <https://github.com/jcf94/>`_

In this tutorial, we will show how TVM's Auto Scheduling feature can find
optimal schedules without the need for writing a custom template.

Different from the template-based :doc:`AutoTVM <autotvm_matmul_x86>` which relies on
manual templates to define the search space, the auto-scheduler does not
require any templates.  Users only need to write the computation declaration
without any schedule commands or templates.  The auto-scheduler can
automatically generate a large search space and find a good schedule in the
space.

We use matrix multiplication as an example in this tutorial.

.. note::
  Note that this tutorial will not run on Windows or recent versions of macOS. To
  get it to run, you will need to wrap the body of this tutorial in a :code:`if
  __name__ == "__main__":` block.
"""

import os
import logging
import sys

import numpy as np
import tvm
from tvm import te, auto_scheduler, autotvm
from tvm import topi
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.utils import get_const_int
from tvm.contrib.utils import tempdir

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
K_val = 128

@autotvm.template("ant/topk_conv2d")
def topk_conv2d(N, I, H, W, O, dtype="float32"):
    K = te.var("K", dtype="int32")
    data = te.placeholder((N, I, H, W), name="data", dtype=dtype)
    kernel = te.placeholder((O, I, 1, 1), name="kernel", dtype=dtype)
    c_list = te.placeholder((K,), name="c_list", dtype="int64")

    '''
    i = te.reduce_axis((0, I), name="i")
    conv = te.compute((N, K, H, W),
                     lambda n, k, h, w: te.sum(data[n, i, h, w] * kernel[c_list[k], i, 0, 0], axis=[i,]),
                     name="conv")

    s = te.create_schedule([out.op])
    '''
    cfg = autotvm.get_config()
    cfg.add_flop(2*N*I*H*W*K_val)

    CO = K_val if isinstance(K, (tvm.tir.Var, tvm.tir.Any)) else K
    #CO = K
    #CO = O
    CI = I
    OH = H
    OW = W
    IH = H
    IW = W
    KH = 1
    KW = 1

    #n, co, oh, ow = cfg.axis(N), cfg.axis(CO), cfg.axis(OH), cfg.axis(OW)
    n, oh, ow = cfg.axis(N), cfg.axis(OH), cfg.axis(OW)
    ci, kh, kw = cfg.reduce_axis(I), cfg.reduce_axis(KH), cfg.reduce_axis(KW)

    #co, vc = cfg.define_split("tile_co", co, num_outputs=2)
    oh, vh = cfg.define_split("tile_oh", oh, num_outputs=2)
    ow, vw = cfg.define_split("tile_ow", ow, num_outputs=2)

    cfg.define_knob("tile_co", [2, 4, 8, 16])
    #cfg.define_knob("reorder_0", [0, 1])

    '''
    cfg.define_reorder(
        "reorder_0",
        [n, co, oh, ow, ci, kh, kw, vh, vw, vc],
        policy="candidate",
        candidate=[
            [n, co, oh, ow, ci, kh, kw, vh, vw, vc],
            [n, co, oh, ow, ci, kh, kw, vc, vh, vw],
        ],
    )
    '''

    #cfg.define_annotate("ann_reduce", [kh, kw], policy="try_unroll")
    #fg.define_annotate("ann_spatial", [vh, vw, vc], policy="try_unroll_vec")
    cfg.define_annotate("ann_spatial", [vh, vw], policy="try_unroll_vec")

    #VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod

    VC = idxdiv(CO, cfg["tile_co"].val)

    #kvshape = (CO // VC, CI, KH, KW, VC)
    kvshape = (CO, CI, KH, KW)
    #ovshape = (N, CO // VC, OH // VH, OW // VW, VH, VW, VC)
    #ovshape = (N, cfg["tile_co"].val, OH // VH, OW // VW, VH, VW, VC)
    ovshape = (N, CO, OH // VH, OW // VW, VH, VW)
    oshape = (N, CO, OH, OW)

    dvshape = (N, OH // VH, OW // VW, CI, VH, VW)
    data_vec = te.compute(
        dvshape,
        lambda n, h, w, ci, vh, vw: data[n][ci][h*VH+vh][w*VW+vw],
        name="data_vec",
    )

    '''
    if autotvm.GLOBAL_SCOPE.in_tuning and False:
        kernel_vec = tvm.te.placeholder(kvshape, kernel.dtype, name="kernel_autotvm")
    else:
        kernel_vec = te.compute(
            kvshape,
            lambda co, ci, kh, kw, vc: kernel[co*VC+vc][ci][kh][kw],
            name="kernel_vec",
        )
    '''
    kernel_vec = kernel

    ci = te.reduce_axis((0, CI), name="ci")
    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")


    conv = te.compute(
        ovshape,
        #lambda n, co, h, w, vh, vw, vc: te.sum(
        lambda n, co, h, w, vh, vw: te.sum(
            data_vec[n, h, w, ci, vh+kh, vw+kw].astype("float32")
            #* kernel_vec[idxdiv(c_list[co*VC+vc], VC), ci, kh, kw, idxmod(c_list[co*VC+vc], VC)].astype("float32"),
            * kernel_vec[c_list[co], ci, kh, kw].astype("float32"),
            axis=[ci, kh, kw],
        ),
        name="conv",
    )


    output = te.compute(
        oshape,
        lambda n, co, h, w: conv[
            n,
            #idxdiv(co, VC),
            co,
            idxdiv(h, VH),
            idxdiv(w, VW),
            idxmod(h, VH),
            idxmod(w, VW),
            #idxmod(co, VC),
        ],
        name="output_unpack",
        tag="spatial_conv2d_output",
    )

    s = te.create_schedule(output.op)

    #n, co, oh, ow, vh, vw, vc = s[conv].op.axis
    n, co, oh, ow, vh, vw = s[conv].op.axis
    ci, kh, kw = s[conv].op.reduce_axis

    # schedule conv
    #cfg["reorder_0"].apply(s, conv, [n, co, oh, ow, ci, kh, kw, vh, vw, vc])
    #if cfg["reorder_0"].val == 0:
    #    s[conv].reorder(n, co, oh, ow, ci, kh, kw, vh, vw, vc)
    #else:
    #    s[conv].reorder(n, co, oh, ow, ci, kh, kw, vc, vh, vw)
    s[conv].reorder(n, co, oh, ow, ci, kh, kw, vh, vw)
    '''
    cfg["ann_reduce"].apply(
        s,
        conv,
        [kh, kw],
        axis_lens=[get_const_int(kh.dom.extent), get_const_int(kw.dom.extent)],
        max_unroll=None,
        cfg=cfg,
    )
    '''
    cfg["ann_spatial"].apply(
        s, conv,
        #[vh, vw, vc],
        [vh, vw],
        #axis_lens=[cfg["tile_oh"].size[-1], cfg["tile_ow"].size[-1], cfg["tile_co"].size[-1]],
        axis_lens=[cfg["tile_oh"].size[-1], cfg["tile_ow"].size[-1]],
        max_unroll=None,
        cfg=cfg,
    )

    # schedule fusion
    n, co, h, w = s[output].op.axis
    #co, vc = cfg["tile_co"].apply(s, output, co)
    #co, vc = s[output].split(co, nparts=cfg["tile_co"].val)
    oh, vh = cfg["tile_oh"].apply(s, output, h)
    ow, vw = cfg["tile_ow"].apply(s, output, w)
    #s[output].reorder(n, co, oh, ow, vh, vw, vc)
    s[output].reorder(n, co, oh, ow, vh, vw)
    s[conv].compute_at(s[output], ow)

    # mark parallel
    co, vc = s[output].split(co, nparts=cfg["tile_co"].val)
    s[output].parallel(co)
    _, h, _, _, _, _ = s[data_vec].op.axis
    s[data_vec].parallel(h)

    if kernel_vec.op.name == "kernel_vec":
        if not autotvm.GLOBAL_SCOPE.in_tuning:
            co, _, _, _, _ = s[kernel_vec].op.axis
            s[kernel_vec].parallel(co)

    #print(tvm.lower(s, [data, kernel, c_list, output], simple_mode=True))

    return s, [data, kernel, c_list, output]

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

logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))
N = 1
I = 128
H = 56
W = 56
O = 128
K = K_val
dtype = "float32"

score = np.random.randn(O)

#data = tvm.nd.array(np.random.randn(N, I, H, W).astype("float32"))
#weight = tvm.nd.array(np.zeros((O, I, 1, 1)).astype("float32"))
#c_list = tvm.nd.array(np.argsort(score[::-1][:K]))
data = np.random.randn(N, I, H, W).astype("float32")
weight = np.zeros((O, I, 1, 1)).astype("float32")
c_list = np.argsort(score[::-1][:K])
out = np.zeros((N, K, H, W)).astype("float32")
#out = np.zeros((N, O, H, W)).astype("float32")

print(tvm.nd.array(c_list))
#task = tvm.auto_scheduler.SearchTask(func=top_k_conv2d, args=(N, I, H, W, O, K, "float32"), target=target, task_inputs=task_inputs, task_inputs_overwrite=True)
#task = tvm.auto_scheduler.SearchTask(func=top_k_conv2d, args=(N, I, H, W, O, K, "float32"), target=target)
task = autotvm.task.create(
    "ant/topk_conv2d", args=(N, I, H, W, O, "float32"), target=target
)
tasks = [task,]
log_file = "top_k_conv2d.json"

tuning_option = {
    "log_filename": log_file,
    "tuner": "xgb",
    "n_trial": 3200,
    "early_stopping": 800,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"),
        runner=autotvm.RPCRunner(
            device_key,
            host=rpc_host,
            port=rpc_port,
            number=5,
            timeout=10,
        ),
    ),
}

def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        measure_option["runner"].ref_input = [data, weight, c_list, out]
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "xgb_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="itervar")
        elif tuner == "xgb_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="curve")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # process tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)

tune_tasks(tasks, **tuning_option)
#exit()

dispatch_context = autotvm.apply_history_best(log_file)
best_config = dispatch_context.query(task.target, task.workload)
print("\nBest config:")
print(best_config)

with autotvm.apply_history_best(log_file):
    with target:
        s, arg_bufs = topk_conv2d(N, I, H, W, O, "float32")
        print(tvm.lower(s, arg_bufs, simple_mode=True))
        func = tvm.build(s, arg_bufs)

remote = autotvm.measure.request_remote(device_key, rpc_host, rpc_port, timeout=10000)
dev = remote.device(str(target), 0)

temp = tempdir()
path_tar = temp.relpath("dev_lib.tar")
func.export_library(path_tar)
remote.upload(path_tar)
rlib = remote.load_module("dev_lib.tar")


data_tvm = tvm.nd.array(data, device=dev)
kernel_tvm = tvm.nd.array(weight, device=dev)
c_list_tvm = tvm.nd.array(c_list[:64], device=dev)
out_tvm = tvm.nd.array(out[:, :64], device=dev)

evaluator = rlib.time_evaluator(func.entry_name, dev, number=20)
print(evaluator(data_tvm, kernel_tvm, c_list_tvm, out_tvm).mean)
exit()

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
