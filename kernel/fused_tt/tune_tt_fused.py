import numpy as np
import os

import tvm
from tvm import relay, auto_scheduler
from tvm.relay import data_dep_optimization as ddo
import tvm.relay.testing
from tvm.contrib import graph_executor
from tvm.contrib.utils import tempdir

import argparse

import ml_collections
from tt_mixer import TTMixer
import torch

from tvm.relay.dataflow_pattern import *
from tvm.relay.testing import run_opt_pass
from tvm.relay.op.op import register_injective_schedule

def get_mixer_b16_tt_config(args):
    """Returns TTMixer-B/16 configuration"""
    config = ml_collections.ConfigDict()
    config = ml_collections.ConfigDict()
    config.name = 'Mixer-B_16'
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_dim = 768
    config.hidden_shape = args.hidden_tt_shape
    config.num_blocks = 12
    config.tokens_mlp_dim = 384
    config.channels_mlp_dim = 3072
    config.channels_mlp_shape = args.channels_tt_shape
    config.tt_ranks = args.tt_ranks
    return config

def set_configs(args):

    args.save_path = "saved_models/B_16_cifar_10.pt"
    args.img_size = 224
    args.num_classes = 10
    args.tt_ranks = [int(i) for i in args.tt_ranks.split(',')]
    args.hidden_tt_shape = [int(i) for i in args.hidden_tt_shape.split(',')]
    args.channels_tt_shape = [int(i) for i in args.channels_tt_shape.split(',')]
    args.target_layer = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    return args

parser = argparse.ArgumentParser()
# TT-format Configuration

# ranks config [64, 64], [32, 32], [16, 16], [8, 8]
parser.add_argument("--tt-ranks", default="16, 16",
                    type=str,
                    help="Ranks for TT-Format")
# 768 factorize (e.g. 768 = 8 x 8 x 12)
parser.add_argument("--hidden-tt-shape", default="8, 8, 12",
                    type=str,
                    help="Factorized hidden dimension for TT-format")
# 3072 factorize (e.g. 3072 = 12 x 12 x 16)
parser.add_argument("--channels-tt-shape", default="12, 16, 16",
                    type=str,
                    help="Factorized channel dimension for TT-format")
args = parser.parse_args()

args = set_configs(args)

import logging
#logging.getLogger('auto_scheduler').setLevel(logging.DEBUG)

#def compute_tt_linear(data, w1, w2, w3):
@relay.op.register_compute("tt_linear")
def compute_tt_linear(attrs, inputs):
    out = inputs[0] + 1
    return [out]

relay.op.op.register_injective_schedule("tt_linear")
relay.op.op.register_pattern("tt_linear", relay.op.op.OpPattern.INJECTIVE)

def get_network(name, batch_size, layout="NHWC", dtype="float32", use_sparse=True):
    """Get the symbol definition and random weight of a network"""

    config = get_mixer_b16_tt_config(args)
    model = TTMixer(config,
                    args.img_size,
                    num_classes=args.num_classes,
                    patch_size=16,
                    zero_head=False,
                    target_layer=args.target_layer)

    input_shape = (1, 3, 224, 224)
    output_shape = (1, 10)
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()

    input_name = "input0"
    shape_list = [(input_name, input_shape)]

    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    def make_tt_linear_pattern():
        w1 = is_op("transpose")(wildcard())

        x1 = is_op("nn.dense")(wildcard(), w1)
        x1 = is_op("reshape")(x1)
        #x1 = is_op("reshape")(x1)
        x1 = is_op("transpose")(x1)
        x1 = is_op("reshape")(x1)
        #x1 = is_op("reshape")(x1)

        w2 = is_op("transpose")(wildcard())

        x2 = is_op("nn.dense")(x1, w2)
        x2 = is_op("reshape")(x2)
        #x2 = is_op("reshape")(x2)
        x2 = is_op("transpose")(x2)
        x2 = is_op("reshape")(x2)
        #x2 = is_op("reshape")(x2)

        w3 = is_op("transpose")(wildcard())

        x3 = is_op("nn.dense")(x2, w3)
        return x3

    pattern_table = [("tt_linear", make_tt_linear_pattern())]


    class TTLinearCallback(DFPatternCallback):
        def __init__(self, require_type=False):
            super().__init__(require_type)
            self.x1 = wildcard()
            self.w1 = wildcard()
            self.w2 = wildcard()
            self.w3 = wildcard()
            self.b = wildcard()
            self.pattern = is_op("reshape")(self.x1)
            self.pattern = is_op("transpose")(self.pattern)
            self.pattern = is_op("reshape")(self.pattern)
            self.pattern = is_op("nn.dense")(self.pattern, is_op("transpose")(self.w1))
            self.pattern = is_op("reshape")(self.pattern)
            self.pattern = is_op("transpose")(self.pattern)
            self.pattern = is_op("reshape")(self.pattern)
            self.pattern = is_op("nn.dense")(self.pattern, is_op("transpose")(self.w2))
            self.pattern = is_op("reshape")(self.pattern)
            self.pattern = is_op("transpose")(self.pattern)
            self.pattern = is_op("reshape")(self.pattern)
            self.pattern = is_op("nn.dense")(self.pattern, is_op("transpose")(self.w3))
            self.pattern = is_op("reshape")(self.pattern)
            self.pattern = is_op("add")(self.b, self.pattern)

            self.x2 = self.pattern

            self.pattern = is_op("multiply")(self.pattern, wildcard())
            self.pattern = is_op("erf")(self.pattern)
            self.pattern = is_op("multiply")(self.pattern, wildcard())
            self.pattern = is_op("add")(wildcard(), self.pattern)
            self.pattern = is_op("multiply")(self.x2, self.pattern)
            self.w4 = wildcard()
            self.w5 = wildcard()
            self.w6 = wildcard()
            self.b2 = wildcard()
            self.pattern = is_op("reshape")(self.pattern)
            self.pattern = is_op("transpose")(self.pattern)
            self.pattern = is_op("reshape")(self.pattern)
            self.pattern = is_op("nn.dense")(self.pattern, is_op("transpose")(self.w4))
            self.pattern = is_op("reshape")(self.pattern)
            self.pattern = is_op("transpose")(self.pattern)
            self.pattern = is_op("reshape")(self.pattern)
            self.pattern = is_op("nn.dense")(self.pattern, is_op("transpose")(self.w5))
            self.pattern = is_op("reshape")(self.pattern)
            self.pattern = is_op("transpose")(self.pattern)
            self.pattern = is_op("reshape")(self.pattern)
            self.pattern = is_op("nn.dense")(self.pattern, is_op("transpose")(self.w6))
            self.pattern = is_op("reshape")(self.pattern)
            self.pattern = is_op("add")(self.b2, self.pattern)

        def callback(self, pre, post, node_map):
            print("haha")
            x1 = node_map[self.x1][0]
            w1 = node_map[self.w1][0]
            w2 = node_map[self.w2][0]
            w3 = node_map[self.w3][0]
            return x1


    expr = run_opt_pass(mod["main"], relay.transform.SimplifyExpr())
    mod = tvm.IRModule.from_expr(expr)

    #expr = run_opt_pass(mod["main"], relay.transform.MergeComposite(pattern_table), import_prelude=False)
    #mod = tvm.IRModule.from_expr(expr)

    #expr = rewrite(TTLinearCallback(), mod["main"])
    #mod = tvm.IRModule.from_expr(expr)

    print(mod)

    return mod, params, input_shape, output_shape

#### DEVICE CONFIG ####

# Replace "aarch64-linux-gnu" with the correct target of your board.
# This target is used for cross compilation. You can query it by :code:`gcc -v` on your device.
# FIXME(tmoreau89, merrymercy): We leave '-device=arm_cpu' out of the target string
#                               because we're sharing x86 op strategy.
target = tvm.target.Target("llvm -mtriple=arm-linux-gnueabihf -mattr=+neon")

# Also replace this with the device key, rpc host and rpc port in your tracker
device_key = "xu4"
rpc_host = "10.201.135.165"
rpc_port = 8109

# Set this to True if you use ndk tools for cross compiling
# And also set the environment variable below to point to the cross compiler
use_ndk = False
# os.environ["TVM_NDK_CC"] = "/usr/bin/aarch64-linux-gnu-g++"

#### TUNING OPTION ####
network = "mobilenet"
use_sparse = True
batch_size = 1
#layout = args.layout
dtype = "float32"
log_file = "R%d.json" % (args.tt_ranks[0])
#log_file = "test.json"

#################################################################
# Extract Search Tasks
# --------------------
# Next, we extract the search tasks and their weights from a network.
# The weight of a task is the number of appearances of the task's subgraph
# in the whole network.
# By using the weight, we can approximate the end-to-end latency of the network
# as :code:`sum(latency[t] * weight[t])`, where :code:`latency[t]` is the
# latency of a task and :code:`weight[t]` is the weight of the task.
# The task scheduler will just optimize this objective.

# Extract tasks from the network
print("Get model...")
mod, params, input_shape, output_shape = get_network(
    network, batch_size, "NCHW", dtype=dtype, use_sparse=use_sparse
)
print("Extract tasks...")
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)


#################################################################
# Tuning and Evaluation
# ---------------------
# Now, we set some options for tuning and launch the search tasks
#
# * :code:`num_measure_trials` is the number of measurement trials we can use during the tuning.
#   You can set it to a small number (e.g., 200) for a fast demonstrative run.
#   In practice, we recommend setting it around :code:`800 * len(tasks)`,
#   which is typically enough for the search to converge.
#   For example, there are 29 tasks in resnet-50, so we can set it as 20000.
#   You can adjust this parameter according to your time budget.
# * In addition, we use :code:`RecordToFile` to dump measurement records into a log file,
#   The measurement records can be used to query the history best, resume the search,
#   and do more analyses later.
# * see :any:`auto_scheduler.TuningOptions`,
#   :any:`auto_scheduler.LocalRunner` for more parameters.
#
# After auto-tuning, we can compile the network with the best schedules we found.
# All measurement records are dumped into the log file during auto-tuning,
# so we can read the log file and load the best schedules.


def tune_and_evaluate():
    print("Begin tuning...")
    '''
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=2000,  # change this to 20000 to achieve the best performance
        builder=auto_scheduler.LocalBuilder(build_func="ndk" if use_ndk else "default"),
        runner=auto_scheduler.RPCRunner(
            device_key,
            host=rpc_host,
            port=rpc_port,
            timeout=3000,
            repeat=1,
            min_repeat_ms=200,
            enable_cpu_cache_flush=True,
        ),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)
    #'''

    # Compile with the history best
    print("Compile...")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(
            opt_level=3, config={"relay.backend.use_auto_scheduler": True}
        ):
            lib = relay.build(mod, target=target, params=params)

    # Export library
    tmp = tempdir()
    if use_ndk:
        from tvm.contrib import ndk

        filename = "net.so"
        lib.export_library(tmp.relpath(filename), ndk.create_shared)
    else:
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
    module.set_input("input0", data_tvm)

    # Evaluate
    print("Evaluate inference time cost...")
    print(module.benchmark(dev, repeat=3, min_repeat_ms=500))


# We do not run the tuning in our webpage server since the server doesn't have a Raspberry Pi,
# or device tracker running.
# Uncomment the following line to run it by yourself.

tune_and_evaluate()


######################################################################
# .. note:: Explaining the printed information during tuning
#
#   During the tuning, a lot of information will be printed on the console.
#   They are used for debugging purposes. The most important info is the output
#   of the task scheduler. The following table is a sample output.
#
#   .. code-block:: c
#
#    ----------------------------------------------------------------------
#    ------------------------------  [ Task Scheduler ]
#    ----------------------------------------------------------------------
#    |  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
#    -------------------------------------------------
#    |    0 |        0.013 |           0.31 |     64 |
#    |    1 |        0.845 |           2.43 |    448 |
#    |    2 |        0.046 |          -0.00 |     64 |
#    |    3 |        4.194 |          24.53 |   2112 |
#    |    4 |        0.109 |           9.21 |     64 |
#    |    5 |        1.759 |          29.27 |    896 |
#    |    6 |        0.083 |           6.01 |     64 |
#    |    7 |        3.084 |          33.38 |   7680 |
#    |    8 |        0.136 |          14.78 |    384 |
#    |    9 |        1.349 |          38.23 |    768 |
#    |   10 |        0.133 |           7.55 |    128 |
#    |   11 |        2.747 |          37.56 |   1536 |
#    |   12 |        0.338 |          11.87 |    192 |
#    |   13 |        1.295 |          40.00 |    704 |
#    |   14 |        0.482 |           4.16 |    256 |
#    |   15 |        2.686 |          38.56 |   1344 |
#    |   16 |        0.884 |           9.08 |    448 |
#    |   17 |        1.332 |          39.18 |    704 |
#    |   18 |        1.045 |           3.84 |    576 |
#    |   19 |        1.391 |          38.09 |    704 |
#    |   20 |        0.777 |          10.34 |    448 |
#    |   21 |        0.739 |          30.97 |    448 |
#    -------------------------------------------------
#     Estimated total latency: 38.347 ms      Trials: 19992   Used time : 19260 s     Next ID: 3
#
#   This table lists the latency and (estimated) speed of all tasks.
#   It also lists the allocation of measurement trials for all tasks.
#   The last line prints the total weighted latency of these tasks,
#   which can be a rough estimation of the end-to-end execution time
#   of the network.
#   The last line also prints the total number of measurement trials,
#   total time spent on auto-tuning and the id of the next task to tune.
#
#   There will also be some "dmlc::Error"s errors, because the
#   auto-scheduler will try some invalid schedules.
#   You can safely ignore them if the tuning can continue, because these
#   errors are isolated from the main process.
#

######################################################################
# .. note:: Terminate the tuning earlier
#
#   You can terminate the tuning earlier by forcibly killing this process.
#   As long as you get at least one valid schedule for each task in the log file,
#   you should be able to do the compilation (the secion below).
#

#################################################################
# Other Tips
# ----------
# 1. During the tuning, the auto-scheduler needs to compile many programs and
#    extract feature from them. This part is CPU-intensive,
#    so a high-performance CPU with many cores is recommended for faster search.
# 2. You can use :code:`python3 -m tvm.auto_scheduler.measure_record --mode distill -i log.json`
#    to distill the large log file and only save the best useful records.
# 3. You can resume a search from the previous log file. You just need to
#    add a new argument :code:`load_log_file` when creating the task scheduler
#    in function :code:`run_tuning`. Say,
#    :code:`tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)`
# 4. If you have multiple target CPUs, you can use all of them for measurements to
#    parallelize the measurements. Check this :ref:`section <tutorials-autotvm-scale-up-rpc-tracker>`
#    to learn how to use the RPC Tracker and RPC Server.
#    To use the RPC Tracker in auto-scheduler, replace the runner in :code:`TuningOptions`
#    with :any:`auto_scheduler.RPCRunner`.
