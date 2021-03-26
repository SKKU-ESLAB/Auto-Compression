#!/bin/bash
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
echo $THIS_DIR
cd $THIS_DIR
cfg=$1
bs=${2:-256}
bs_per_gpu=${3:-64}
kappa=${4:-1}
num_gpus=$(( bs / bs_per_gpu ))
num_nodes=$(( ( num_gpus - 1 ) / 8 + 1 ))
num_proc_per_nodes=$(( num_gpus < 8 ? num_gpus : 8 ))
echo $bs
echo $bs_per_gpu
echo $num_gpus
echo $num_nodes
echo $num_proc_per_nodes
if [ ! -f $cfg ]; then
  echo "Config not found!"
fi
#pip3 install torch==1.2.0 torchvision==0.4.0
#pip3 install -U memory_profiler
python3 -m torch.distributed.launch --nproc_per_node=$num_proc_per_nodes --nnodes=$num_nodes --node_rank=$METIS_TASK_INDEX --master_addr=$METIS_WORKER_0_HOST --master_port=$METIS_WORKER_0_PORT train.py app:$cfg bs:$bs kappa:$kappa
