#!/bin/bash
#SBATCH --job-name=GPU-dpo-random
#SBATCH --partition=kill-exclusive
## 3 day max run time for community.q, kill.q, exclusive.q, and htc.q. 1 Hour max run time for sb.q
#SBATCH --time=00-10:00:00 ## time format is DD-HH:MM:SS

#SBATCH --cpus-per-task=1
#SBATCH --mem=100G ## max amount of memory per node you require
#SBATCH --core-spec=0 ## Uncomment to allow jobs to request all cores on a node

#SBATCH --gres=gpu:NV-V100-SXM2:4  ## request both GPUs in the GPU node.
### To request only 1 of the two GPUs in the node, you would do: gpu:NV-K40:1
## GPU TYPES:
## NV-K40        [systems have 2 of these] gpu-00[01-02]
## NV-RTX2080Ti  [systems have either 4 or 8 of these] gpu-00[03], gpu-00[04-08]
## NV-RTX2070    [systems have 8 of these] gpu-00[09]

#SBATCH --error=gpu-test-%A.err ## %A - filled with jobid
#SBATCH --output=gpu-test-%A.out ## %A - filled with jobid

## Useful for remote notification
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=honggen@hawaii.edu

## All options and environment variables found on schedMD site: http://slurm.schedmd.com/sbatch.html
module purge
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
module load  lang/Anaconda3

source activate DPO
python -u train.py model=pythia28 datasets=[hh] loss=dpo loss.beta=0.1 exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=2 batch_size=16 eval_batch_size=16 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=./cache/honggen/anthropic_dpo_pythia28_2024-03-02_sft/policy.pt
##bandwidthTest
##cudaOpenMP
##deviceQuery
##simpleMultiGPU
