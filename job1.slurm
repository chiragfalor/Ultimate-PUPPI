#!/bin/bash
#SBATCH -J sluma_4GPUs
#SBATCH -o sluma_4GPUs_%j.out
#SBATCH -e sluma_4GPUs_%j.err
#SBATCH --mail-user=cfalor@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --partition=sched_system_all_8

## User python environment
HOME2=/home/$(whoami)
#HOME2=/nobackup/users/paus
PYTHON_VIRTUAL_ENVIRONMENT=pyt6
CONDA_ROOT=$HOME2/.conda

## Activate WMLCE virtual environment
source /home/software.ppc64le/spack/v0.16.2/spack/opt/spack/linux-rhel8-power9le/gcc-8.3.1/anaconda3-2020.02-2ks5tchtak3kzzbryjloiqhusujnh67c/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

cd /home/cfalor/
#cd /home/paus/puma/grapple/
# export PYTHONPATH=${PYTHONPATH}:${PWD}
cd -
nvidia-smi


ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE


####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
echo " Run started at:- "
date

cd /nobackup/users/cfalor/upuppi/Ultimate-PUPPI/
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
python retrain_model.py

echo "Run completed at:- "
date
