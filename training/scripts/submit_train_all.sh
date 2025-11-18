#!/bin/bash
#SBATCH --job-name=train-nn-gpu
#SBATCH -t 00:10:00                  # estimated time # TODO: adapt to your needs/ the full training run will take approx. 5 h on an A100
#SBATCH -p kisski            # the partition you are training on (i.e., which nodes), for nodes see sinfo -o "%25N  %5c  %10m  %32f  %10G %18P " | grep gpu
#SBATCH -G A100:1                # request one A100 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32         # 8 CPU cores
#SBATCH --mem-per-cpu=8000          # Request 8GB of shared memory per CPU core (total 64GB per node)
#SBATCH --mail-type=all              # send mail when job begins and ends
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error

# path = . will be the directory from which you submit the job with sbatch

# load new software stack
# see https://docs.hpc.gwdg.de/software_stacks/index.html
export PREFERRED_SOFTWARE_STACK=nhr-lmod
source /sw/etc/profile/profile.sh

export TMPDIR=$TMPDIR
export PYTHONPATH=${HOME}/cross_domain_syntra
export PROJECTDIR=/mnt/vast-kisski/projects/kisski-gauss-zentrum
echo $PYTHONPATH
# ulimit 65535
ulimit -f unlimited


module load apptainer
module load gcc/11.5.0
module load cuda/11.8.0
# source activate dl-gpu # Or whatever you called your environment.

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# apptainer run --bind ${HOME}:${HOME},${PROJECTDIR}:${PROJECTDIR} --nv ${PROJECTDIR}/syntra.sif df -h /dev/shm

# Run the script:
apptainer run --bind ${HOME}:${HOME},${PROJECTDIR}:${PROJECTDIR} --nv ${PROJECTDIR}/syntra.sif \
        python -u ${PYTHONPATH}/syntra/train_one_gpu.py \
        --config syntra_all.yaml -d ${PROJECTDIR}/syntra_data \
        -o ${PROJECTDIR}/logs --resume ${PROJECTDIR}/logs/syntra_all/model_latest.pth

# Run the script:
# PYTHONPATH=. python -u syntra/train_one_gpu.py --config syntra_cityscapes_n4_slurm.yaml -d ${PROJECT}/dyntra_data/cityscapes

# Run the script with logger:
#python -u train_with_logger.py -l ~/${SLURM_JOB_NAME}_${SLURM_JOB_ID}  -t True -p True -d True -s True -f True
