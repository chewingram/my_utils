#!/bin/bash
# Submission script for NIC5
#SBATCH --job-name=NPT
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --ntasks=64
#SBATCH --mem-per-cpu=4000
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --partition=batch
#SBATCH --requeue

exec > ${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOBID}.log
echo "------------------ Work dir --------------------"
cd ${SLURM_SUBMIT_DIR} && echo ${SLURM_SUBMIT_DIR}
echo "------------------ Job Info --------------------"
echo "jobid : $SLURM_JOBID"
echo "jobname : $SLURM_JOB_NAME"
#echo "job type : $PBS_ENVIRONMENT"  I CANNOT FIND THE SLURM ANALOGUE
echo "submit dir : $PWD"
echo "queue : $SLURM_JOB_PARTITION"
echo "user : $SLURM_JOB_USER"
echo "threads : $OMP_NUM_THREADS"


####################################################################################
# Load modules and python environment
module --force purge
source /home/users/s/l/slongo/modules/abinit_2020b
source /scratch/users/s/l/slongo/venvs/env3.8.6/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo -n "The job started on "
date
####################################################################################
# The actual job
python RunNPT_instance.py

echo -n "The job finished on "
date

