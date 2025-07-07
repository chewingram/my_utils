#!/bin/bash
# Submission script for MareNostrum5
#SBATCH --job-name=NVT
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --ntasks=
#SBATCH --cpus-per-task=1
#SBATCH --time=
#SBATCH --partition=gpp
#SBATCH --qos=gp_ehpc
#SBATCH --account=ehpc14
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
#source /gpfs/home/ulie/ulie583683/environments/intel_2023.2.0
source /gpfs/home/ulie/ulie583683/environments/gcc_12.3.0_p3.12.1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo -n "The job started on "
date
####################################################################################
# The actual job
python RunNVT.py

echo -n "The job finished on "
date

