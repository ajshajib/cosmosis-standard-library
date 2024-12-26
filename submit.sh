#!/bin/bash

#SBATCH --output=/home/ajshajib/Logs"/joblog.%j"
#SBATCH --error=/home/ajshajib/Logs"/error.%j"
#SBATCH --partition=caslake
#SBATCH --account=pi-jfrieman                                                          
#SBATCH -t 36:00:00
#SBATCH --cpus-per-task=1                                                        
#SBATCH --mail-user=ajshajib@rcc.uchicago.edu
#SBATCH --mail-type=FAIL 

source /home/ajshajib/.bashrc
conda activate cosmosis
source cosmosis-configure

export JOB_NAME=${RUN_NAME}${MOD_NAME}

export OMP_NUM_THREADS=1

mpirun -n ${NUM_PROC} cosmosis --mpi /scratch/midway3/ajshajib/cosmosis-standard-library/inis/${RUN_NAME}.ini
