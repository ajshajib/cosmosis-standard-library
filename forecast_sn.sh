#!/bin/bash

#SBATCH --job-name="sn_forecast"
#SBATCH --output=/home/ajshajib/Logs"/joblog.%j"
#SBATCH --error=/home/ajshajib/Logs"/error.%j"
#SBATCH --partition=caslake
#SBATCH --account=pi-jfrieman                                                          
#SBATCH -t 36:00:00      
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1                                                          
#SBATCH --mail-user=ajshajib@rcc.uchicago.edu                                   
#SBATCH --mail-type=FAIL   

source /home/ajshajib/.bashrc
conda activate cosmosis
source cosmosis-configure

export OMP_NUM_THREADS=1
mpirun -n 96 cosmosis --mpi /home/ajshajib/cosmosis-standard-library/forecast_inis/srd_sn.ini
