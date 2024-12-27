#!/bin/bash

#SBATCH --job-name="sn"
#SBATCH --output=/home/ajshajib/Logs"/joblog.%j"
#SBATCH --error=/home/ajshajib/Logs"/error.%j"
#SBATCH --partition=caslake
#SBATCH --account=pi-jfrieman                                                          
#SBATCH -t 36:00:00      
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mail-user=ajshajib@uchicago.edu                                   
#SBATCH --mail-type=ALL    

source /home/ajshajib/.bashrc
conda activate cosmosis
source cosmosis-configure

mpirun -n 48 cosmosis --mpi /scratch/midway3/ajshajib/cosmosis-standard-library/inis/sn.ini
