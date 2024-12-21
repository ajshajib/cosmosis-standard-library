#!/bin/bash

#SBATCH --job-name="cosmosis"
#SBATCH --output=/home/ajshajib/Logs"/joblog.%j"
#SBATCH --error=/home/ajshajib/Logs"/error.%j"
#SBATCH --partition=caslake
#SBATCH --account=pi-jfrieman                                                          
#SBATCH -t 36:00:00      
#SBATCH --ntasks=192
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000
#SBATCH --exclusive                                                          
#SBATCH --mail-user=ajshajib@gmail.com                                   
#SBATCH --mail-type=FAIL  

source /home/ajshajib/.bashrc
conda activate cosmosis
source cosmosis-configure

which cosmosis

mpirun -n 192 cosmosis --mpi /scratch/midway3/ajshajib/cosmosis-standard-library/forecast_inis/desi_ext.ini