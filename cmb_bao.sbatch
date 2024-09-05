#!/bin/bash

#SBATCH --job-name="cosmosis"
#SBATCH --output=/home/ajshajib/Logs"/joblog.%j"
#SBATCH --error=/home/ajshajib/Logs"/error.%j"
#SBATCH --partition=broadwl
#SBATCH --account=pi-jfrieman                                                          
#SBATCH -t 36:00:00      
#SBATCH --ntasks=112
#SBATCH --mem-per-cpu=2000
#SBATCH --exclusive                                                          
#SBATCH --mail-user=ajshajib@uchicago.edu                                   
#SBATCH --mail-type=ALL    

source /home/ajshajib/.bashrc
conda activate cosmosis
source cosmosis-configure

which conda
which cosmosis

mpirun -n 112 cosmosis --mpi /home/ajshajib/cosmosis-standard-library/inis/desi_planck_lodha24.ini
