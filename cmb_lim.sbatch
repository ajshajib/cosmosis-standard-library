#!/bin/bash

#SBATCH --job-name="cmb_lim"
#SBATCH --output=/home/ajshajib/Logs"/joblog.%j"
#SBATCH --error=/home/ajshajib/Logs"/error.%j"
#SBATCH --partition=caslake
#SBATCH --account=pi-jfrieman                                                          
#SBATCH -t 36:00:00      
#SBATCH --ntasks=192
#SBATCH --cpus-per-task=1                                                      
#SBATCH --mail-user=ajshajib@gmail.com                                   
#SBATCH --mail-type=FAIL   

source /home/ajshajib/.bashrc
conda activate cosmosis
source cosmosis-configure

which cosmosis

export OMP_NUM_THREADS=1
mpirun -n 192 cosmosis --mpi /scratch/midway3/ajshajib/cosmosis-standard-library/inis/planck_lim.ini
