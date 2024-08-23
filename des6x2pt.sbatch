#!/bin/bash

#SBATCH --job-name="cosmosis"
#SBATCH --output=/home/ajshajib/Logs"/joblog.%j"
#SBATCH --error=/home/ajshajib/Logs"/error.%j"
#SBATCH --partition=broadwl
#SBATCH --account=pi-jfrieman                                                          
#SBATCH -t 36:00:00      
#SBATCH --ntasks=56
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2000
#SBATCH --exclusive                                                          
#SBATCH --mail-user=ajshajib@uchicago.edu                                   
#SBATCH --mail-type=ALL    

source /home/ajshajib/.bashrc
conda activate cosmosis
source cosmosis-configure

which conda
which cosmosis

export OMP_NUM_THREADS=2
export OMP_PROC_BIND=true
export OMP_PLACES=threads
mpirun -n 56 cosmosis --mpi /home/ajshajib/cosmosis-standard-library/inis/des-y3-maglim.ini
