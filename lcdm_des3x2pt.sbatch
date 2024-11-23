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
#SBATCH --mail-user=ajshajib@uchicago.edu                                   
#SBATCH --mail-type=ALL    

source /home/ajshajib/.bashrc
conda activate cosmosis
source cosmosis-configure

which conda
which cosmosis

export OMP_NUM_THREADS=1
mpirun -n 192 cosmosis --mpi /scratch/midway3/ajshajib/cosmosis-standard-library/inis_lcdm/des-y3-maglim.ini