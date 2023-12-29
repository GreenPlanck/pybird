#!/bin/sh
#SBATCH --time=48:00:00 
#SBATCH --partition=cpu
#SBATCH --output=log/sample.o%j  # Standard output and error log

#SBATCH --mem-per-cpu=4G

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1  # This is needed for openmp

#echo "#SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#echo "#OMP_NUM_THREADS   : $OMP_NUM_THREADS"

module load gcc/8.2.0
module load python/3.8.5
module load openmpi/4.1.4
cd /cluster/work/senatore/simon_bisp_montepython/

mpirun -n 16 python -u montepython/MontePython.py run -o chains/$1/$2 --superupdate 20 -f $3 -T $4 -N 10000 -b chains/$1/$2/$2.bestfit -c chains/$1/$2/$2.covmat

