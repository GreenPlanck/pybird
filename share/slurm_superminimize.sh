folder=min10quad

# f=0.5
# T=0.1

# f=0.1
# T=0.005

f=0.005
T=0.001

for chain in superminLcdmBPLSS
#for chain in superminLcdmPLSSbao
#for chain in superminLcdmBPLSS superminLcdmPLSS superminpk_LcdmBPLSS
do
	sbatch -A es_senatore slurm/superminimize.sh $folder $chain $f $T
done
