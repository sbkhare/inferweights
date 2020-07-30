#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=sbkhare@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-99%25
#SBATCH --time=0-6:0:00
#SBATCH --mem=16gb
#SBATCH --partition=general
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=infer_wts_n3
#SBATCH --output=n3_out
#SBATCH --error=error_n3

######  Module commands #####
module load python/3.6.9



######  Job commands go below this line #####
cd /N/u/sbkhare/BigRed3/I698/inferweights/n3/0
current_dir=${PWD}
echo "Infer Weights " $SLURM_ARRAY_TASK_ID
python3 infer_weights.py ${current_dir: -1} $SLURM_ARRAY_TASK_ID
echo "Done " $SLURM_ARRAY_TASK_ID
