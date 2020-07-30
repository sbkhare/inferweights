#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=sbkhare@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-4:0:00
#SBATCH --mem=16gb
#SBATCH --partition=general
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=evolve_n2
#SBATCH --output=n2_out
#SBATCH --error=error_n2

######  Module commands #####
module load python/3.6.9



######  Job commands go below this line #####
python3 evolve_ctrnn_osc.py $SLURM_ARRAY_TASK_ID