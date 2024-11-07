#!/bin/bash

# Source slurm configuration files
# GENERIC CONFIGURATION FOR COMPUTATION ON THE AIX-MARSEILLE MESOCENTRE

# Generic configuration
#SBATCH --account='b219'

# Run configuration
#SBATCH --partition=volta
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:1
#SBATCH -e output_slurm/svort.%A.err
#SBATCH -o output_slurm/svort.%A.out



################################################################################
# Perform ROSI-registration on data pre-registered with svort
################################################################################


COMMAND=$1
DATA=$2


source activate rosienv
cd /scratch/cmercier/code/pyrecon/ROSI
echo $COMMAND
echo $DATA
singularity exec --nv -B $DATA:/data /scratch/cmercier/softs/nesvor_latest.sif nesvor register $COMMAND
