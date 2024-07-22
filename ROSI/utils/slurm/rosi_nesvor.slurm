#!/bin/bash

# Source slurm configuration files
# GENERIC CONFIGURATION FOR COMPUTATION ON THE AIX-MARSEILLE MESOCENTRE

# Generic configuration
#SBATCH --account='b219'

# Run configuration
#SBATCH --partition=volta
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH -e output_slurm/nesvor.%A_%j.err
#SBATCH -o output_slurm/nesvor.%A_%j.out



################################################################################
# Perform ROSI-registration on data pre-registered with svort
################################################################################


COMMAND_1=$1
COMMAND_2=$2
DATA=$3

source activate rosienv
cd /scratch/cmercier/code/pyrecon/ROSI
python convert_to_svort.py $COMMAND_1
singularity exec --nv -B $DATA:/data /scratch/cmercier/softs/nesvor_latest.sif nesvor reconstruct $COMMAND_2