#!/bin/bash

# Source slurm configuration files
# GENERIC CONFIGURATION FOR COMPUTATION ON THE AIX-MARSEILLE MESOCENTRE

# Generic configuration
#SBATCH --account='b219'

# Run configuration
#SBATCH --partition=volta
#SBATCH -t 00:05:00
#SBATCH --gres=gpu:1
#SBATCH -e output_slurm/nesvor.%A_%j.err
#SBATCH -o output_slurm/nesvor.%A_%j.out



################################################################################
# Perform ROSI-registration on data pre-registered with svort
################################################################################


COMMAND_1=$1
SCRIPT_1=$2
SCIRPT_2=$3
DATA=$3

source activate rosienv
cd /scratch/cmercier/code/pyrecon/ROSI
python convert_to_svort_init.py $COMMAND_1
singularity exec -B $DATA:/data,$SCRIPT:/app/NiftyMIC /scratch/cmercier/softs/niftymic.multifact_latest.sif python /app/NiftyMIC/niftymic_run_reconstruction_pipeline_slices.py $COMMAND_1

singularity exec -B $DATA:/data,$SCRIPT:/app/NiftyMIC/niftymic/application /scratch/cmercier/softs/niftymic.multifact_latest.sif /app/NiftyMIC/niftymic/application/niftymic_run_reconstruction_pipeline_slices.py
