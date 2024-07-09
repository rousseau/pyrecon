#!/bin/bash

# Source slurm configuration files
# GENERIC CONFIGURATION FOR COMPUTATION ON THE AIX-MARSEILLE MESOCENTRE

# Generic configuration
#SBATCH --account='b219'

# Run configuration
#SBATCH --partition=skylake
#SBATCH -t 00-30:00:00
#SBATCH -n 10
#SBATCH -e rosi.%A_%j.err
#SBATCH -o rosi.%A_%j.out
#SBATCH --array=1-300%10



################################################################################
# Perform ROSI-registration on data pre-registered with svort
################################################################################


INPUT_SLICES=$1
OUPUT_DIR=$2


conda activate rosienv
cd /scratch/cmercier/code/pyrecon/ROSI
run_registration_svort.py --filenames ${INPUT_SLICES} --output ${OUPUT_DIR} 