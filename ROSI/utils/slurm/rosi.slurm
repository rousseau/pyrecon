#!/bin/bash

# Source slurm configuration files
# GENERIC CONFIGURATION FOR COMPUTATION ON THE AIX-MARSEILLE MESOCENTRE

# Generic configuration
#SBATCH --account='b219'

# Run configuration
#SBATCH --partition=skylake
#SBATCH -t 04-00:00:00
#SBATCH -n 10
#SBATCH -e output_slurm/rosi.%A_%j.err
#SBATCH -o output_slurm/rosi.%A_%j.out
#SBATCH --array=1-300%10



################################################################################
# Perform ROSI-registration on data pre-registered with svort
################################################################################


INPUT_SLICES=$1
OUPUT_DIR=$2



source activate rosienv
cd /scratch/cmercier/code/pyrecon/ROSI
echo INPUT_SLICES
echo $INPUT_SLICES
python run_registration_svort.py --filenames ${INPUT_SLICES} --output ${OUPUT_DIR} --no_multistart 1