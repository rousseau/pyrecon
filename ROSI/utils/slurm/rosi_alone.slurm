#!/bin/bash

# Source slurm configuration files
# GENERIC CONFIGURATION FOR COMPUTATION ON THE AIX-MARSEILLE MESOCENTRE

# Generic configuration
#SBATCH --account='b219'

# Run configuration
#SBATCH --partition=skylake
#SBATCH -t 24:00:00
#SBATCH --mem-per-cpu=50G
#SBATCH -c 1
#SBATCH -e output_slurm/rosi_alone.%A.err
#SBATCH -o output_slurm/rosi_alone.%A.out



################################################################################
# Perform ROSI-registration on data pre-registered with svort
################################################################################


COMMAND=$1



source activate rosienv
cd /scratch/cmercier/code/pyrecon/ROSI
python run_registration.py $COMMAND