#!/bin/bash

# Source slurm configuration files
# GENERIC CONFIGURATION FOR COMPUTATION ON THE AIX-MARSEILLE MESOCENTRE

# Generic configuration
#SBATCH --account='b219'

# Run configuration
#SBATCH --partition=skylake
#SBATCH -t 05:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH -c 1
#SBATCH -e output_slurm/ms.%A.err
#SBATCH -o output_slurm/ms.%A.out



################################################################################
# Perform ROSI-registration on data pre-registered with svort
################################################################################


COMMAND=$1



source activate rosienv
cd /scratch/cmercier/code/pyrecon/ROSI
python test_multi_start_real.py $COMMAND