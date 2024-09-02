#!/bin/bash

# Source slurm configuration files
# GENERIC CONFIGURATION FOR COMPUTATION ON THE AIX-MARSEILLE MESOCENTRE

# Generic configuration
#SBATCH --account='b219'  # identifier of the related Mesocentre Project

# Mailing configuration


# Run configuration
#SBATCH --partition=skylake
#SBATCH -c 1
# ask for 20G0 of RAM per cpu
#SBATCH --mem-per-cpu=5G
# Files pattern
#SBATCH -e denoising-%j.err
#SBATCH -o denoising-%j.out

################################################################################
# Perform 3D Gaussian Denoising of stack volumes with default parameters
################################################################################

INPUT_IMAGE=$1
OUTPUT_IMAGE=$2

/scratch/gauzias/softs/ants/bin/DenoiseImage -d 3 -i ${INPUT_IMAGE} -o ${OUTPUT_IMAGE}

