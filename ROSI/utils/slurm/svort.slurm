#!/bin/bash

# Source slurm configuration files
# GENERIC CONFIGURATION FOR COMPUTATION ON THE AIX-MARSEILLE MESOCENTRE

# Generic configuration
#SBATCH --account='b219'

# Run configuration
#SBATCH --partition=volta
#SBATCH -t 24:00:00
#SBATCH --mem-per-cpu=50G
#SBATCH -c 1
#SBATCH -e output_slurm/svort.%A_%j.err
#SBATCH -o output_slurm/svort.%A_%j.out



################################################################################
# Perform ROSI-registration on data pre-registered with svort
################################################################################


INPUT_SLICES=$1
INPUT_MASK=$2
OUTPUT_DIR=$3
DATA=/scratch/gauzias/data/datasets/MarsFet/derivatives/preprocessing


source activate rosienv
cd /scratch/cmercier/code/pyrecon/ROSI
echo INPUT_SLICES
echo $INPUT_SLICES
echo $DATA
echo $OUTPUT_DIR
singularity exec --nv -B $DATA:/data /scratch/cmercier/softs/nesvor_latest.sif nesvor register --input-stacks $INPUT_SLICES --stack-masks $INPUT_MASK --output-slices $OUTPUT_DIR