#!/bin/bash

# Source slurm configuration files
# GENERIC CONFIGURATION FOR COMPUTATION ON THE AIX-MARSEILLE MESOCENTRE

# Generic configuration
#SBATCH --account='b219'

# Run configuration
#SBATCH --partition=volta
#SBATCH -t 00:05:00
#SBATCH --gres=gpu:1
#SBATCH -e output_slurm/niftymic.%A_%j.err
#SBATCH -o output_slurm/niftymic.%A_%j.out



################################################################################
# Perform ROSI-registration on data pre-registered with svort
################################################################################


COMMAND_1=$1
Data=$2
path_to_reconstruction=$3
path_to_res=$4


source activate rosienv
cd /scratch/cmercier/code/pyrecon/ROSI
echo $Data
echo $path_to_reconstruction
echo $path_to_res
#python /app/NiftyMIC/niftymic_run_reconstruction_pipeline_slices.py $COMMAND_1
#singularity exec -B $DATA:/data,$SCRIPT:/app/NiftyMIC/niftymic/application /scratch/cmercier/softs/niftymic.multifact_latest.sif /app/NiftyMIC/niftymic/application/niftymic_run_reconstruction_pipeline_slices.py
singularity run --no-home -B ${Data}:/data,${path_to_res}:/results,${path_to_reconstruction}/monai_dynunet_inference.py:/app/NiftyMIC/application/monai_dynunet_inference.py,${path_to_reconstruction}/run_reconstruction_pipeline_slices.py:/app/NiftyMIC/application/run_reconstruction_pipeline_slices.py,${path_to_reconstruction}/niftymic_run_reconstruction_pipeline_slices.py:/app/NiftyMIC/niftymic_run_reconstruction_pipeline_slices.py  /scratch/cmercier/softs/niftymic.multifact_latest.sif