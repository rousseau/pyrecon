## Registration Based on Orthogonal Slices Intersection (ROSI) 
ROSI performs registration on fetal MRI stacks of 2D slices. To perform 3D reconstruction, results can be plugged to other 3D reconstruction method (ex: NiftyMIC (https://github.com/gift-surg/NiftyMIC), or NesVOR (https://github.com/daviddmc/NeSVoR)).

ROSI consists in three stages: 1. motion correction is applied to all slices. 2. a trained classifier is used to identify potentially misaligned slices. 3. a multi-start optimization approach allows for the correction of potentially misaligned slices.

Results are saved in a joblib format, wich contains: 
- the list of slices
- the value of the cost at each iteration
- the value of the dice coefficient at each iteration
- the value of the squared error between each pair of slices, at each iteration
- the number of points on which the square error is computed between each pair of slices, at each iteration
- the number of intersecting points between each pair of slices, at each iteration
- the number points on the union between each pair of slices, at each iteration
- the list of indices corresponding to rejected slices
- the list containing features for all slices.
- (the Target Registration Error (TRE) for each slices)

# To run ROSI on real data : 
A basic usage is : 

```
python run_registration.py --filenames stack_1.nii.gz stack_2.nii.gz stack_3.nii.gz --filenames_mask brainmask_1.nii.gz brainmask_2.nii.gz brainmask_3.nii.gz --output name_output
```

Eventually, you could specify different parameters for the optimization, for example:
```
python run_registration.py --filenames stack_1.nii.gz stack_2.nii.gz stack_3.nii.gz --filenames_mask brainmask_1.nii.gz brainmask_2.nii.gz brainmask_3.nii.gz --output name_output --intial_simplex 4 --final_simplex 0.25 --local_convergence 0.25 --omega 0 --optimisation "Nelder-Mead"
```

If you don't want to use multistart, specify : 
```
python run_registration.py --filenames stack_1.nii.gz stack_2.nii.gz stack_3.nii.gz --filenames_mask brainmask_1.nii.gz brainmask_2.nii.gz brainmask_3.nii.gz --output name_output --no_multistart 1
```

# To run ROSI on simulated data :

Data can be simulated with : 
```
python scriptSimulData.py --hr HR_image.nii.gz --mask HR_brainmask.nii.gz --output ouptut_directory --motion -3 3
```
Then you can run ROSI with : 
```
python run_registration.py --filenames stack_1.nii.gz stack_2.nii.gz stack_3.nii.gz --filenames_mask brainmask_1.nii.gz brainmask_2.nii.gz brainmask_3.nii.gz --output name_output --tre 1
```
The Joblib output will include the TRE value for each slice.

##3D Reconstruction using NiftyMIC and Nesvor

In the folowing, we explain how to reconstruct a 3D volume with Nesvor or NiftyMIC, using singularity images

#With NiftyMIC

To run ROSI with NiftyMIC, run the folowing command: 

```
python run_registration.py --filenames stack_1.nii.gz stack_2.nii.gz stack_3.nii.gz --filenames_mask brainmask_1.nii.gz brainmask_2.nii.gz brainmask_3.nii.gz --output rosi_output 

python convert_to_niftimic.py  --input-stacks stack_1.nii.gz stack_2.nii.gz stack_3.nii.gz --input_mask brainmask_1.nii.gz brainmask_2.nii.gz brainmask_3.nii.gz --results rosi_output.joblib.gz --output dir_motion 

path_to_res='/ouptut_path' #change here to your output path
path_to_reconstruction='ROSI/rosi/reconstruction'
Data='/data_path' #change to the path to your data 


singularity exec --no-home -B ${Data}:/data,${path_to_res}:/results,${path_to_reconstruction}/monai_dynunet_inference.py:/app/NiftyMIC/application/monai_dynunet_inference.py
${path_to_reconstruction}/run_reconstruction_pipeline_slices.py:/app/NiftyMIC/niftymic/application/run_reconstruction_pipeline_slices.py
${path_to_reconstruction}/niftymic_run_reconstruction_pipeline_slices.py:/app/NiftyMIC/niftymic_run_reconstruction_pipeline_slices.py 
niftymic.multifact_latest.sif python /app/NiftyMIC/niftymic_run_reconstruction_pipeline_slices.py --filenames stack_1.nii.gz stack_2.nii.gz stack_3.nii.gz --filenames-masks brainmask_1.nii.gz brainmask_2.nii.gz brainmask_3.nii.gz --dir-output niftymic_output --dir-input-mc dir_input
```
# With NeSVoR:

## Without initialization from SvoRT:

To run ROSI with NeSVoR without SvoRT initialization, execute the following commands:

```bash
python run_registration.py --filenames stack_1.nii.gz stack_2.nii.gz stack_3.nii.gz --filenames_mask brainmask_1.nii.gz brainmask_2.nii.gz brainmask_3.nii.gz --output rosi_output

python convert_to_svort.py --input-stacks stack_1.nii.gz stack_2.nii.gz stack_3.nii.gz --input-mask brainmask_1.nii.gz brainmask_2.nii.gz brainmask_3.nii.gz --results rosi_output.joblib.gz --output output_rosi_slices

DATA='/data_path' # Change this to the actual path to your data

singularity exec --nv -B $DATA:/data /scratch/cmercier/softs/nesvor_latest.sif nesvor reconstruct --input-slices output_rosi_slices --output-volumes output_nesvor --registration none --no-transformation-optimization --inference-batch-size 255 --n-inference-samples 128
```


To run ROSI with NeSVoR, run the following command, #with initilisation from SvoRT:

```bash

DATA='/data_path' # Change this to the actual path to your data

singularity exec --nv -B $DATA:/data /scratch/cmercier/softs/nesvor_latest.sif nesvor register --input-stacks stack_1.nii.gz stack_2.nii.gz stack_3.nii.gz --stack-masks brainmask_1.nii.gz brainmask_2.nii.gz brainmask_3.nii.gz --output-slices input_rosi_slices

python run_registration_svort.py --filenames input_rosi_slices --output rosi_output

python convert_to_svort_init.py --input-slices input_rosi_slices --results rosi_output.joblib.gz --output output_rosi_slices

# N.B. The input slices for `convert_to_svort_init.py` are the output of SvoRT. This is done to preserve the intensity normalization performed by SvoRT. The motion of the slices is corrected using the transformation estimated in `rosi_output.joblib.gz`.

singularity exec --nv -B $DATA:/data /scratch/cmercier/softs/nesvor_latest.sif nesvor reconstruct --input-slices output_rosi_slices --output-volumes output_nesvor --registration none --no-transformation-optimization --inference-batch-size 255 --n-inference-samples 128

```

# References 

[1] Mercier, C., Faisan, S., Pron, A., Girard, N., Auzias, G., Chonavel, T., & Rousseau, F. (2023, October). Retrospective motion estimation for fetal brain MRI. In 2023 Twelfth International Conference on Image Processing Theory, Tools and Applications (IPTA) (pp. 1-6). IEEE.

