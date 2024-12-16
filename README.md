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

# References 

[1] Mercier, C., Faisan, S., Pron, A., Girard, N., Auzias, G., Chonavel, T., & Rousseau, F. (2023, October). Retrospective motion estimation for fetal brain MRI. In 2023 Twelfth International Conference on Image Processing Theory, Tools and Applications (IPTA) (pp. 1-6). IEEE.

