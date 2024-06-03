## Registration Based on Orthogonal Slices Intersection (ROSI) 
ROSI performs registration on fetal MRI stacks of 2D slices. To perform 3D reconstruction, results can be plugged to other 3D reconstruction method (ex: NiftyMIC [3][4][5], or NesVOR [6]).

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

[1] Mercier, C., Faisan, S., Pron, A., Girard, N., Auzias, G., Chonavel, T., & Rousseau, F. (2023, August). Estimation de mouvement rétrospective pour l'IRM cérébrale foetale. In GRETSI 2023.

[2] Mercier, C., Faisan, S., Pron, A., Girard, N., Auzias, G., Chonavel, T., & Rousseau, F. (2023, October). Retrospective motion estimation for fetal brain MRI. In 2023 Twelfth International Conference on Image Processing Theory, Tools and Applications (IPTA) (pp. 1-6). IEEE.

[3] Ebner, M., Wang, G., Li, W., Aertsen, M., Patel, P. A., Aughwane, R., Melbourne, A., Doel, T., Dymarkowski, S., De Coppi, P., David, A. L., Deprest, J., Ourselin, S., Vercauteren, T. (2020). An automated framework for localization, segmentation and super-resolution reconstruction of fetal brain MRI. NeuroImage, 206, 116324.

[4] Ebner, M., Wang, G., Li, W., Aertsen, M., Patel, P. A., Melbourne, A., Doel, T., David, A. L., Deprest, J., Ourselin, S., Vercauteren, T. (2018). An Automated Localization, Segmentation and Reconstruction Framework for Fetal Brain MRI. In Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2018 (pp. 313–320). Springer.

[5] Ebner, M., Chung, K. K., Prados, F., Cardoso, M. J., Chard, D. T., Vercauteren, T., Ourselin, S. (2018). Volumetric reconstruction from printed films: Enabling 30 year longitudinal analysis in MR neuroimaging. NeuroImage, 165, 238–250

[6] Xu, J., Moyer, D., Gagoski, B., Iglesias, J. E., Grant, P. E., Golland, P., & Adalsteinsson, E. (2023). Nesvor: Implicit neural representation for slice-to-volume reconstruction in mri. IEEE Transactions on Medical Imaging.
