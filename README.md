# Motion Correction of 2D slices MRI

## Registration of Real Data

To correct motion on real data, use the script **ROSI/main_realdata.py**. This script corrects motion in stacks of 2D images acquired in different orientations. It provides the evolution of the error (mse and dice), the evolution of the motion parameters (3 rotations and 3 translations) and the evolution of the estimated transformations between image coordinates and world coordinates. These results are stored in the file "res_reigstration.joblib.gz".

Example : 
```
python main_realdata.py \
--filenames image_axial.nii.gz image_coronal.nii.gz image_sagittal.nii.gz \
--filenames_masks mask_axial.nii.gz mask_coronal.nii.gz mask_sagittal.nii.gz \
--hyperparameters 4 0.25 2000 2 0 0
--ablation multistart no_dice Nelder-Mead
--output res_registration
```

where filenames and output are required. filenames are the stacks of 2D images. filenames_mask are the corresponding brain masks. 

Hyperparameters corresponds to the choice of hyperparameters and represents: delta (size of the initial simplex), xatol (tolerance on parameters for the optimisation algorithm), maxiter (maximum number of iterations in the optimisation algorithm), error (local convolution criterion), omega (number of cubes in the cost function), sigma (value for Gaussian filtering).
**It is recommended to use the suggested parameters**.

Ablation has been added for development purposes. The first parameter allows you to choose between 'no_multisart' and 'multistart'. If you choose 'multistart', multistart will be performed after registration. Second parameter, you can choose between 'dice' and 'no_dice'. The third parameter is the optimisation algorithm. **It is recommended to use Nelder-Mead.


Corrected transformations for the slices are saved in 'res_registration_mvt'. The format of saved transformation is adapted for reconstruction with **NiftyMIC algorithm**[1][2][3].

# Reconstruction with NiftyMIC

It is possible to reconstruct the motion corrected data from our algorithm using NiftyMIC:

After installing NiftyMIC (using Docker is easier), call the function : **niftymic_reconstruct_volume_from_slices.py**. It's a slightly modified version of **niftymic_reconstruct_volume.py** and allows you to run the pipeline suggested in NiftyMIC on the registered data. 
Move **ROSI/rosi/NiftyMIC/niftymic_reconstruct_volume_from_slices.py** into your NiftyMIC folder.

The script do : 
- Brain segmentation
- Biais field correction
- 3D Reconstruction only, using previously estimated transformation
- Volumetric reconstruction in template space (for better visualisation)


Example :
```
python niftymic_reconstruct_volume_from_slices.py \
--filenames image_axial.nii.gz image_coronal.nii.gz image_sagittal.nii.gz
--filenames-masks mask_axial.nii.gz mask_coronal.nii.gz mask_sagittal.nii.gz \
--dir-input-mc res_registration_mvt 
--dir-output reconstruction_niftymic
```


# References 

[1] [EbnerWang2020] Ebner, M., Wang, G., Li, W., Aertsen, M., Patel, P. A., Aughwane, R., Melbourne, A., Doel, T., Dymarkowski, S., De Coppi, P., David, A. L., Deprest, J., Ourselin, S., Vercauteren, T. (2020). An automated framework for localization, segmentation and super-resolution reconstruction of fetal brain MRI. NeuroImage, 206, 116324.

[2] [EbnerWang2018] Ebner, M., Wang, G., Li, W., Aertsen, M., Patel, P. A., Melbourne, A., Doel, T., David, A. L., Deprest, J., Ourselin, S., Vercauteren, T. (2018). An Automated Localization, Segmentation and Reconstruction Framework for Fetal Brain MRI. In Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2018 (pp. 313–320). Springer.

[3] [Ebner2018]PANIST Ebner, M., Chung, K. K., Prados, F., Cardoso, M. J., Chard, D. T., Vercauteren, T., Ourselin, S. (2018). Volumetric reconstruction from printed films: Enabling 30 year longitudinal analysis in MR neuroimaging. NeuroImage, 165, 238–250
