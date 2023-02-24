# Motion Correction of 2D MRI
## Data simulation

To simulate LR (low-resolution) data with inter-slice motion from HR (High-resolution) image use the script **sciptSimulData.py**. The scipt simulates 3 LR images in 3 different orientation (Axial, Coronnal, and Sagittal) and save them in a specified directory, along with the simulated parameters and transformation.
Already simulated data are available in data directory.

## Registration on Simulated Data

To correct motion on simulated data use the script **main.py**. The script provides the evolution of the error (mse and dice), the evolution of the motion parameters (3 rotations and 3 translations), the evolution of the transformations between images coordinates and world coordinates and the error of registration (in mm). Those results can be visualised with the script **3Ddisplay.ipynb**.


Example : 
```
python main.py \
--filenames image_axial.nii.gz image_coronal.nii.gz image_sagittal.nii.gz \
--filenames_masks mask_axial.nii.gz mask_coronal.nii.gz mask_sagittal.nii.gz \
--simulation transformation_axial.py transformation_cornal.py transformation_sagitall.py
--ablation all
--hyperparameters 4 0.25 1e-10 2 1 4
--output res_registration
```

where filenames, simulatoin, ablationn, hyperparameters and output are required. filenames are the stacks of 2D images. filenames_mask are the brain mask and should be in the same order than filenames. 
ablation is a parameters adeed for the ablation study. If you want the algorithm to perform normally choose 'all'. You can choose between 'no_dice','no_gaussian' and 'no_multistart'.
hyperparameters corespond to the choice of hyperparameters and represents : delta (the size of the initial simplex), xatol (tolerance on parameters for the optimisation algorithm), fatol (tolerance on function value for the optimisation algorithm), error (the accepted error between position of a slices before and after registration), omega (ponderation of dice in the cost function), sigma (value for gaussian filtering).

Corrected transformation for the slices are saved in 'res_registration_mvt'. The format of saved transformation is adapted for reconstruction with **NiftyMIC algorithm**[1][2][3].

## Registration on Real Data

To correct motion on real data use the script **main_realdata.py**. This script correct movement in stacks of 2D images acquiered in different orientation. It provides the evolution of the error (mse and dice), the evolution of the motion parameters (3 rotations and 3 translations) and the evolution of the transformations between images coordinates and world coordinates. Those results can be visualised with the script **3Ddisplay_realdata.ipynb**

Example : 
```
python main_realdata.py \
--filenames image_axial.nii.gz image_coronal.nii.gz image_sagittal.nii.gz \
--filenames_masks mask_axial.nii.gz mask_coronal.nii.gz mask_sagittal.nii.gz \
--ablation all
--hyperparameters 4 0.25 1e-10 2 1 4
--output res_registration
```

where filenames and output are required. filenames are the stacks of 2D images. filenames_mask are the brain mask and should be in the same order than filenames.

ablation is a parameters adeed for the ablation study. If you want the algorithm to perform normally choose 'all'. You can choose between 'no_dice','no_gaussian' and 'no_multistart'.

hyperparameters corespond to the choice of hyperparameters and represents : delta (the size of the initial simplex), xatol (tolerance on parameters for the optimisation algorithm), fatol (tolerance on function value for the optimisation algorithm), error (the accepted error between position of a slices before and after registration), omega (ponderation of dice in the cost function), sigma (value for gaussian filtering).


Corrected transformation for the slices are saved in 'res_registration_mvt'. The format of saved transformation is adapted for reconstruction with **NiftyMIC algorithm**[1][2][3].

# Reconstruction with NiftyMIC

It is possible to reconstruct the motion-corrected data from our algorithm with NiftyMIC :

After installing NiftyMIC (using docker is easier), call function : **niftymic_reconstruct_volume_from_slices.py**

Example :
```
python niftymic_reconstruct_volume_from_slices.py \
--filenames image_axial.nii.gz image_coronal.nii.gz image_sagittal.nii.gz
--filenames-masks mask_axial.nii.gz mask_coronal.nii.gz mask_sagittal.nii.gz \
--dir-input-mc res_registration_mvt 
--output reconstruction_niftymic
```


# Visualisation

The results can be visualize with **3Ddisplay.ipynb** and **3Ddisplay_realdata.ipynb**. You can select directly the pass to the joblib directory

# References 

[1] [EbnerWang2020] Ebner, M., Wang, G., Li, W., Aertsen, M., Patel, P. A., Aughwane, R., Melbourne, A., Doel, T., Dymarkowski, S., De Coppi, P., David, A. L., Deprest, J., Ourselin, S., Vercauteren, T. (2020). An automated framework for localization, segmentation and super-resolution reconstruction of fetal brain MRI. NeuroImage, 206, 116324.

[2] [EbnerWang2018] Ebner, M., Wang, G., Li, W., Aertsen, M., Patel, P. A., Melbourne, A., Doel, T., David, A. L., Deprest, J., Ourselin, S., Vercauteren, T. (2018). An Automated Localization, Segmentation and Reconstruction Framework for Fetal Brain MRI. In Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2018 (pp. 313–320). Springer.

[3] [Ebner2018]PANIST Ebner, M., Chung, K. K., Prados, F., Cardoso, M. J., Chard, D. T., Vercauteren, T., Ourselin, S. (2018). Volumetric reconstruction from printed films: Enabling 30 year longitudinal analysis in MR neuroimaging. NeuroImage, 165, 238–250
