# Motion Correction of 2D MRI
## Simulated Data

To correct motion on simulated data use the script **main.py**.  This script creates movement on stacks of aligned 2D images, acquired in different orientations, then correct this movement using a motion correction algorithm. The script provides the evolution of the error (mse and dice), the evolution of the motion parameters (3 rotations and 3 translations), the evolution of the transformations between images coordinates and world coordinates and the error of registration (in mm). Those results can be visualised with the script **3Ddisplay.ipynb**

Example : 
```
python main.py \
--filenames image_axial.nii.gz image_coronal.nii.gz image_sagittal.nii.gz \
--filenames_masks mask_axial.nii.gz mask_coronal.nii.gz mask_sagittal.nii.gz \
--simulation_angle 3 \
--simulation_translation 3 \
--output test_simulation
```

where filenames and output are required. filenames are the stacks of 2D images. filenames_mask are the brain mask and should be in the same order than filenames. simulation_angle  and simulation_translation represent the movement created in the image. Movement is simulated by applying a rigid transformation to the image. On the above exemple, the angle parameters of the rigid transformation are random number between -3 and 3 and translation parameters are random number comprised between -3 and 3.

## Real Data

To correct motion on real data use the script **main_realdata.py**. This script correct movement in stacks of 2D images acquiered in different orientation. It provides the evolution of the error (mse and dice), the evolution of the motion parameters (3 rotations and 3 translations) and the evolution of the transformations between images coordinates and world coordinates. Those results can be visualised with the script **3Ddisplay_realdata.ipynb**

Example : 
```
python main.py \
--filenames image_axial.nii.gz image_coronal.nii.gz image_sagittal.nii.gz \
--filenames_masks mask_axial.nii.gz mask_coronal.nii.gz mask_sagittal.nii.gz \
--output test_simulation
```

where filenames and output are required. filenames are the stacks of 2D images. filenames_mask are the brain mask and should be in the same order than filenames.

# Visualisation
The results can be visualize with **3Ddisplay.ipynb** and **3Ddisplay_realdata.ipynb**. You can select directly the pass to the joblib directory
