#bin/bash

for file in "sub-0258_ses-0305_t2_haste_512_pace_9.nii.gz" "sub-0258_ses-0305_t2_haste_512_pace_11.nii.gz" "sub-0258_ses-0305_t2_haste_512_pace_13.nii.gz"
do
	python ../niftymic_correct_bias_field.py  --filename ../data/export_chloe_29_07_2022/rawdata/sub-0258/ses-0305/${file} --filename-mask ../data/export_chloe_29_07_2022/derivatives/brain_masks/sub-0258/ses-0305/${file} --output ../data/export_chloe_29_07_2022/rawdata/sub-0258/ses-0305/${file}

done

python code/main_realdata.py --filenames ../data/export_chloe_29_07_2022/rawdata/sub-0258/ses-0305/sub-0258_ses-0305_t2_haste_512_pace_9.nii.gz ../data/export_chloe_29_07_2022/rawdata/sub-0258/ses-0305/sub-0258_ses-0305_t2_haste_512_pace_11.nii.gz ../data/export_chloe_29_07_2022/rawdata/sub-0258/ses-0305/sub-0258_ses-0305_t2_haste_512_pace_13.nii.gz  --filenames_masks ../data/export_chloe_29_07_2022/derivatives/brain_masks/sub-0258/ses-0305/sub-0258_ses-0305_t2_haste_512_pace_9.nii.gz ../data/export_chloe_29_07_2022/derivatives/brain_masks/sub-0258/ses-0305/sub-0258_ses-0305_t2_haste_512_pace_11.nii.gz ../data/export_chloe_29_07_2022/derivatives/brain_masks/sub-0258/ses-0305/sub-0258_ses-0305_t2_haste_512_pace_13.nii.gz --output ../res_article/res_sub-0258_ses-0305 --ablation no_multistart --hyperparameters 4 0.25 1e-10 2 0 4


python ../niftymic_reconstruct_volume_from_slices.py --filenames ../data/export_chloe_29_07_2022/rawdata/sub-0258/ses-0305/sub-0258_ses-0305_t2_haste_512_pace_9.nii.gz ../data/export_chloe_29_07_2022/rawdata/sub-0258/ses-0305/sub-0258_ses-0305_t2_haste_512_pace_11.nii.gz ../data/export_chloe_29_07_2022/rawdata/sub-0258/ses-0305/sub-0258_ses-0305_t2_haste_512_pace_13.nii.gz    --filenames-masks ../data/export_chloe_29_07_2022/derivatives/brain_masks/sub-0258/ses-0305/sub-0258_ses-0305_t2_haste_512_pace_9.nii.gz ../data/export_chloe_29_07_2022/derivatives/brain_masks/sub-0258/ses-0305/sub-0258_ses-0305_t2_haste_512_pace_11.nii.gz ../data/export_chloe_29_07_2022/derivatives/brain_masks/sub-0258/ses-0305/sub-0258_ses-0305_t2_haste_512_pace_13.nii.gz --output ../res/rec_ebner_res_sub-0258_ses-0305.nii.gz --dir-input-mc ../res/res_sub-0258_ses-0305_mvt --use-masks-srr 1

