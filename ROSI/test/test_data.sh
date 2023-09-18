#!/bin/bash 
images_file='../data_inter/export_chloe_29_07_2022/rawdata/'
images_list="$(find $images_file  -name '*sub-0408*'  -type d)"
output_path='../res/omega_test'
mkdir $output_path
echo $image_list
docker start wizardly_brattain

fonction(){

	
				data=$1
				
				list="$(find $data -name '*.nii.gz' -type f)"
				mask_file='../data_inter/export_chloe_29_07_2022/derivatives/brain_masks/'
				sub_file=${data#"$images_file"}

				mask=${mask_file}${sub_file}
				echo $mask
				list_mask="$(find $mask -name '*.nii.gz' -type f)"
				
				echo images
				echo $list
				echo mask
				echo $list_mask

				image=()
				for im in $list
					do
						image+=(${im#"${data}"})
				#	echo $name_image
				#	mask_name=${mask}${name_image}
				#	python ../niftymic_correct_bias_field.py  --filename  ${image} --filename-mask ${mask_name}  --output ${image}

					done
				echo herssion
				echo "${image[@]}"

				output_registration=${output_path}'/'${sub_file}'/omega'${omega}'/res_test_omega'${omega}
				file1=${sub_file%"/"*}
				echo $file1
				file2=${sub_file#*"/"}
				echo $file2
				
				#if [ ! -d ${output_path}'/'${file1} ]
				#then
				mkdir ${output_path}'/'${file1}
				#fi
				
				#if [ ! -d ${output_path}'/'${file1}'/'${file2} ]
				#then
				mkdir ${output_path}'/'${file1}'/'${file2}
				#fi
				mkdir ${output_path}'/'${file1}'/'${file2}'/omega'${omega}
				echo ${output_path}'/'${file1}'/'${file2}'/omega'${omega}
				
				list_docker=()
				mask_docker=()
				for im in "${image[@]}"
				do
					echo ${im}
					list_docker+=('NiftyMIC/data_inter/export_chloe_29_07_2022/rawdata/'${sub_file}${im})
					mask_docker+=('NiftyMIC/data_inter/export_chloe_29_07_2022/derivatives/brain_masks/'${sub_file}${im})
				done
				#python ../niftymic_segment_fetal_brains.py --filenames $list --filenames-masks $list_mask --ablation 
				
				python main_realdata.py --filenames $list  --filenames_masks $list_mask  --output $output_registration  --ablation no_multistart --hyperparameters 4 0.25 1e-10 2 $omega 4 

				dir_output_motion=${output_registration}'_mvt'
				echo $dir_output
				output_reconstruction_inter=${output_path}'/'${sub_file}'/omega'${omega}
				#output_reconstruction_pipeline=${output_path}'/'${sub_file}'/pipeline.nii.gz'
				echo $output_reconstruction_inter
				dir_output=${output_path}'/'${sub_file}'/pipeline_intersection'

				echo "surcouf"
				echo "${list_docker[@]}"
				echo "${mask_docjer[@]}"

				docker exec wizardly_brattain mkdir 'NiftyMIC/ipta/'${file1}
				docker exec wizardly_brattain mkdir 'NiftyMIC/ipta/'${file1}'/'${file2}
				docker exec wizardly_brattain mkdir 'NiftyMIC/ipta/'${file1}'/'${file2}'/omega'${omega}

				docker cp ${dir_output_motion} wizardly_brattain:'/app/NiftyMIC/ipta/'${sub_file}'/omega'${omega}'/res_test_omega'${omega}'_mvt'
				docker exec wizardly_brattain python NiftyMIC/niftymic_run_reconstruction_pipeline_slices.py --filenames "${list_docker[@]}"  --filenames-masks "${mask_docker[@]}" --dir-output 'NiftyMIC/ipta/'${sub_file}'/omega'${omega} --dir-input-mc 'NiftyMIC/ipta/'${sub_file}'/omega'${omega}'/res_test_omega'${omega}'_mvt'  
				docker cp wizardly_brattain:'/app/NiftyMIC/ipta/'${sub_file}'/omega'${omega} $output_registration 
				#python ../niftymic_run_reconstruction_pipeline_slices.py --filenames $list --filenames-masks $list_mask --dir-output $dir_output --dir-input-mc $dir_output_motion &
				
				#output_reconstruction_pipeline=${output_path}'/'${sub_file}'/pipeline'	
				#output_reconstruction_ebner=${output_path}'/'${sub_file}'/ebner.nii.gz'
				#echo $output_reconstruction_ebner
			#python ../niftymic_run_reconstruction_pipeline.py --filenames $list  --filenames-mask $list_mask  --dir-output $output_reconstruction_pipeline &
}


for test in "test1" "test2" "test3" : 
do
	omega=0
	if [ $test="test1" ]
	then
		echo 2
		omega=2
		echo $omega
		for file in ${images_list}
		do     
			echo file
			echo $file
			file_list="$(find $file -name '*ses*' -type d)"
			for data in ${file_list}
			do
				fonction $data &
				echo "fonction"

			done
		done
	fi
	if [ $test="test2" ]
	then
		echo 4
		omega=4
		echo $omega
		for file in ${images_list}
		do     
			echo file
			echo $file
			file_list="$(find $file -name '*ses*' -type d)"
			for data in ${file_list}
			do
				fonction $data &
				echo "fonction"

			done
		done
	fi
	if [ $test="test3" ]
	then
		echo 6
		omega=6
		echo $omega
		for file in ${images_list}
		do     
			echo file
			echo $file
			file_list="$(find $file -name '*ses*' -type d)"
			for data in ${file_list}
			do
				fonction $data &
				echo "fonction"

			done
		done
	fi

	echo $omega
	echo $test

	
	done
done