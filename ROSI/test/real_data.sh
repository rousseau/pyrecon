#!/bin/bash 
images_file='../data_inter/export_chloe_29_07_2022/rawdata/'
echo $images_file
images_list="$(find $images_file -name '*sub*' -type d)"
output_path='../res/omega_end'
mkdir $output_path
echo $images_list

docker start wizardly_brattain
docker update --cpuset-cpus "0-12" wizardly_brattain

#taskset --cpu-list 17
for test in "test1" #"test2" #"test3"
do		
	if [ "$test" = "test1" ]
	then
		omega=1
	fi
	if [ "$test" = "test2" ]
	then
		omega=0.01
	fi
	if [ "$test" = "test3" ]
	then 
		omega=6
	fi

	#echo "omega "${omega}
	#echo "ablation"${ablation}
	#echo "name"${name}
	
	results='../res/omega_end/value'${omega}
	mkdir $results
	for file in ${images_list}
		do     
		echo file
		echo $file
		file_list="$(find $file -name '*ses*' -type d)"
		fonction(){

			 data=$1

			 list="$(find $data -name '*.nii.gz' -type f)"

		     mask_file='../data_inter/export_chloe_29_07_2022/derivatives/brain_masks/'
	             sub_file=${data#"$images_file"}

		     list_docker=()
		     mask_docker=()
                  
		     mask=${mask_file}${sub_file}


		     #echo $mask
		     list_mask="$(find $mask -name '*.nii.gz' -type f)"

                     for doc in $list
                     do
                             list_docker+=('NiftyMIC/code/'$doc)
                     done

		     for doc in $list_mask
		     do
			     mask_docker+=('NiftyMIC/code/'$doc)

	             done
		      
		     #echo images
	             #echo $list
		     #echo mask
		     #echo $list_mask

		     
		     file1=${sub_file%"/"*}
		     #echo $file1
		     file2=${sub_file#*"/"}
		     #echo $file2
		     mkdir ${results}'/'${file1}
		     mkdir ${results}'/'${file1}'/'${file2}
		     output_registration=${results}'/'${sub_file}'/res_test_omega'${omega}
		     #echo $output_registration
		     
		     
		     #python ../niftymic_segment_fetal_brains.py --filenames $list --filenames-masks $list_mask
		     echo "vive les herissons"
		     python main_realdata.py --filenames $list  --filenames_masks $list_mask  --output $output_registration  --ablation no_multistart no_dice --hyperparameters 4 0.25 2000 2 $omega 4 
		     echo "les herissons n ont pas de cerveau"

		     #echo $dir_output

		     output_reconstruction_inter=${output_path}'/'${sub_file}'/intersection_test.nii.gz'
		     output_reconstruction_pipeline='NiftyMIC/ipta/'${sub_file}'/omega'${omega}
		     #echo $output_reconstruction_inter
		     dir_output=${output_path}'/'${sub_file}'/pipeline_omega'${omega}'/recon_subject_space/motion_correction'
		     echo $output_registration
	     	    
		     echo "${list_docker[@]}"
		     echo "${mask_docker[@]}"

		     #docker start wizardly_brattain

	     	 docker exec wizardly_brattain mkdir 'NiftyMIC/ipta/'${file1}
		     docker exec wizardly_brattain mkdir 'NiftyMIC/ipta/'${file1}'/'${file2}
             docker exec wizardly_brattain mkdir 'NiftyMIC/ipta/'${file1}'/'${file2}'/omega'${omega}


		     docker cp ${output_registration}'_mvt' wizardly_brattain:'/app/NiftyMIC/ipta/'${file1}'/'${file2}'/omega'${omega}'/res_test_omega'${omega}'_mvt'

		     dir_output_motion='/app/NiftyMIC/ipta/'${sub_file}'/omega'${omega}'/res_test_omega'${omega}'_mvt'
		     echo $dir_output_motion
		     
                     
		     docker exec wizardly_brattain python NiftyMIC/niftymic_run_reconstruction_pipeline_slices.py --filenames "${list_docker[@]}" --filenames-masks "${mask_docker[@]}" --dir-output $output_reconstruction_pipeline  --dir-input-mc $dir_output_motion 
	     	 docker cp wizardly_brattain:'/app/NiftyMIC/ipta/'${sub_file}'/omega'${omega} ${results}'/'${sub_file}'/'
	     	    #docker stop wizardly_brattain
		     
		     #output_reconstruction_pipeline=${output_path}'/'${sub_file}'/pipeline'


		     #python ../niftymic_reconstruct_volume_from_slices.py --filenames $list  --filenames-masks $list_mask --output $output_reconstruction_inter --dir-input-mc $dir_output  --use-masks-srr 1 &
		     #python ../niftymic_run_reconstruction_pipeline.py --filenames $list --filenames-masks $list_mask --dir-output $dir_output --dir-input-mc $dir_output_motion
			
		     output_reconstruction_ebner=${output_path}'/'${sub_file}'/ebner.nii.gz'
			}
		for data in ${file_list}
	     	do	
		     #echo data
		     echo $data
		     fonction "$data" &

		     #echo $output_reconstruction_ebner
		     #python ../niftymic_reconstruct_volume.py --filenames $list  --filenames-mask $list_mask  --output $output_reconstruction_ebner --use-masks-srr 1


		done

	done

	task (){
		#echo $simul
		simul=$1
		simul_path='../data_inter/test/'
		simul_data=${simul_path}${simul}
		image="$(find $simul_data'/data' -name 'Lr*Nifti*nii.gz' -not -name '*nomvt*' -type f)"
		images_name=()
		for im in ${image}
		do
		images_name+=(${im#${simul_data}'/data/'*})
		done
	#	
		suffix_image=()
		for im in "${images_name[@]}"
		do
			suf=(${im#*'_'})
			suffix_image+=(${suf%*'.nii.gz'})
		done

		prefix_image=()
		for im in "${images_name[@]}"
		do
		prefix_image1=${im#'Lr'*}
		prefix_image+=(${prefix_image1%'Nifti'*})
		done
		

		#prefix_image1=${images_name%'Lr'*}
		#prefix_image2=${prefix_image1#*'Nifti'}
		i=0
		mask=()
		nomvt=()
		for im in ${image}
		do
			mask+=(${simul_data}'/brain_mask/Lr'${prefix_image[$i]}$'Nifti_'${suffix_image[$i]}'.nii.gz')
			nomvt+=(${simul_data}'/Lr'${prefix_image[$i]}$'Nifti_nomvt.nii.gz')
			let "i++"
		done

		transfo=()
		i=0
		for im in ${image}
		do
			transfo+=(${simul_data}'/transfo'${prefix_image[$i]}'_'${suffix_image[$i]}'.npy')
			let "i++"
		done
		echo ${transfo[@]}
	#	echo "herisson"
	#	echo $mask
	#	echo $nomvt

		mkdir ${results}'/simul_data/'${simul}
		mkdir ${results}'/simul_data/'${simul}'/omega'${omega}
		
		list_docker=()
		for doc in $image
                     do
                             list_docker+=('NiftyMIC/code/'$doc)
                     done
		mask_docker=()
                for doc in "${mask[@]}"
                     do
                             mask_docker+=('NiftyMIC/code/'$doc)

                     done


		output_simul_path=${results}'/simul_data/'${simul}/'omega'${omega}
		output_simul=${output_simul_path}'/res_test_omega'${omega}
		output_res='NiftyMIC/ipta/'${simul}'/omega'${omega}
		echo "herisson"
		echo ${output_simul}
                
		python main_newsimul.py --filenames ${image}  --filenames_masks "${mask[@]}" --nomvt "${nomvt[@]}" --simulation "${transfo[@]}" --output ${output_simul}  --ablation no_multistart dice --hyperparameters 4 0.25 2000 2 $omega 4 
        docker exec e91b0f478887 mkdir 'NiftyMIC/ipta/'${simul} 
		docker exec e91b0f478887 mkdir 'NiftyMIC/ipta/'${simul}'/omega'${omega}
		docker cp ${output_simul}'_mvt' e91b0f478887:'/app/NiftyMIC/ipta/'${simul}'/omega'${omega}'/res_test_omega'${omega}'_mvt' 
		dir_output_motion='/app/NiftyMIC/ipta/'${simul}'/omega'${omega}'/res_test_omega'${omega}'_mvt'
		docker exec e91b0f478887 python NiftyMIC/niftymic_run_reconstruction_pipeline_slices.py --filenames "${list_docker[@]}" --filenames-masks "${mask_docker[@]}" --dir-output $output_res  --dir-input-mc $dir_output_motion  
		docker cp e91b0f478887:'/app/NiftyMIC/ipta/'${simul}'/omega'${omega} ${results}'/simul_data/'${simul}'/omega'${omega} 
	}

	mkdir ${results}'/simul_data/'
	for simul in "Petit1" "Moyen2" "Grand3"
	do
	
		#task "$simul" &
		echo $simul
	       	
	done

done 
