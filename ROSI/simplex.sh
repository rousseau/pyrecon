#!/bin/bash 
images_file='../data_inter/export_chloe_29_07_2022/rawdata/'
simul_file='../simu'


echo $images_file
images_list="$(find $images_file -name '*sub-0058*' -type d)"
images_simul="$(find $simul_file -type d )"
mask="$(find $simul_file -type d -name '*brain_mask*')"
#echo $images_simul
#echo $mask

listr=($(comm -3 <(printf "%s\n" "${images_simul[@]}" | sort) <(printf "%s\n" "${mask[@]}" | sort) | sort -n))
echo $listr
#for sim in ${lstr}
#do
#	sim_list+=(${sim#*${simul_file}'/'})
#done

output_path='../res/simplex'
mkdir $output_path
#echo $images_list

docker start wizardly_brattain
docker update --cpuset-cpus "0-12" wizardly_brattain


for test in "test3" #"test2" "test3" "test4"
do		
	if [ "$test" = "test1" ]
	then
		simplex=1
	fi
	if [ "$test" = "test2" ]
	then
		simplex=2
	fi
	if [ "$test" = "test3" ]
	then 
		simplex=4
	fi
	if [ "$test" = "test4" ]
	then 
		simplex=6
	fi


	results='../res/simplex/value'${simplex}
	mkdir $results
	for file in ${images_list}
		do     
		#echo file
		#echo $file
		file_list="$(find $file -name '*ses*' -type d)"
		fonction(){

			 data=$1

			 list="$(find $data -name '*.nii.gz' -type f)"

		     mask_file='../data_inter/export_chloe_29_07_2022/derivatives/brain_masks/'
	         sub_file=${data#"$images_file"}

		     list_docker=()
		     mask_docker=()
                  
		     mask=${mask_file}${sub_file}
		     
		     list_mask="$(find $mask -name '*.nii.gz' -type f)"

                     for doc in $list
                     do
                             list_docker+=('NiftyMIC/code/'$doc)
                     done

		     for doc in $list_mask
		     do
			     mask_docker+=('NiftyMIC/code/'$doc)

	             done
		      
		
		     file1=${sub_file%"/"*}
		    
		     file2=${sub_file#*"/"}
		    
		     mkdir ${results}'/'${file1}
		     mkdir ${results}'/'${file1}'/'${file2}
		     output_registration=${results}'/'${sub_file}'/res_test_simplex'${simplex}
		      
		     
		     #echo "vive les herissons"
		     python ROSI/main_realdata.py --filenames $list  --filenames_masks $list_mask  --output $output_registration  --ablation no_multistart no_dice --hyperparameters $simplex 0.3 2000 2 1 0 
		     #echo "les herissons n ont pas de cerveau"

		  
		     output_reconstruction_inter=${output_path}'/'${sub_file}'/intersection_test.nii.gz'
		     output_reconstruction_pipeline='NiftyMIC/ipta/'${sub_file}'/simplex'${simplex}
		    
		     dir_output=${output_path}'/'${sub_file}'/pipeline_simplex'${simplex}'/recon_subject_space/motion_correction'
		     #echo $output_registration
	     	    
		     #echo "${list_docker[@]}"
		     #echo "${mask_docker[@]}"

	     	 #docker exec wizardly_brattain mkdir 'NiftyMIC/ipta/'${file1}
		     #docker exec wizardly_brattain mkdir 'NiftyMIC/ipta/'${file1}'/'${file2}
             #docker exec wizardly_brattain mkdir 'NiftyMIC/ipta/'${file1}'/'${file2}'/simplex'${simplex}

		     docker cp ${output_registration}'_mvt' wizardly_brattain:'/app/NiftyMIC/ipta/'${file1}'/'${file2}'/simplex'${simplex}'/res_test_simplex'${simplex}'_mvt'

		     dir_output_motion='/app/NiftyMIC/ipta/'${sub_file}'/simplex'${simplex}'/res_test_simplex'${simplex}'_mvt'
		     echo $dir_output_motion
		                   
		     docker exec wizardly_brattain python NiftyMIC/niftymic_run_reconstruction_pipeline_slices.py --filenames "${list_docker[@]}" --filenames-masks "${mask_docker[@]}" --dir-output $output_reconstruction_pipeline  --dir-input-mc $dir_output_motion 
	     	 docker cp wizardly_brattain:'/app/NiftyMIC/ipta/'${sub_file}'/simplex'${simplex} ${results}'/'${sub_file}'/'
	     	    
		     
			}

		for data in ${file_list}
	     	do	

			 #echo 'igli is so cute'
		     #echo $data
		     fonction "$data" 

		done

	done

	task (){
		
		simul=$1

		suffix=${simul#*${simul_file}'/'}
		
		if  [ "$suffix" = "Petit1" ];
		then
			
			#echo ${suffix}

			simul_data=${suffix}
			image="$(find $simul'/' -name 'Lr*Nifti*nii.gz' -not -name '*nomvt*' -not -name '*gradient*' -not -path "*/brain_mask/*" -type f)"
		
			#echo $image
			images_name=()
			for im in ${image}
			do
			images_name+=(${im#${simul}'/'})
			done
			
			#echo $images_name
			
			suffix_image=()
			for im in "${images_name[@]}"
			do
				suf=(${im#*'_'})
				suffix_image+=(${suf%*'.nii.gz'})
			done

			#echo $suffix_image
			
			prefix_image=()
			for im in "${images_name[@]}"
			do
			prefix_image1=${im#'Lr'*}
			prefix_image+=(${prefix_image1%'Nifti'*})
			done
			
			#echo $prefix_image
			
			i=0
			mask=()
			nomvt=()
			for im in ${image}
			do
				mask+=(${simul}'/brain_mask/Lr'${prefix_image[$i]}$'Nifti_'${suffix_image[$i]}'.nii.gz')
				nomvt+=(${simul}'/Lr'${prefix_image[$i]}$'Nifti_nomvt.nii.gz')
				let "i++"
			done

			#echo $nomvt

			transfo=()
			i=0
			for im in ${image}
			do
				transfo+=(${simul}'/transfo'${prefix_image[$i]}'_'${suffix_image[$i]}'.npy')
				let "i++"
			done
			#echo $transfo

			mkdir ${results}'/simul_data/'${suffix}
			mkdir ${results}'/simul_data/'${suffix}'/simplex'${simplex}
			
			list_docker=()
			im=0
			for doc in $image
						do
							list_docker+=('NiftyMIC/simu/'${suffix}/'Lr'${prefix_image[$im]}$'Nifti_'${suffix_image[$im]}'.nii.gz')
							let im++
			
					done
			im=0
			mask_docker=()
					for doc in "${mask[@]}"
						do
							mask_docker+=('NiftyMIC/simu/brain_mask/'${suffix}'/Lr'${prefix_image[$im]}$'Nifti_'${suffix_image[$im]}'.nii.gz')
							let im++
						done
			
			output_simul_path=${results}'/simul_data/'${suffix}/'simplex'${simplex}
			output_simul=${output_simul_path}'/res_test_simplex'${simplex}
			ouptut_motion_correction='../../res/simplex/simul_data/'${suffix}/'simplex'${simplex}/res_test_simplex'${simplex}'
			mkdir $output_simul
			output_res='NiftyMIC/ipta/'${suffix}'/simplex'${simplex}
			
			
			#echo $list_docker
			#echo "herisson"
			#echo ${output_simul_path}
			#echo ${output_res}
					
			python ROSI/main.py --filenames ${image}  --filenames_masks "${mask[@]}" --nomvt "${nomvt[@]}" --simulation "${transfo[@]}" --output ${output_simul}  --ablation no_multistart no_dice --hyperparameters $simplex 0.3 2000 2 1 0 
			#docker exec e91b0f478887 mkdir 'NiftyMIC/ipta/'${suffix} 
			#docker exec e91b0f478887 mkdir 'NiftyMIC/ipta/'${suffix}'/simplex'${simplex}
			#docker cp ${output_simul}'_mvt'  e91b0f478887:'/app/NiftyMIC/ipta/'${suffix}'/simplex'${simplex}'/res_test_simplex'${simplex}'_mvt' 
			dir_output_motion='/app/NiftyMIC/ipta/'${suffix}'/simplex'${simplex}'/res_test_simplex'${simplex}'_mvt'
			#docker exec e91b0f478887 python NiftyMIC/niftymic_run_reconstruction_pipeline_slices.py --filenames "${list_docker[@]}" --filenames-masks "${mask_docker[@]}" --dir-output $output_res  --dir-input-mc $dir_output_motion  
			#docker cp e91b0f478887:'/app/NiftyMIC/ipta/'${suffix}'/simplex'${simplex} ${results}'/simul_data/'${suffix}'/simplex'${simplex}
			
		fi 
	}

	mkdir ${results}'/simul_data/'
	for simul in ${listr[@]}
	do
		#simuldata=${simul_file}${simul}
		#task "$simul" 
		echo 'simul directory'
		#echo $simul
		#docker exec e91b0f478887 rm -r 'NiftyMIC/ipta/'
		#docker exec e91b0f478887 mkdir 'NiftyMIC/ipta/'
	       	
	done

	

done 