#!/bin/bash 
images_file='../data_inter/export_chloe_29_07_2022/rawdata/'
simul_file='../simu_no_flip'


echo $images_file
images_list="$(find $images_file -name '*sub-0828*' -type d)"
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

output_path='../res/check_code'
mkdir $output_path
#echo $images_list

docker start wizardly_brattain
#docker update --cpuset-cpus "0-24" wizardly_brattain

results='../res/check_code/value'
mkdir $results
for file in ${images_list}
	do     
		#echo filelamb202
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
                             list_docker+=('NiftyMIC/code_copie/'$doc)
                     done

		     for doc in $list_mask
		     do
			     mask_docker+=('NiftyMIC/code_copie/'$doc)

	             done
		      
		
		     file1=${sub_file%"/"*}
		    
		     file2=${sub_file#*"/"}
		    
		     mkdir ${results}'/'${file1}
		     mkdir ${results}'/'${file1}'/'${file2}
		     output_registration=${results}'/'${sub_file}'/res_test_lamb0'
			 echo $output_registration

		
		     #
			 python ./ROSI/run_registration.py --filenames $list  --filenames_masks $list_mask  --output $output_registration  --no_multistart 0
			 #--ablation no_multistart dice Nelder-Mead --hyperparameters 4 0.25 2000 2 0 0 --classifier 'ROSI/my_model_mse_inter_std_intensity_mask_proportion_dice.pickle' 
		

		     #output_reconstruction_inter=${output_path}'/'${sub_file}'/intersection_test.nii.gz'
		     #output_reconstruction_pipeline='NiftyMIC/ipta/'${sub_file}'/lamb20'
		    
		     #dir_output=${output_path}'/'${sub_file}'/pipeline_lamb20/recon_subject_space/motion_correction'
		     #echo $output_registration
	     	    
		     #echo "${list_docker[@]}"
		     #echo "${mask_docker[@]}"
			 #docker exec wizardly_brattain mkdir 'NiftyMIC/ipta/'
	     	 #docker exec wizardly_brattain mkdir 'NiftyMIC/ipta/'${file1}
		     #docker exec wizardly_brattain mkdir 'NiftyMIC/ipta/'${file1}'/'${file2}
             #docker exec wizardly_brattain mkdir 'NiftyMIC/ipta/'${file1}'/'${file2}'/lamb20'${lamb20}

		     #docker cp ${output_registration}'_mvt' wizardly_brattain:'/app/NiftyMIC/ipta/'${file1}'/'${file2}'/lamb20/res_test_lamb20_mvt'

		     #dir_output_motion='NiftyMIC/ipta/'${sub_file}'/lamb20'${lamb20}'/res_test_lamb20'${lamb20}'_mvt'
		     #echo $dir_output_motion
		                   
		     #docker exec wizardly_brattain python NiftyMIC/niftymic_run_reconstruction_pipeline_slices.py --filenames "${list_docker[@]}" --filenames-masks "${mask_docker[@]}" --dir-output $output_reconstruction_pipeline  --dir-input-mc $dir_output_motion 
	     	 #docker cp wizardly_brattain:'/app/NiftyMIC/ipta/'${sub_file}'/lamb20'${lamb20} ${results}'/'${sub_file}'/'
	     	    
		     
		}

for data in ${file_list}
	    do	

		     #echo $data
		     fonction "$data" 

		done

done

task (){
		
		simul=$1

		suffix=${simul#*${simul_file}'/'}
		echo $suffix
		
		if [ "$suffix" != "../simu"  ] 
		#!= "../simu" ];
		
			#
		then
			
			echo ${suffix}

			simul_data=${suffix}
			image="$(find $simul'/' -name 'Lr*Nifti*nii.gz'  -not -name '*nomvt*' -not -name '*gradient*' -not -path "*/brain_mask/*" -type f)"
				
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
				#mask+=(${simul}'/'${prefix_mask[$i]}'Mask_nomvt.nii.gz')
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
			mkdir ${results}'/simul_data/'${suffix}'/no_flip'
					
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
									mask_docker+=('NiftyMIC/simu/'${suffix}'/brain_mask/Lr'${prefix_image[$im]}$'Nifti_'${suffix_image[$im]}'.nii.gz')
									let im++
						done
					
			output_simul_path=${results}'/simul_data/'${suffix}/'no_flip'
			output_simul=${output_simul_path}'/res_test_lamb0'
			mkdir output_simul
			#output_res='NiftyMIC/ipta/'${suffix}'/lamb20'
					
					
					#echo $list_docker
					#echo "herisson"
					#echo ${output_simul_path}
					#echo ${output_res}
		    echo "${mask[@]}"
			echo "${image[@]}"
			python ROSI/main.py --filenames $image  --filenames_masks ${mask[@]}  --output $output_simul --simulation ${transfo[@]} --nomvt ${nomvt[@]}  --ablation no_multistart dice Nelder-Mead --hyperparameters 4 0.25 2000 0.25 2 0 
			#docker exec e91b0f478887 mkdir 'NiftyMIC/ipta/'${suffix} 
			#docker exec e91b0f478887 mkdir 'NiftyMIC/ipta/'${suffix}'/lamb20'
			#docker cp ${output_simul}'_mvt'  e91b0f478887:'app/NiftyMIC/ipta/'${suffix}'/lamb20/res_test_lamb20_mvt' 
			#dir_output_motion='NiftyMIC/ipta/'${suffix}'/lamb20/res_test_lamb20_mvt'
			#docker exec e91b0f478887 python NiftyMIC/niftymic_run_reconstruction_pipeline_slices.py --filenames "${list_docker[@]}" --filenames-masks "${mask_docker[@]}" --dir-output $output_res  --dir-input-mc $dir_output_motion  
			#docker cp e91b0f478887:'/app/NiftyMIC/ipta/'${suffix}'/lamb20' ${results}'/simul_data/'${suffix}'/lamb20'
					
		fi 
		
}

mkdir ${results}'/simul_data/'
for simul in ${listr[@]}
do
		#simuldata=${simul_file}${simul}
		echo $simul
		
		#task "$simul" &
		
		#echo 'simul directory'
		#echo $simul &
		#pro=ps -A --no-headers | wc -l
		#echo $pro
		#docker exec e91b0f478887 rm -r 'NiftyMIC/ipta/'
		#docker exec e91b0f478887 mkdir 'NiftyMIC/ipta/'
	       	
done

	


