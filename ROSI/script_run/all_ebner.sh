#!/bin/bash 
images_file='/home/aorus-users/Chloe/data_inter/export_chloe_29_07_2022/rawdata/'
simul_file='../simu_no_flip'


echo $images_file
images_list="$(find $images_file -name '*sub-0270*' -o -name '*sub-0363*'  -type d)"
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

output_path='/home/aorus-users/Chloe/res/Ebner'
mkdir $output_path
#echo $images_list

docker start wizardly_brattain
#docker update --cpuset-cpus "0-24" wizardly_brattain

results='/home/aorus-users/Chloe/res/Ebner/value'
mkdir $results
for file in ${images_list}
	do     
		#echo fileEbner2
		#echo $file
	file_list="$(find $file -name '*ses*' -type d)"
	fonction(){

			 data=$1

			 list="$(find $data -name '*.nii.gz' -type f)"

		     mask_file='/home/aorus-users/Chloe/data_inter/export_chloe_29_07_2022/derivatives/brain_masks/'
	         sub_file=${data#"$images_file"}

		     list_docker=()
		     mask_docker=()
                  
		     mask=${mask_file}${sub_file}
		     
		     list_mask="$(find $mask -name '*.nii.gz' -type f)"

                     for doc in $list
                     do
                             list_docker+=('NiftyMIC/export_chloe_29_07_2022/'$doc)
                     done

		     for doc in $list_mask
		     do
			     mask_docker+=('NiftyMIC/export_chloe_29_07_2022/'$doc)

	             done
		      
		
		     file1=${sub_file%"/"*}
		    
		     file2=${sub_file#*"/"}
		    
		     mkdir ${results}'/'${file1}
		     mkdir ${results}'/'${file1}'/'${file2}
		     output_registration=${results}'/'${sub_file}'/ebner_no_rejection'
		    
		     output_reconstruction_pipeline='NiftyMIC/ipta/'${sub_file}'/Ebner'
		    
		     dir_output=${output_path}'/'${sub_file}'/pipeline_Ebner/recon_subject_space/motion_correction'
		     #echo $output_registration
	     	    
		     #echo "${list_docker[@]}"
		     #echo "${mask_docker[@]}"
			 docker exec ebner mkdir 'NiftyMIC/ipta/'
	     	 docker exec ebner mkdir 'NiftyMIC/ipta/'${file1}
		     docker exec ebner mkdir 'NiftyMIC/ipta/'${file1}'/'${file2}
             docker exec ebner mkdir 'NiftyMIC/ipta/'${file1}'/'${file2}'/Ebner'

		     echo $dir_output_motion
		                   
		     docker exec ebner python NiftyMIC/niftymic_run_reconstruction_pipeline.py --filenames "${list_docker[@]}" --filenames-masks "${mask_docker[@]}" --dir-output $output_reconstruction_pipeline --outlier-rejection 0
	     	 #docker cp wizardly_brattain:'/app/NiftyMIC/ipta/'${sub_file}'/Ebner' ${results}'/'${sub_file}'/'
	     	    
		     
		}

for data in ${file_list}
	    do	

		     echo $data
		     fonction "$data" 

		done

done

task (){
		
		simul=$1

		suffix=${simul#*${simul_file}'/'}
		echo $suffix
		
		if [ "$suffix"!="../simu"  ] 
		
			#
		then
			
			echo ${suffix}
			echo $simul
			simul_data=${suffix}
			image="$(find $simul'/' -name 'Lr*Nifti*nii.gz'  -not -name '*nomvt*' -not -name '*gradient*' -not -path "*/brain_mask/*" -type f)"
				
			
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
			mkdir ${results}'/simul_data/'${suffix}'/Ebner'
					
			list_docker=()
			im=0
			for doc in $image
						do
									list_docker+=('NiftyMIC/simu_no_flip/'${suffix}/'Lr'${prefix_image[$im]}$'Nifti_'${suffix_image[$im]}'.nii.gz')
									let im++
					
						done
			im=0
			mask_docker=()
			for doc in "${mask[@]}"
						do
									mask_docker+=('NiftyMIC/simu_no_flip/'${suffix}'/brain_mask/Lr'${prefix_image[$im]}$'Nifti_'${suffix_image[$im]}'.nii.gz')
									let im++
						done
					
			output_simul_path=${results}'/simul_data/'${suffix}/'Ebner'
			output_simul=${output_simul_path}'/ebner_no_rejection'
			mkdir output_simul
			output_res='NiftyMIC/ipta/'${suffix}'/Ebner'
					
					
					#echo $list_docker
					#echo "herisson"
					#echo ${output_simul_path}
					#echo ${output_res}
		    echo "${mask[@]}"
			#python ROSI/main.py --filenames ${image}  --filenames_masks "${mask[@]}" --nomvt "${nomvt[@]}" --simulation "${transfo[@]}" --output ${output_simul}  --ablation multistart no_dice --hyperparameters 4 0.3 2000 2 $Ebner 0 --classifier 'ROSI/my_model_mse_inter_std_intensity_mask_proportion_dice.pickle' 
			docker exec ebner mkdir 'NiftyMIC/eb/'
			docker exec ebner mkdir 'NiftyMIC/eb/'${suffix} 
			docker exec ebner mkdir 'NiftyMIC/eb/'${suffix}'/Ebner'
		
			docker exec ebner python NiftyMIC/niftymic_run_reconstruction_pipeline.py --filenames "${list_docker[@]}" --filenames-masks "${mask_docker[@]}" --dir-output 'NiftyMIC/eb/'${suffix}'/Ebner' #--outlier-rejection 1
			#docker cp ebner:'/app/NiftyMIC/eb/'${suffix}'/Ebner' ${results}'/simul_data/'${suffix}'/Ebner_outliers'
					
		fi 
		
}

mkdir ${results}'/simul_data/'
for simul in "/Petit1"
#${listr[@]}
do
		simuldata=${simul_file}${simul}
		echo $simul
		
		#task "$simuldata" 
		
		#echo 'simul directory'
		#echo $simul &
		#pro=ps -A --no-headers | wc -l
		#echo $pro
		#docker exec e91b0f478887 rm -r 'NiftyMIC/ipta/'
		#docker exec e91b0f478887 mkdir 'NiftyMIC/ipta/'
	       	
done

	


