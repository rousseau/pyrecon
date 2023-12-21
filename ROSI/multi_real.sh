#!/bin/bash 
images_file='../data_inter/export_chloe_29_07_2022/rawdata/'
simul_file='../simu'


echo $images_file
images_list="$(find $images_file -name '*sub*'  -type d)"
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


#echo $images_list

docker start wizardly_brattain
#docker update --cpuset-cpus "0-24" wizardly_brattain


for multi_start in "Nelder-Mead"
#"LM" "BFGS" "Powell" "TNC" "CG" "Nelder-Mead"
do
	mkdir '../res/multi/'
	output_path='../res/multi/'${multi_start}
	mkdir $output_path
	results='../res/multi/'${multi_start}'/value'
	mkdir $results
	echo "opti"
	echo $multi_start

	task(){

			 data=$1
			 file_list="$(find $data -name '*ses*' -type d)"
			 list="$(find $file_list -name '*.nii.gz' -type f)"
			 echo $list

		     mask_file='../data_inter/export_chloe_29_07_2022/derivatives/brain_masks/'
	         sub_file="${file_list#*"rawdata/"}"
			 echo $sub_file

		     list_docker=()
		     mask_docker=()
                  
		     mask=${mask_file}${sub_file}
			 echo $mask
		     
		     list_mask="$(find $mask -name '*.nii.gz' -type f)"
			 echo $list_mask

                     for doc in $list
                     do
                             list_docker+=('NiftyMIC/code_copie/'$doc)
                     done

		     for doc in $list_mask
		     do
			     mask_docker+=('NiftyMIC/code_copie/'$doc)

	             done
		      
		
		     file1=${sub_file%"/"*}
		 	 echo $file1   

		     file2=${sub_file#*"/"}
			 echo $file2
		    
		     mkdir ${results}'/'${file1}
		     mkdir ${results}'/'${file1}'/'${file2}
		     

			 job="/mnt/Data/Chloe/res/omega/value0/"${sub_file}"/res_test_omega.joblib.gz"

						
						
			 output_simul_path=${results}'/'${sub_file}'/'
			 output_simul=${output_simul_path}'/res_test_'${multi_start}
			 output_registration=${output_simul_path}'/res_test_'${multi_start}'_mvt'
			 mkdir output_simul
			 output_res='NiftyMIC/ipta/'${sub_file}'/'${multi_start}
						
						
						#echo $list_docker
						#echo "herisson"
						#echo ${output_simul_path}
						#echo ${output_res}
								
			 #python ROSI/test_multi_start_real.py --filenames $job  --filenames_masks $list_mask  --output ${output_simul}  --ablation  $multi_start --hyperparameters 4 0.3 2000 0.25 0 0 
			

	     	 docker exec wizardly_brattain mkdir 'NiftyMIC/ipta/'${file1}
		     docker exec wizardly_brattain mkdir 'NiftyMIC/ipta/'${file1}'/'${file2}
             docker exec wizardly_brattain mkdir 'NiftyMIC/ipta/'${file1}'/'${file2}'/'${multi_start}


		     docker cp ${output_registration} wizardly_brattain:'/app/NiftyMIC/ipta/'${file1}'/'${file2}'/'${multi_start}'/res_test_'${multi_start}'_mvt'

		     dir_output_motion='NiftyMIC/ipta/'${sub_file}'/'${multi_start}'/res_test_'${multi_start}'_mvt'
		     echo $dir_output_motion

		                   
		     docker exec wizardly_brattain python NiftyMIC/niftymic_run_reconstruction_pipeline_slices.py --filenames "${list_docker[@]}" --filenames-masks "${mask_docker[@]}" --dir-output $output_res  --dir-input-mc $dir_output_motion 
	     	 docker cp wizardly_brattain:'/app/NiftyMIC/ipta/'${sub_file}'/'${multi_start} ${results}'/'${sub_file}'/'
						
			
	}


	for data in ${images_list}
	do
			echo $data &
			task $data &		
	done 

done