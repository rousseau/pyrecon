#!/bin/bash 
images_file='../data_inter/export_chloe_29_07_2022/rawdata/'
simul_file='../simu_no_flip'


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

#output_path='../res/Ebner'
#mkdir $output_path
#echo $images_list

docker start wizardly_brattain
#docker update --cpuset-cpus "0-24" wizardly_brattain

results='home/results'
mkdir $results
for file in ${images_list}
	do     
		#echo fileEbner2
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
                             list_docker+=('home/data/'$doc)
                     done

		     for doc in $list_mask
		     do
			     mask_docker+=('home/data/'$doc)

	             done
		      
		
		     file1=${sub_file%"/"*}
		    
		     file2=${sub_file#*"/"}
		    
		     mkdir ${results}'/'${file1}
		     mkdir ${results}'/'${file1}'/'${file2}
		     #output_registration=${results}'/'${sub_file}'/ebner_no_rejection'
		      
	

		  
		     #output_reconstruction_inter=${output_path}'/'${sub_file}'/intersection_test.nii.gz'
		     #output_reconstruction_pipeline='NiftyMIC/ipta/'${sub_file}'/Ebner'
		    
		     #dir_output=${output_path}'/'${sub_file}'/pipeline_Ebner/recon_subject_space/motion_correction'
		     #echo $output_registration
	     	    
		     echo "${list_docker[@]}"
		     echo "${mask_docker[@]}"
			 #docker exec wizardly_brattain mkdir 'NiftyMIC/ipta/'
	     	 #docker exec wizardly_brattain mkdir 'NiftyMIC/ipta/'${file1}
		     #docker exec wizardly_brattain mkdir 'NiftyMIC/ipta/'${file1}'/'${file2}
             #docker exec wizardly_brattain mkdir 'NiftyMIC/ipta/'${file1}'/'${file2}'/Ebner'

		     #echo $dir_output_motion
		                   
		     #docker exec wizardly_brattain python NiftyMIC/niftymic_run_reconstruction_pipeline.py --filenames "${list_docker[@]}" --filenames-masks "${mask_docker[@]}" --dir-output $output_reconstruction_pipeline --outlier-rejection 0
	     	 #docker cp wizardly_brattain:'/app/NiftyMIC/ipta/'${sub_file}'/Ebner' ${results}'/'${sub_file}'/'
	     	    
		     
		}

for data in ${file_list}
	    do	

		     #echo $data
		     fonction "$data" 

		done

done



	


