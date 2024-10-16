#!/bin/bash 
images_file='/home/aorus-users/Chloe/data_inter/export_chloe_29_07_2022/rawdata/'

echo $images_file
images_list="$(find $images_file -name  '*sub*' -type d)"
images_simul="$(find $simul_file -type d )"
echo $images_simul
mask="$(find $simul_file -type d -name '*brain_mask*')"
#echo $images_simul
#echo $mask

listr=($(comm -3 <(printf "%s\n" "${images_simul[@]}" | sort) <(printf "%s\n" "${mask[@]}" | sort) | sort -n))
echo $listr


results='/home/aorus-users/Chloe/svort_rosi_nesvor'
mkdir $results
for file in ${images_list}
	do     
		#echo filemulti/final_multi_start2
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
			 echo "mask"
			 echo $mask
		     
		     list_mask="$(find $mask -name '*.nii.gz' -type f)"
			 echo "list_mask"
			 echo $list_mask

                     for doc in $list
                     do
					 		 d=${doc//'/home/aorus-users/Chloe/data_inter/'}
							 echo $d
                             list_docker+=('/usr/local/NeSVoR/data/'$d)
							 echo "list_docker"
							 echo $list_docker
                     done

		     for doc in $list_mask
		     do
			 	 d=${doc//'/home/aorus-users/Chloe/data_inter/'}
				 echo $doc
			     mask_docker+=('/usr/local/NeSVoR/data/'$d)
				 echo $mask_docker

	             done
		      
		
		     file1=${sub_file%"/"*}
		    
		     file2=${sub_file#*"/"}
		    
		     mkdir ${results}'/'${file1}
		     mkdir ${results}'/'${file1}'/'${file2}
		    
	     	    
		     echo "${list_docker[@]}"
		     echo "${mask_docker[@]}"

			 docker exec nesvor_contener mkdir '/usr/local/NeSVoR/res_alone'
	     	 docker exec nesvor_contener mkdir '/usr/local/NeSVoR/res_alone/'${file1}
		     docker exec nesvor_contener mkdir '/usr/local/NeSVoR/res_alone/'${file1}'/'${file2}

			 
			 output_rosi='/home/aorus-users/Chloe/rosi_nesvor/'${file1}'/'${file2}'/rosi'
			 mkdir $output_rosi
			 python run_registration.py --filenames ${list} --filenames_mask ${list_mask} --output ${output_rosi} --no_multistart 1
			 output_slices='/home/aorus-users/Chloe/rosi_nesvor/'${file1}'/'${file2}'/rosi/slices'
			 mkdir '/home/aorus-users/Chloe/rosi_nesvor/'${file1}'/'${file2}'/rosi'
			 mkdir $output_slices
			 output_slices_masks='/home/aorus-users/Chloe/rosi_nesvor/'${file1}'/'${file2}'/rosi/masks'
			 mkdir $output_slices_masks
			 
			 python convert_to_svort.py  --input_stacks ${list} --input_mask ${list_mask} --output $output_slices  --results ${output_rosi}'/res.joblib.gz'
			
			 input_rosi_docker='/usr/local/NeSVoR/res_alone/'${file1}'/'${file2}'/rosi/slices'
			 docker exec nesvor_contener mkdir '/usr/local/NeSVoR/res_alone/'${file1}'/'${file2}'/rosi/'
			 docker exec nesvor_contener mkdir $input_rosi_docker
			 output_nesvor_docker='/usr/local/NeSVoR/res_alone/'${file1}'/'${file2}'/nesvor'
			 docker exec nesvor_contener mkdir $output_nesvor_docker
			 output_nesvor_image='/usr/local/NeSVoR/res_alone/'${file1}'/'${file2}'/nesvor/res.nii'
			 docker cp ${output_slices} nesvor_contener:'/usr/local/NeSVoR/res_alone/'${file1}'/'${file2}'/rosi/'
			 docker exec nesvor_contener nesvor reconstruct --input-slices ${input_rosi_docker} --output-volume ${output_nesvor_image} --registration none  --no-transformation-optimization  --output-resolution 0.5  --inference-batch-size 255  --n-inference-samples 128 

		}

for data in ${file_list}
	    do	

		     #echo $data
		     fonction "$data" 

		done

done
