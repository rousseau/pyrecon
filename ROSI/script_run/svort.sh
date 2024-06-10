#!/bin/bash 
images_file='../data_inter/export_chloe_29_07_2022/rawdata/'
simul_file='../simu_no_flip'


#echo $images_file
images_list="$(find $images_file -name '*sub-0828*'  -type d)"
#echo $images_list
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

output_path='/mnt/Data/res/rosi_svort/'
mkdir $output_path
#echo $images_list


#docker update --cpuset-cpus "0-24" wizardly_brattain

results='/mnt/Data/res/rosi_svort/'
echo $results
mkdir $results

	fonction(){

			 data=$1

			 echo $data
			 list="$(find $data -name '*.nii.gz' -type f)"

		     mask_file='../data_inter/export_chloe_29_07_2022/derivatives/brain_masks/'
	         sub_file=${data#"$images_file"}
			 echo $sub_file

		     list_docker=()
		     mask_docker=()
                  
		     mask=${mask_file}${sub_file}
		     
		     list_mask="$(find $mask -name '*.nii.gz' -type f)"

                     for doc in $list
                     do
					 		 im=${doc#*"/data_inter/"}
                             list_docker+=('/usr/local/NeSVoR/data/'$im)
                     done

		     for doc in $list_mask
		     do
			 	 ma=${doc#*"/data_inter/"}
			     mask_docker+=('/usr/local/NeSVoR/data/'$ma)

	             done
		      
		
		     file1=${sub_file%"/"*}
		    
		     file2=${sub_file#*"/"}

			 output_simul_path=${results}${sub_file}'/'
			 output_simul=${output_simul_path}'/svort_mvt'

		     mkdir ${results}'/'${file1}
		     mkdir ${results}'/'${file1}'/'${file2}
		    	     	    
			 job_res='/mnt/Data/res/lamb0/value/'${sub_file}'/res_test_lamb0.joblib.gz'
			 echo $list_mask
	   		 echo $list	  	 
			 python ROSI/convert_to_svort.py --input_stacks ${list}  --input_mask ${list_mask}  --output ${output_simul}  --results $job_res
	     	 #docker cp wizardly_brattain:'/app/NiftyMIC/ipta/'${sub_file}'/all'${all} ${results}'/'${sub_file}'/'
	     	    
		     
		}

for data in ${images_list}
	    do	

			file_list="$(find $data -name '*ses*' -type d)"

			for file in ${file_list}
				do
				echo 'igli is so cute'
		   		echo $file
		    	fonction "$file" 
				done

		done


task (){
		
		simul=$1

		suffix=${simul#*${simul_file}'/'}
		echo $simul
		
		if [ "$suffix"!="../simul_data" ];
		
			#"../simu"
		then
			
			echo ${suffix}

			simul_data=${suffix}
			image="$(find $simul'/' -name 'Lr*Nifti*nii.gz'  -not -name '*nomvt*' -not -name '*gradient*' -not -path "*/brain_mask/*" -type f)"

			echo "image"	
			echo $image
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

			echo ${suffix}
			mkdir ${results}'/'${suffix}
					
			list_docker=()
			im=0
			for doc in $image
						do
									list_docker+=('../simul_data/'${suffix}/'Lr'${prefix_image[$im]}$'Nifti_'${suffix_image[$im]}'.nii.gz')
									let im++
					
						done
			im=0
			mask_docker=()
			for doc in "${mask[@]}"
						do
									mask_docker+=('../simul_data/'${suffix}'/brain_mask/Lr'${prefix_image[$im]}$'Nifti_'${suffix_image[$im]}'.nii.gz')
									let im++
						done
					
			output_simul_path=${results}'/'${suffix}/
			output_simul=${output_simul_path}'/svort_mvt'
			mkdir output_simul
			output_res='NiftyMIC/ipta/'${suffix}'/all'
					
			job_res='../res/no_flip/value/simul_data/'${suffix}'/no_flip/res_test_lamb0.joblib.gz'
					#echo $list_docker
					#echo "herisson"
					#echo ${output_simul_path}
					#echo ${output_res}
		    echo "${mask[@]}"
			python ROSI/convert_to_svort.py --input_stacks "${nomvt[@]}"  --input_mask "${mask[@]}"  --output ${output_simul}  --results $job_res  
			#python ROSI/convert_to_svort.py --input_stacks $image --input_mask "${mask[@]}"  --output ${output_simul}  --results $job_res 
					
		fi 
		
}

mkdir ${results}'/simul_data/'
for simul in ${listr[@]}
do
		#simuldata=${simul_file}${simul}
		echo $simul
		
		#task "$simul" 
		
		#echo 'simul directory'
		#echo $simul &
		#pro=ps -A --no-headers | wc -l
		#echo $pro
		#docker exec e91b0f478887 rm -r 'NiftyMIC/ipta/'
		#docker exec e91b0f478887 mkdir 'NiftyMIC/ipta/'
	       	
done

	


