#!/bin/bash 
images_file='../data_inter/export_chloe_29_07_2022/rawdata/'
simul_file='../simu_no_flip'


echo $images_file
images_list="$(find $images_file -name '*sub-0234*' -o -name '*sub-0361*' -o -name '*sub-0474*' -o -name '*sub-0365*' -o -name '*sub-0828*' -o -name '*sub-0910*'  -type d)"
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

output_path='../res/svort'
mkdir $output_path
#echo $images_list


#docker update --cpuset-cpus "0-24" wizardly_brattain

results='../res/svort'
mkdir $results
for file in ${images_list}
	do     
		#echo fileall2
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
		    
		     mkdir ${results}'/'${file1}
		     mkdir ${results}'/'${file1}'/'${file2}

		      
		     output_reconstruction_pipeline='/usr/local/NeSVoR/res/'${sub_file}
		    	     	    
		     #echo "${list_docker[@]}"
		     #echo "${mask_docker[@]}"

	     	 docker exec nesvor_contener mkdir '/usr/local/NeSVoR/res/'${file1}
		     docker exec nesvor_contener mkdir '/usr/local/NeSVoR/res/'${file1}'/'${file2}
             docker exec nesvor_contener mkdir '/usr/local/NeSVoR/res/'${file1}'/'${file2}

		     #docker cp ${output_registration}'_mvt' wizardly_brattain:'/app/NiftyMIC/ipta/'${file1}'/'${file2}'/all'${all}'/res_test_all'${all}'_mvt'

		     #dir_output_motion='NiftyMIC/ipta/'${sub_file}'/all'${all}'/res_test_all'${all}'_mvt'
		     #echo $dir_output_motion
			 docker exec nesvor_contener echo 'docker is running'
			 #docker exec nesvor_contener nesvor -h
		     
		     docker exec nesvor_contener nesvor reconstruct --input-stacks "${list_docker[@]}" --stack-masks "${mask_docker[@]}" --output-volume $output_reconstruction_pipeline'/volume.nii.gz'  --registration svort
	     	 #docker cp wizardly_brattain:'/app/NiftyMIC/ipta/'${sub_file}'/all'${all} ${results}'/'${sub_file}'/'
	     	    
		     
		}

for data in ${file_list}
	    do	

			 #echo 'igli is so cute'
		     echo $data
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
			mkdir ${results}'/simul_data/'${suffix}'/all'
					
			list_docker=()
			im=0
			for doc in $image
						do
									list_docker+=('/usr/local/NeSVoR/simu_no_flip/'${suffix}/'Lr'${prefix_image[$im]}$'Nifti_'${suffix_image[$im]}'.nii.gz')
									let im++
					
						done
			im=0
			mask_docker=()
			for doc in "${mask[@]}"
						do
									mask_docker+=('/usr/local/NeSVoR/simu_no_flip/'${suffix}'/brain_mask/Lr'${prefix_image[$im]}$'Nifti_'${suffix_image[$im]}'.nii.gz')
									let im++
						done
					
			output_simul_path=${results}'/simul_data/'${suffix}/'all'
			output_simul=${output_simul_path}'/res_test_all'
			mkdir output_simul
			output_res='/usr/local/NeSVoR/res/nesvor/'${suffix}
					
					
					#echo $list_docker
					#echo "herisson"
					#echo ${output_simul_path}
					#echo ${output_res}
		    echo "${mask[@]}"
			docker exec nesvor_contener mkdir '/usr/local/NeSVoR/res/nesvor'
			docker exec nesvor_contener mkdir '/usr/local/NeSVoR/res/nesvor/'${suffix}
			docker exec nesvor_contener nesvor reconstruct --input-stacks "${list_docker[@]}" --stack-masks "${mask_docker[@]}" --output-volume $output_res'/volume.nii.gz'  --output-slices $output_res --registration svort
  
	
					
		fi 
		
}

mkdir ${results}'/simul_data/'
for simul in $simul_file'/Grand3'
#${listr[@]}
do
		#simuldata=${simul_file}${simul}
		#echo $simul
		
		task "$simul" 
		
		#echo 'simul directory'
		#echo $simul &
		#pro=ps -A --no-headers | wc -l
		#echo $pro
		#docker exec e91b0f478887 rm -r 'NiftyMIC/ipta/'
		#docker exec e91b0f478887 mkdir 'NiftyMIC/ipta/'
	       	
done

	


