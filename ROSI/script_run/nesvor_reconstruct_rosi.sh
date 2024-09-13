#!/bin/bash 
images_file='../data_inter/export_chloe_29_07_2022/rawdata/'
simul_file='../simu'


echo $images_file
images_list="$(find $images_file -name '*sub-0182*' -type d)"
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

	     	dir_input="/usr/local/NeSVoR/"${sub_file}"/svort_mvt"
			dir_output='/usr/local/NeSVoR/nesvor_rosi_reconstruct/'${sub_file}
			docker exec nesvor_contener mkdir '/usr/local/NeSVoR/nesvor_rosi_reconstruct/'
			docker exec nesvor_contener mkdir '/usr/local/NeSVoR/nesvor_rosi_reconstruct/'${sub_file}
			echo 'youpi'
		    docker exec nesvor_contener nesvor reconstruct --input-slices $dir_input  --output-volume $dir_output'/volume_nobiais.nii' --output-slices $dir_output  --registration none --output-slices $dir_output --no-transformation-optimization --no-pixel-variance  --no-slice-variance
	     	#docker cp wizardly_brattain:'/app/NiftyMIC/ipta/'${sub_file}'/all'${all} ${results}'/'${sub_file}'/'
	     	    
		     
		}

for data in ${file_list}
	    do	

			file_list="$(find $data -name '*ses*' -type d)"

			for file in ${file_list}
				do
		     	echo $data
		     	fonction "$data"
				done 

		done

done

task (){
		
		simul=$1

		suffix=${simul#*${simul_file}'/'}
		echo $suffix
		
		if [ "$suffix"== "Grand1"  ] 
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

					
			
			#dir_input="/usr/local/NeSVoR/simul_data/"${suffix}"/svort_mvt"
			dir_input="/usr/local/NeSVoR/simul_data/"${suffix}"/svort_mvt"
			dir_output='/usr/local/NeSVoR/nesvor_rosi_reconstruct/'${suffix}
			docker exec nesvor_contener mkdir '/usr/local/NeSVoR/nesvor_rosi_reconstruct/'
			docker exec nesvor_contener mkdir '/usr/local/NeSVoR/nesvor_rosi_reconstruct/'${suffix}_test
		    docker exec nesvor_contener nesvor reconstruct --input-slices $dir_input  --output-volume $dir_output".nii.gz"  --registration none --no-transformation-optimization --output-slices $dir_output 
			#--no-transformation-optimization
			#--no-pixel-variance --no-slice-variance

		fi 
		
}

mkdir ${results}'/simul_data/'
for simul in ${simul_file}'/Grand1'
#${listr[@]}
do
		#simuldata=${simul_file}${simul}
		echo "simu"
		#echo $simul
		
		#task "$simul" 
		
		#echo 'simul directory'
		#echo $simul &
		#pro=ps -A --no-headers | wc -l
		#echo $pro
		#docker exec e91b0f478887 rm -r 'NiftyMIC/ipta/'
		#docker exec e91b0f478887 mkdir 'NiftyMIC/ipta/'
	       	
done

	


