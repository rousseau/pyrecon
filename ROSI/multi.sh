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


for multi_start in  "Nelder-Mead"
#"LM" "BFGS" "Powell" "TNC" "CG"
do
	mkdir '../res/multi_6/'
	output_path='../res/multi_6/'${multi_start}
	mkdir $output_path
	results='../res/multi_6/'${multi_start}'/value'
	mkdir $results
	echo "opti"
	echo $multi_start
	

	task (){
			
			simul=$1

			suffix=${simul#*${simul_file}'/'}
			echo $suffix
			
			if [ "$suffix" != "../simu" ];
			
				#
			then
				
				echo 'suffix'
				echo ${suffix}

				simul_data=${suffix}
				job="/mnt/Data/Chloe/res/omega/value0/simul_data/"${suffix}"/omega0/res_test_omega0.joblib.gz"
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
				mkdir ${results}'/simul_data/'${suffix}'/'${multi_start}
						
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
						
				output_simul_path=${results}'/simul_data/'${suffix}'/'${multi_start}
				output_simul=${output_simul_path}'/res_test_'${multi_start}
				mkdir output_simul
				output_res='NiftyMIC/ipta/'${suffix}'/'${multi_start}
						
						
						#echo $list_docker
						#echo "herisson"
						#echo ${output_simul_path}
						#echo ${output_res}
								
				python ROSI/main.py --filenames ${image}  --filenames_masks "${mask[@]}" --nomvt "${nomvt[@]}" --simulation "${transfo[@]}" --output ${output_simul}  --ablation  multistart dice Nelder-Mead --hyperparameters 4 0.01 2000 0.25 1 0 
	
						
			fi 
	}

	mkdir ${results}'/simul_data/'
	for simul in ${listr[@]}
	do

			echo $simul &
			task $simul &
			
				
		done

		

	done 

done