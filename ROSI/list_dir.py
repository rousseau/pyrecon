import os
import csv

if __name__=="__main__":
	
	DB_path = "/scratch/gauzias/data/datasets/MarsFet/"
	stacks_path  = os.path.join(DB_path,"derivatives","preprocessing/")
	print(stacks_path)
	#"/envau/work/meca/data/Fetus/datasets/MarsFet/derivatives/preprocessing"
	#"/envau/work/meca/users/2024_mercier.c/results/"
	res_path = "/home/cmercier/results"
	#list_data = [('sub-0000','ses-0000')]

	subjects = [sub for sub in os.listdir(stacks_path) if os.path.isdir(os.path.join(stacks_path,sub))]
	subjects.sort()

	for subject in subjects:
		subj_dir = os.path.join(stacks_path, subject)
		sessions = os.listdir(subj_dir)
		sessions.sort()
		print(subj_dir)
		#print(subject)
		for session in sessions:
				print(subject,session)
				if True #(subject,session) in list_data:
					print('ici')
					list_stacks = []
					list_masks = []
					dir_session = os.path.join(subj_dir, session)
					print("--------------" + subject)
					dir_reconst = os.path.join(stacks_path, "", subject, session,"anat")
					print(dir_reconst)
					list_files = os.listdir(dir_session)
					for file in list_files:
							#print(file)
							if file.endswith("denoised_T2w.nii.gz") and 'haste' in file:
								#stack = os.path.join(dir_reconst, subject+ "_"+ session + "_"+ "acq-"+ sequence+ "_"+ "run" + "-" + serie + "_desc-denoised_T2w.nii.gz")
								path_to_file = os.path.join('/data',subject,session,file)
								list_stacks.append(path_to_file)
								run = file.find("run") #find the number of the run to make sure the stack is associated with its corresponding mask
								num_index = run + 4
								num = "run-%s" %(file[num_index])
								print("number run",num)
								for file in list_files:
									if file.endswith("brainmask_T2w.nii.gz") and 'haste' in file and num in file:
										path_to_file = os.path.join('/data',subject,session,file)
										list_masks.append(path_to_file)
										break
					print(list_stacks)
					print(list_masks)
					list_stacks = ' '.join(str(list_stacks) for list_stacks in list_stacks)
					list_masks = ' '.join(str(list_masks) for list_masks in list_masks)
					output_sub = os.path.join(res_path,subject)
					if not os.path.exists(output_sub):
							os.mkdir(output_sub)
					output_ses = os.path.join(output_sub,session)
					if not os.path.exists(output_ses):
							os.mkdir(output_ses)
					print(output_ses)
					output = os.path.join(output_ses,"res")
					command = "singularity exec --nv -B %s:/data /scratch/cmercier/softs/nesvor_latest.sif nesvor register --input-stacks %s --stack-masks %s --output-slices %s" %(stacks_path,list_stacks,list_masks,output)
						#command = 'python run_registration.py --filenames %s --filenames_mask %s --output %s --no_multistart 1' %(list_stacks,list_masks,output)
					os.system(command)
					print(command)
					print('---stacks----')
					print(list_stacks)
					print('---masks----')
					print(list_masks)
					print('--output--')
					print(output)


            			
                    
