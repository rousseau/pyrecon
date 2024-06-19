import os

if __name__=="__main__":
	
	DB_path = "/envau/work/meca/data/Fetus/datasets/MarsFet/derivatives/preprocessing"
	res_path = "/envau/work/meca/users/2024_mercier.c/results/"
	
	subjects = os.listdir(DB_path)
	subjects.sort()

	for subject in subjects:
		subj_dir = os.path.join(DB_path, subject)
		sessions = os.listdir(subj_dir)
		sessions.sort()
		#print(subj_dir)
		#print(subject)
		if subject=='sub-0018':
			for session in sessions:
				print(session)
				if session=='ses-0021':
					list_stacks = []
					list_masks = []
					dir_session = os.path.join(subj_dir, session)
					print("--------------" + subject)
					dir_reconst = os.path.join(DB_path, "", subject, session)
					list_files = os.listdir(dir_reconst)
					for file in list_files:
						#print(file)
						if file.endswith("_desc-denoised_T2w.nii.gz") and 'haste' in file:
							#stack = os.path.join(dir_reconst, subject+ "_"+ session + "_"+ "acq-"+ sequence+ "_"+ "run" + "-" + serie + "_desc-denoised_T2w.nii.gz")
							list_stacks.append(file)
						elif file.endswith("_desc-brainmask_T2w.nii.gz") and 'haste' in file:
							list_masks.append(file)
				list_stacks = ''.join(str(list_stacks) for list_stacks in list_stacks)
				list_masks = ''.join(str(list_masks) for list_masks in list_masks)
				output_sub = os.path.join(res_path,subject)
				if not os.path.exists(output_sub):
					os.mkdir(output_sub)
				output_ses = os.path.join(output_sub,session)
				if not os.path.exists(output_ses):
					os.mkdir(output_ses)
				print(output_ses)
				output = os.path.join(output_ses,"res")
				command = 'run_registration_function.py --filenames %s --filenames_mask %s --ouptut %s --no_multistart 1' %(list_stacks,list_masks,output)
				print(command)
				print('---stacks----')
				print(list_stacks)
				print('---masks----')
				print(list_masks)


            			
                    
