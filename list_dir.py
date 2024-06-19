import os

if __name__=="__main__":
	
	DB_path = "/envau/work/meca/data/Fetus/datasets/MarsFet/derivatives/preprocessing"
	
	subjects = os.listdir(DB_path)
	subjects.sort()

	list_stacks = []
	list_masks = []
	
	for subject in subjects:
		subj_dir = os.path.join(DB_path, subject)
		sessions = os.listdir(subj_dir)
		sessions.sort()
		for session in sessions:
			dir_session = os.path.join(subj_dir, session)
			print("--------------" + subject)
			dir_reconst = os.path.join(DB_path, "", subject, session)
			list_files = os.listdir(dir_reconst)
			for file in list_files:
				#print(file)
				if file.endswith("_desc-denoised_T2w.nii.gz") and 'haste' in file:
					#stack = os.path.join(dir_reconst, subject+ "_"+ session + "_"+ "acq-"+ sequence+ "_"+ "run" + "-" + serie + "_desc-denoised_T2w.nii.gz")
					list_stacks.append(file)
				elif file.endswith("_desc-brainmask_T2w.nii.gz"):
					list_masks.append(file)
		print('---stacks----')
		print(list_stacks)
		print('---masks----')
		print(list_masks)
            			
                    
