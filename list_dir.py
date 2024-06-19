import os

if __name__=="__main__":
	
	DB_path = "/envau/work/meca/data/Fetus/datasets/MarsFet/derivatives/preprocessing"
	sequences = ["haste", "tru"]

	subjects = os.listdir(DB_path)
	subjects.sort()
	
	for subject in subjects:
		subj_dir = os.path.join(DB_path, subject)
		sessions = os.listdir(subj_dir)
		sessions.sort()
		for session in sessions:
			dir_session = os.path.join(subj_dir, session)
			for sequence in sequences:
				print("--------------" + subject)
				dir_reconst = os.path.join(DB_path, "", subject, session, sequence)
				list_files = os.listdir(dir_reconst)
				for file in list_files:
					print(file)
					if file.endswith("_desc-denoised_T2w.nii.gz"):
						#stack = os.path.join(dir_reconst, subject+ "_"+ session + "_"+ "acq-"+ sequence+ "_"+ "run" + "-" + serie + "_desc-denoised_T2w.nii.gz")
						print(file)
            			
                    
