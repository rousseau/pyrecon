import os

if __name__=="__main__":
	
	DB_path = "/envau/work/meca/data/Fetus/datasets/MarsFet/preprocessing"
	sequences = ["haste", "tru"]

	subjects = os.listdir(DB_path)
	subjects.sort()
	
	for subject in subjects:
		subj_dir = os.path.join(DB_path, "preprocessing", subject)
		sessions = os.listdir(subj_dir)
		sessions.sort()
		for session in sessions:
            		dir_session = os.path.join(subj_dir, session)
            		for sequence in sequences:
            			print("--------------" + subject)
            			dir_reconst = os.path.join(DB_path, "", subject, session, sequence,)
						stack = os.path.join(dir_reconst, subject+ "_"+ session + "_"+ "acq-"+ sequence+ "_"+ "run-1_desc-denoised_T2w.nii.gz",)
						print(stack)
            			
                    
