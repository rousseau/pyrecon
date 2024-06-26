import os
import csv

if __name__=="__main__":
	
	DB_path = "/envau/work/meca/data/Fetus/datasets/MarsFet/derivatives/preprocessing"
	res_path = "/envau/work/meca/users/2024_mercier.c/results/"
	path_to_subjectofinterest = "/envau/work/meca/users/2024_mercier.c/marsfet_tables/marsfet_latest_participants.csv"
	list_data = []

	with open(path_to_subjectofinterest,'r') as csvfile:
		csvsub = csv.reader(csvfile,delimiter=',')

		for row in csvsub:
			print(row)
			data = (row[0],row[1])
			print(data)
			list_data.append((data))

	print(list_data)
	subjects = os.listdir(DB_path)
	subjects.sort()

	for subject in subjects:
		subj_dir = os.path.join(DB_path, subject)
		sessions = os.listdir(subj_dir)
		sessions.sort()
		#print(subj_dir)
		#print(subject)
		for session in sessions:
				print(subject,session)
				if (subject,session) in list_data:
					print('ici')
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
							path_to_file = os.path.join(dir_reconst,file)
							list_stacks.append(path_to_file)
						elif file.endswith("_desc-brainmask_T2w.nii.gz") and 'haste' in file:
							path_to_file = os.path.join(dir_reconst,file)
							list_masks.append(path_to_file)
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
					command = 'python run_registration.py --filenames %s --filenames_mask %s --output %s --no_multistart 1' %(list_stacks,list_masks,output)
					#os.system(command)
					print(command)
					print('---stacks----')
					print(list_stacks)
					print('---masks----')
					print(list_masks)


            			
                    
