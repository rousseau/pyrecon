import os
import re
import pathlib
import subprocess

def get_identifier(filepath, pattern):
    """Get project identifier(s)"""
    filename = os.path.basename(filepath)
    regex = re.compile(pattern)
    match = regex.search(filename)
    if match is not None:
        identifier = match.group()
    else:
        identifier = None
    return identifier


def execute_rosi(input_slices,output_dir,slurm_config):
    """
    Execute ROSI, with svort initialisation
    :filenames: dir input with individual slices, in an nifti file
    :output: output directory
    :return: None
    """
    dir_out = pathlib.Path(output_dir)
    dir_out.mkdir(parents=True, exist_ok=True)

    dir_in = pathlib.Path(input_slices)


    if os.path.exists(dir_in):
        cmd = [
            "sbatch",
            slurm_config,
            dir_in,
            dir_out,
            ]
        subprocess.run(cmd)
    pass

if __name__ == "__main__":

    rosi_slurm = (
        "/scratch/cmercier/code/ROSI/utils/slurm/rosi.slurm"
    )

    MARSFET_DATAPATH = "/scratch/gauzias/data/datasets/MarsFet/derivatives/preprocessing"  

    MARSFET_MESO_SVORT_INIT = "/scratch/cmercier/results/"
    MARSFET_MESO_ROSI = "/scratch/cmercier/results/ROSI/"
    
    input_data = MARSFET_MESO_SVORT_INIT
    output_data = MARSFET_MESO_ROSI
    stacks_path = MARSFET_DATAPATH

    subjects = [sub for sub in os.listdir(stacks_path) if os.path.isdir(os.path.join(stacks_path,sub))]
    subjects.sort()

    for subject in subjects:
        dir_subject = os.path.join(stacks_path, subject)
        sessions = os.listdir(dir_subject)
        for session in sessions:
            if subject == 'sub-0001' and session == 'ses-0001':
                input_slices = os.path.join(input_data,subject, session)
                dir_out = os.path.join(output_data, subject, session)
                print('input_slices:',input_slices)
                print('dir_output:',dir_out)
                execute_rosi(input_slices,dir_out,rosi_slurm)

