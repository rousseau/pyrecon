#!/usr/bin/env python2
"""

  This software is governed by the CeCILL-B license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL-B
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".

  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited
  liability.

  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.

  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL-B license and that you accept its terms.

"""

import os
import string

path = "/home/miv/faisan/data/data4sylvain/"
pathres = "/home/miv/faisan/enCours/registration/pyrecon/res/"

fileName    = ["mar0027_exam01_T2_haste_axial_crop.nii.gz","stras0001_exam01_T2_haste_axial_crop_nlm.nii.gz","stras0002_exam01_T2_haste_axial_crop_nlm.nii.gz","stras0003_exam01_T2_haste_axial_crop_nlm.nii.gz"]
saveComment = ["mar0027.txt","stras0001.txt","stras0002.txt","stras0003.txt"]
Mask     = [1,0,0,0]

for i in range(len(fileName)): 
  useMask = Mask[i]
  filenameAxial = fileName[i]
  filenameCoronal = string.replace(filenameAxial, 'axial', 'coronal')
  filenameSagittal = string.replace(filenameAxial, 'axial', 'sagittal')
  filenameAxialMask = string.replace(filenameAxial, 'axial', 'axial_mask')
  filenameCoronalMask = string.replace(filenameAxial, 'axial', 'coronal_mask')
  filenameSagittalMask = string.replace(filenameAxial, 'axial', 'sagittal_mask')
  filenameRes = string.replace(filenameAxial, 'axial', 'reconstructed')
  
  command = "python pyrecon.py -r " + path + filenameAxial 
  if useMask == 1:
    command = command + " --refmask=" + path + filenameAxialMask 
  command = command +  " -i " + path + filenameCoronal + " --reslice=True" 

  if useMask == 1:
    command = command + " --inmask=" + path + filenameCoronalMask  
  command = command + " -o " + pathres + filenameCoronal + "  -s 0 --rx -10 10 --ry -10 10 --rz -10 10 --criterium L1" 

  os.system(command)

  command = "python pyrecon.py -r " + path + filenameAxial 
  if useMask == 1:
    command = command + " --refmask=" + path + filenameAxialMask 
  command = command +  " -i " + path + filenameSagittal  + " --reslice=True" 
  if useMask == 1:
    command = command + " --inmask=" + path + filenameSagittalMask  
  command = command + " -o " + pathres + filenameSagittal + " -s 0 --rx -10 10 --ry -10 10 --rz -10 10 --criterium L1" 
  os.system(command)
#  
  cmd = "python kim.py --input "+ path + filenameAxial + " --input " + pathres + filenameSagittal + " --input " + pathres + filenameCoronal
  if useMask == 1:
    cmd = cmd + " --inmask=" + path + filenameAxialMask + " --inmask=" + path + filenameSagittalMask + " --inmask=" + path + filenameCoronalMask 
  cmd = cmd + " -o " + pathres + filenameRes # + " > " + saveComment[i] + " 2>&1 "
  os.system(cmd)





