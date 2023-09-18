#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 09:09:19 2022

@author: mercier
"""

import numpy as np
from data_simulation import error_for_each_slices, findCommonPointbtw2V
import sliceObject
import registration as re
import pickle
import joblib as job
import tools
from outliers_detection_intersection import ErrorSlice
import nibabel as nib
from load import loadSlice
from tools import createVolumesFromAlist
import matplotlib.pyplot as plt


print('Hello World')

loaded_model = pickle.load(open('my_model.pickle', "rb"))
data = job.load('../res/outliers_detection/value1/sub-0051/ses-0061/res_test_omega1.joblib.gz')
key = [p[0] for p in data] 
element = [p[1] for p in data]

listSlice = element[key.index('listSlice')]
listError = element[key.index('ListError')]

transfo_axial = '../data/Data11/transfoAx_data11.npy'
transfo_coronnal = '../data/Data11/transfoCor_data11.npy'
transfo_sagittal = '../data/Data11/transfoSag_data11.npy'

axial_nomvt = nib.load('../data/Data11/LrAxNifti_data11.nii.gz')
axial_mask = nib.load('../data/Data11/LrAxNifti_data11_mask.nii.gz')
coronal_nomvt = nib.load('../data/Data11/LrCorNifti_data11.nii.gz')
coronal_mask = nib.load('../data/Data11/LrCorNifti_data11_mask.nii.gz')
sagittal_nomvt = nib.load('../data/Data11/LrSagNifti_data11.nii.gz')
sagittal_mask = nib.load('../data/Data11/LrSagNifti_data11_mask.nii.gz')

listnomvt = []
loadSlice(axial_nomvt,axial_mask,listnomvt,0,0)
loadSlice(coronal_nomvt,coronal_mask,listnomvt,1,1)
loadSlice(sagittal_nomvt,sagittal_mask,listnomvt,2,2)

transfo = np.array([transfo_axial,transfo_coronnal,transfo_sagittal])

listErrorSlice_v2 = [ErrorSlice(slicei.get_orientation(),slicei.get_index_slice()) for slicei in listSlice]
error_for_each_slices(listnomvt,listSlice,listErrorSlice_v2,transfo)

num_slice1 = 4
num_slice2 = 28
slice2 = listSlice[num_slice2]
print(slice2.get_orientation())

error = [e.get_error() for e in listError]
print('error for slice1 :', error[num_slice1])
print('error for slice2 :', error[num_slice2])

#print(error)

def check_multistart(num_slice1,num_slice2,listSlice,transfo):
    
    c1 = listSlice[num_slice1]
    c2 = listSlice[num_slice2]

    Mest_1 = c1.get_transfo()
    Mest_2 = c2.get_transfo()
    M1 = transfo[c1.get_index_slice()] @ c1.get_slice().affine
    M2 = transfo[c2.get_index_slice()] @ c2.get_slice().affine
    T = np.dot(Mest_1,np.linalg.inv(M1))
    M2new = T @ M2
    center = c2.get_center()
    center_mat = np.eye(4)
    center_mat[0:3,3] = center
    center_inv = np.eye(4)
    center_inv[0:3,3] = -center
    
    M_est = center_mat @ Mest_2 @ np.linalg.inv(c2.get_slice().affine) @ center_inv
    x_est = tools.ParametersFromRigidMatrix(M_est)
    

    #print('x_est :', x_est)
    #print('x :', c2.get_parameters())
    
    ge,gn,gi,gu = re.computeCostBetweenAll2Dimages(listSlice)
    grid_slices = np.array([ge,gn,gi,gu])
    set_o = re.detect_misregistered_slice(listSlice,grid_slices,loaded_model)
    set_o = np.zeros(len(listSlice))

    print(re.cost_fct(x_est,num_slice2,listSlice,grid_slices,set_o,0))
    
    ge,gn,gi,gu = re.computeCostBetweenAll2Dimages(listSlice)
    grid_slices = np.array([ge,gn,gi,gu])
    M_theorique = center_mat @ M2new @ np.linalg.inv(c2.get_slice().affine) @ center_inv
    x_theorique = tools.ParametersFromRigidMatrix(M_theorique)
    print(c2.get_parameters())
    print(x_theorique)


    print(re.cost_fct(x_theorique,num_slice2,listSlice,grid_slices,set_o,0)) 


check_multistart(num_slice1,num_slice2,listSlice,np.load(transfo[slice2.get_orientation()]))
ge,gn,gi,gu = re.computeCostBetweenAll2Dimages(listSlice)
print(ge[:,num_slice2],ge[num_slice2,:])
row = [np.sum(n) for n in gn]
#print(row)
col = [np.sum(n) for n in gn.T]
#print(col)
vect_n = np.array([row])+np.array([col])
vect_n = vect_n[0]
#print(vect_n)
print(vect_n[num_slice2])

row = [np.sum(n) for n in ge]
#print(row)
col = [np.sum(n) for n in ge.T]
#print(col)
vect_e = np.array([row])+np.array([col])
vect_e = vect_e[0]

print(vect_e[num_slice2])
#plt.scatter(vect_n,volume,c=e)
#fig.savefig('n_volume.png')
#error_v2 = listErrorSlice_v2[index_slice]
#error_v2 = [e.get_error() for e in listErrorSlice_v2]
#print("error with previous version",error)
#print("error with new version", error_v2)
#print(np.array([error])-np.array([error_v2]))
#volume,images = createVolumesFromAlist(listnomvt)
#rejectedSlice=[]
#print(gn.shape)
#print(len(col))
#print(len(vect_n))
#re.update_feature(listSlice,listError,ge,gn,gi,gu)
#mse = [e.get_mse() for e in listError]
#dice = [e.get_dice() for e in listError]
#volume = [e.get_mask_proportion() for e in listError]
#print(mse)
#print(dice)
#print(volume)
#fig = plt.figure()

#plt.show()

#Choose one slice : 
#index_slice = 3
#error = listError[index_slice]
#error = [e.get_error() for e in listError]
#e = np.array([error])>1.5
#plt.scatter(vect_n,volume,c=e)
#fig.savefig('n_volume.png')
#error_v2 = listErrorSlice_v2[index_slice]
#error_v2 = [e.get_error() for e in listErrorSlice_v2]
#print("error with previous version",error)
#print("error with new version", error_v2)
#print(np.array([error])-np.array([error_v2]))
#volume,images = createVolumesFromAlist(listnomvt)
#rejectedSlice=[]
#findCommonPointbtw2V(volume[0],volume[1],np.load(transfo[0]),np.load(transfo[1]),rejectedSlice)
plt.imshow(slice2.get_slice().get_fdata())
plt.savefig('figure_test.png')
plt.show()
