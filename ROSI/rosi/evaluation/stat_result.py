#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:16:22 2022

@author: mercier
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from rec_ebner import computeRegErrorEbner
from load import loadSlice
from data_simulation import compute_registration_error
import nibabel as nib
from os.path import exists



#read data :
error_before_correction_reallysmall = []   
error_after_correction_reallysmall = []
error_after_correction_ebner_reallysmall_no_rejection=[]
error_before_correction_small_no_rejection=[]
error_after_correction_small_no_rejection=[]
error_after_correction_ebner_small_no_rejection=[]
error_before_correction_medium_no_rejection=[]
error_after_correction_medium_no_rejection=[]
error_after_correction_ebner_medium_no_rejection=[]

proportion_rejected_reallysmall_ebner = np.zeros(5)
proportion_rejected_small_ebner = np.zeros(5)
proportion_rejected_medium_ebner = np.zeros(5)



n=5
for i in range(0,n):
    value_image=i+1
    petit = '../res/new_results/trespetit%d.joblib.gz' %(value_image)
    directory = '../data/test/tres_petit%d/' %(value_image)
    print(os.path.exists(petit))
    
    if os.path.exists(petit) : 
        
        print(petit)
    
        res = joblib.load(open(petit,'rb'))
        print(res)
        key=[p[0] for p in res]
        element=[p[1] for p in res]
        listSlice=element[key.index('listSlice')]
        
        parameters_slices = element[key.index('EvolutionParameters')][-1,:,:]
        
        rejected_slices = element[key.index('RejectedSlices')]
    
        nb_rejected_slices = np.shape(rejected_slices)[0]
        nb_slices = len(listSlice)
        #proportion_rejected_slices_reallysmall[i-1] = nb_rejected_slices/nb_slices
        
        
        transfo_axial =  directory + 'transfoAx_trespetit%d.npy' %(value_image)
        transfo_sagittal  = directory + 'transfoSag_trespetit%d.npy' %(value_image)
        transfo_coronal = directory + 'transfoCor_trespetit%d.npy' %(value_image)
        transfo = np.array([transfo_axial,transfo_coronal,transfo_sagittal])
        
        axial_without_movment = directory + 'LrAxNifti_nomvt.nii.gz'
        img_axial_without_movment = nib.load(axial_without_movment)
        axial_mask_without_movment = directory + 'AxMask_trespetit%d.nii.gz' %(value_image)
        img_axial_mask_without_movment = nib.load(axial_mask_without_movment)
        coronal_without_movment = directory + 'LrCorNifti_nomvt.nii.gz'
        img_coronal_without_movment = nib.load(coronal_without_movment)
        coronal_mask_without_movment = directory + 'CorMask_trespetit%d.nii.gz' %(value_image)
        img_coronal_mask_without_movment = nib.load(coronal_mask_without_movment)
        sagittal_without_movment = directory + 'LrSagNifti_nomvt.nii.gz'
        img_sagittal_without_movment = nib.load(sagittal_without_movment)
        sagittal_mask_without_movment = directory + 'SagMask_trespetit%d.nii.gz' %(value_image)
        img_sagittal_mask_without_movment = nib.load(sagittal_mask_without_movment)
        
        listnomvt = []
        loadSlice(img_axial_without_movment,img_axial_mask_without_movment,listnomvt,0,0)
        loadSlice(img_coronal_without_movment,img_coronal_mask_without_movment,listnomvt,1,1)
        loadSlice(img_sagittal_without_movment,img_sagittal_mask_without_movment,listnomvt,2,2)
        

        listerrorimg1img2_after = compute_registration_error(listSlice,listnomvt,rejected_slices,res,transfo)
            
        error_after_correction_reallysmall.extend(listerrorimg1img2_after) 
        

        dir_motion = '../res/new_results/tres_petit%d_all/motion_correction' %(value_image)
        listtest = []
        for i in range(0,len(listSlice)):
            islice=listSlice[i].copy()
            islice.set_parameters([0,0,0,0,0,0])
            listtest.append(islice)
            
        listerrorimg1img2_after = compute_registration_error(listtest,listnomvt,rejected_slices,res,transfo)
        error_before_correction_reallysmall.extend(listerrorimg1img2_after)  
            
        if exists(dir_motion):
           prefix=['LrAxNifti_trespetit%d'%(value_image),'LrCorNifti_trespetit%d'%(value_image),'LrSagNifti_trespetit%d'%(value_image)]
           error_ebner_no_rejection,listSlice = computeRegErrorEbner(dir_motion,listtest, listnomvt,transfo,prefix)
           error_after_correction_ebner_reallysmall_no_rejection.extend(error_ebner_no_rejection)



n=5
for i in range(0,n):
    value_image=i+1

    petit = '../res/new_results/petit%d.joblib.gz' %(value_image)
    #directory1 = '/home/mercier/Documents/donnee/test/Grand%d/' %(i)
    directory = '../data/test/Petit%d/' %(value_image)
    
    if os.path.exists(petit) : 
        
        print(petit)
    
        res = joblib.load(open(petit,'rb'))
        key=[p[0] for p in res]
        element=[p[1] for p in res]
        listSlice=element[key.index('listSlice')]
        
        
        parameters_slices = element[key.index('EvolutionParameters')][-1,:,:]
        
        rejected_slices = element[key.index('RejectedSlices')]
     
        nb_rejected_slices = np.shape(rejected_slices)[0]
        nb_slices = len(listSlice)
        #proportion_rejected_slices_small[i-1] = nb_rejected_slices/nb_slices
        
        
        transfo_axial =  directory + 'transfoAx_petit%d.npy' %(value_image)
        transfo_sagittal  = directory + 'transfoSag_petit%d.npy' %(value_image)
        transfo_coronal = directory + 'transfoCor_petit%d.npy' %(value_image)
        transfo = np.array([transfo_axial,transfo_coronal,transfo_sagittal])
        
        axial_without_movment = directory + 'LrAxNifti_nomvt.nii.gz'
        img_axial_without_movment = nib.load(axial_without_movment)
        axial_mask_without_movment = directory + 'AxMask_petit%d.nii.gz' %(value_image)
        img_axial_mask_without_movment = nib.load(axial_mask_without_movment)
        coronal_without_movment = directory + 'LrCorNifti_nomvt.nii.gz'
        img_coronal_without_movment = nib.load(coronal_without_movment)
        coronal_mask_without_movment = directory + 'CorMask_petit%d.nii.gz' %(value_image)
        img_coronal_mask_without_movment = nib.load(coronal_mask_without_movment)
        sagittal_without_movment = directory + 'LrSagNifti_nomvt.nii.gz'
        img_sagittal_without_movment = nib.load(sagittal_without_movment)
        sagittal_mask_without_movment = directory + 'SagMask_petit%d.nii.gz' %(value_image)
        img_sagittal_mask_without_movment = nib.load(sagittal_mask_without_movment)
        
        listnomvt = []
        loadSlice(img_axial_without_movment,img_axial_mask_without_movment,listnomvt,0,0)
        loadSlice(img_coronal_without_movment,img_coronal_mask_without_movment,listnomvt,1,1)
        loadSlice(img_sagittal_without_movment,img_sagittal_mask_without_movment,listnomvt,2,2)
        
        
        listerrorimg1img2_after = compute_registration_error(listSlice,listnomvt,rejected_slices,res,transfo)
        error_after_correction_small_no_rejection.extend(listerrorimg1img2_after) 
        
        
        dir_motion = '../res/new_results/Petit%d_all/motion_correction' %(value_image)
        listtest = []
        for i in range(0,len(listSlice)):
            islice=listSlice[i].copy()
            islice.set_parameters([0,0,0,0,0,0])
            listtest.append(islice)
            
        listerrorimg1img2_after = compute_registration_error(listtest,listnomvt,rejected_slices,res,transfo)
        error_before_correction_small_no_rejection.extend(listerrorimg1img2_after) 
            
        if exists(dir_motion):
           prefix=['LrAxNifti_petit%d'%(value_image),'LrCorNifti_petit%d'%(value_image),'LrSagNifti_petit%d'%(value_image)]
           error_ebner_no_rejection,listSlice = computeRegErrorEbner(dir_motion,listtest, listnomvt,transfo,prefix)
           error_after_correction_ebner_small_no_rejection.extend(error_ebner_no_rejection)
        
        
        
n=5    
for i in range(0,n):
    value_image=i+1
    print(value_image)
    petit = '../res/new_results/moyen%d.joblib.gz' %(value_image)
    #directory1 = '/home/mercier/Documents/donnee/test/Grand%d/' %(i)
    directory = '../data/test/Moyen%d/' %(value_image)
    
    if os.path.exists(petit) : 
        
        print(petit)
    
        res = joblib.load(open(petit,'rb'))
        key=[p[0] for p in res]
        element=[p[1] for p in res]
        listSlice=element[key.index('listSlice')]
        
        
        parameters_slices = element[key.index('EvolutionParameters')][-1,:,:]
        
        rejected_slices = element[key.index('RejectedSlices')]
     
        nb_rejected_slices = np.shape(rejected_slices)[0]
        nb_slices = len(listSlice)
        #proportion_rejected_slices_small[i-1] = nb_rejected_slices/nb_slices
        
        
        transfo_axial =  directory + 'transfoAx_moyen%d.npy' %(value_image)
        transfo_sagittal  = directory + 'transfoSag_moyen%d.npy' %(value_image)
        transfo_coronal = directory + 'transfoCor_moyen%d.npy' %(value_image)
        transfo = np.array([transfo_axial,transfo_coronal,transfo_sagittal])
        
        axial_without_movment = directory + 'LrAxNifti_nomvt.nii.gz'
        img_axial_without_movment = nib.load(axial_without_movment)
        axial_mask_without_movment = directory + 'AxMask_moyen%d.nii.gz' %(value_image)
        img_axial_mask_without_movment = nib.load(axial_mask_without_movment)
        coronal_without_movment = directory + 'LrCorNifti_nomvt.nii.gz'
        img_coronal_without_movment = nib.load(coronal_without_movment)
        coronal_mask_without_movment = directory + 'CorMask_moyen%d.nii.gz' %(value_image)
        img_coronal_mask_without_movment = nib.load(coronal_mask_without_movment)
        sagittal_without_movment = directory + 'LrSagNifti_nomvt.nii.gz'
        img_sagittal_without_movment = nib.load(sagittal_without_movment)
        sagittal_mask_without_movment = directory + 'SagMask_moyen%d.nii.gz' %(value_image)
        img_sagittal_mask_without_movment = nib.load(sagittal_mask_without_movment)
        
        listnomvt = []
        loadSlice(img_axial_without_movment,img_axial_mask_without_movment,listnomvt,0,0)
        loadSlice(img_coronal_without_movment,img_coronal_mask_without_movment,listnomvt,1,1)
        loadSlice(img_sagittal_without_movment,img_sagittal_mask_without_movment,listnomvt,2,2)
        
        
        listerrorimg1img2_after = compute_registration_error(listSlice,listnomvt,rejected_slices,res,transfo)
        error_after_correction_medium_no_rejection.extend(listerrorimg1img2_after) 
        
        listtest = []
        for i in range(0,len(listSlice)):
           islice=listSlice[i].copy()
           islice.set_parameters([0,0,0,0,0,0])
           listtest.append(islice) 
        
        listerrorimg1img2_after = compute_registration_error(listtest,listnomvt,rejected_slices,res,transfo)
        error_before_correction_medium_no_rejection.extend(listerrorimg1img2_after) 
        
        
        dir_motion = '../res/new_results/Moyen%d_all/motion_correction' %(value_image)
        if exists(dir_motion):
            prefix=['LrAxNifti_moyen%d'%(value_image),'LrCorNifti_moyen%d'%(value_image),'LrSagNifti_moyen%d'%(value_image)]
            error_ebner_no_rejection, listTest = computeRegErrorEbner(dir_motion,listSlice, listnomvt,transfo,prefix)
            error_after_correction_ebner_medium_no_rejection.extend(error_ebner_no_rejection)
        
   
error_after_correction_reallysmall = np.array(error_after_correction_reallysmall) 
error_after_correction_medium_no_rejection = np.array(error_after_correction_medium_no_rejection)
error_after_correction_small_no_rejection = np.array(error_after_correction_small_no_rejection)
   
error_after_correction_reallysmall = error_after_correction_reallysmall[np.where(error_after_correction_reallysmall>0)]
error_after_correction_medium_no_rejection = error_after_correction_medium_no_rejection[np.where(error_after_correction_medium_no_rejection>0)]
error_after_correction_small_no_rejection = error_after_correction_small_no_rejection[np.where(error_after_correction_small_no_rejection>0)]
   
#fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 6),sharey=True,)

#axs[0].set_ylabel('error (in mm)')

#c='blue'
#axs[0].boxplot(error_after_correction_reallysmall,positions=[0],showfliers=False,showmeans=True,boxprops=dict(color=c),capprops=dict(color=c),whiskerprops=dict(color=c),medianprops=dict(color=c))
#c='green'
#axs[0].boxplot(np.concatenate(error_after_correction_ebner_reallysmall_no_rejection),positions=[1],showfliers=False,showmeans=True,boxprops=dict(color=c),capprops=dict(color=c),whiskerprops=dict(color=c),medianprops=dict(color=c))
#axs[0].set_xticklabels(['small - INTER','small - NiftyMIC'])

#c='blue'
#axs[1].boxplot(error_after_correction_small_no_rejection,positions=[0],showfliers=False,showmeans=True,boxprops=dict(color=c),capprops=dict(color=c),whiskerprops=dict(color=c),medianprops=dict(color=c))
#c='green'
#axs[1].boxplot(np.concatenate(error_after_correction_ebner_small_no_rejection),positions=[1],showfliers=False,showmeans=True,boxprops=dict(color=c),capprops=dict(color=c),whiskerprops=dict(color=c),medianprops=dict(color=c))
#axs[1].set_xticklabels(['medium - INTER','medium - NiftyMIC'])

#c='blue'
#axs[2].boxplot(error_after_correction_medium_no_rejection,positions=[0],showfliers=False,showmeans=True,boxprops=dict(color=c),capprops=dict(color=c),whiskerprops=dict(color=c),medianprops=dict(color=c))
#c='green'
#axs[2].boxplot(np.concatenate(error_after_correction_ebner_medium_no_rejection),positions=[1],showfliers=False,showmeans=True,boxprops=dict(color=c),capprops=dict(color=c),whiskerprops=dict(color=c),medianprops=dict(color=c))
#axs[2].set_xticklabels(['large - INTER','large - NiftyMIC'])

#plt.savefig('inter_niftymic_boxplot.png')


fig,axs = plt.subplots(1,3, figsize=(40, 15))

ebner_toutpetit = np.concatenate(error_after_correction_ebner_reallysmall_no_rejection)
ebner_petit = np.concatenate(error_after_correction_ebner_small_no_rejection)
ebner_moyen = np.concatenate(error_after_correction_ebner_medium_no_rejection)

axs[0].scatter(error_after_correction_reallysmall,error_before_correction_reallysmall,c='green',marker='.',s=170)
axs[0].scatter(ebner_toutpetit,error_before_correction_reallysmall,c='blue',marker='.',s=170)
min=0#np.min(error_before_correction_medium_no_rejection)
max=np.max(error_before_correction_medium_no_rejection)
min2=0#np.min(ebner_moyen)
max2=np.max(error_before_correction_medium_no_rejection)
axs[0].set_yticks(np.arange(min, max),fontsize=40) 
axs[0].set_xticks(np.arange(min2, max2),fontsize=40) 
axs[0].set_ylabel('avant recalage',fontsize=60)
axs[0].set_xlabel('après recalage',fontsize=60)
axs[0].set_title('faible mouvement',fontsize=60)


for tick in axs[0].xaxis.get_majorticklabels():  # example for xaxis
    tick.set_fontsize(30) 

for tick in axs[0].yaxis.get_majorticklabels():  # example for xaxis
    tick.set_fontsize(30)    


#axs.yaxis.get_offset_text().set_fontsize(24)
#axs.xaxis.get_offset_text().set_fontsize(24)



id = np.where(ebner_petit<max)
ebner_before = np.array(error_before_correction_small_no_rejection)[id]
ebner_petit = ebner_petit[id]
id2 = np.where(error_after_correction_small_no_rejection<max)
nous_before = np.array(error_before_correction_small_no_rejection)[id2]
error_after_correction_small_no_rejection = np.array(error_after_correction_small_no_rejection)[id2]
axs[1].scatter(error_after_correction_small_no_rejection,nous_before,c='green',marker='.',s=170)
axs[1].scatter(ebner_petit,ebner_before,c='blue',marker='.',s=170)
axs[1].set_yticks(np.arange(min, max),fontsize=100) 
axs[1].set_xticks(np.arange(min2, max2),fontsize=100) 
axs[1].set_ylabel('avant recalage',fontsize=60)
axs[1].set_xlabel('après recalage',fontsize=60)
axs[1].set_title('mouvement moyen',fontsize=60)

for tick in axs[1].xaxis.get_majorticklabels():  # example for xaxis
    tick.set_fontsize(30) 

for tick in axs[1].yaxis.get_majorticklabels():  # example for xaxis
    tick.set_fontsize(30) 

id = np.where(ebner_moyen<max)
ebner_before = np.array(error_before_correction_medium_no_rejection)[id]
ebner_moyen = ebner_moyen[id]
id2 = np.where(error_after_correction_medium_no_rejection<max)
nous_before = np.array(error_before_correction_medium_no_rejection)[id2]
error_after_correction_medium_no_rejection = np.array(error_after_correction_medium_no_rejection)[id2]
axs[2].scatter(error_after_correction_medium_no_rejection,nous_before,c='green',marker='.',s=170)
axs[2].scatter(ebner_moyen,ebner_before,c='blue',marker='.',s=170)
axs[2].set_yticks(np.arange(min, max),fontsize=100) 
axs[2].set_xticks(np.arange(min2, max2),fontsize=100) 
axs[2].set_ylabel('avant recalage',fontsize=60)
axs[2].set_xlabel('après recalage',fontsize=60)
axs[2].set_title('mouvement large',fontsize=60)

for tick in axs[2].xaxis.get_majorticklabels():  # example for xaxis
    tick.set_fontsize(30) 

for tick in axs[2].yaxis.get_majorticklabels():  # example for xaxis
    tick.set_fontsize(30) 

#axs.yaxis.get_offset_text().set_fontsize(24)
#axs.xaxis.get_offset_text().set_fontsize(24)

fig.tight_layout()

plt.legend(["ROSI", "NiftyMIC"], fontsize=50, loc ="lower right")

plt.savefig('scatter_plot.png')
