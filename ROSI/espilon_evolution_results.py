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
from rosi.registration.tools import separate_slices_in_stacks
from rosi.registration.load import convert2Slices
from rosi.simulation.validation import tre_for_each_slices
import nibabel as nib
from psnrandssim import PSNR,SSIM
import ants

#read data :
error_before_correction_reallysmall=[]  
error_after_correction_reallysmall=[]
error_before_correction_small=[]
error_after_correction_small=[]
error_before_correction_medium=[]
error_after_correction_medium=[]
error_before_correction_grand=[]
error_after_correction_grand=[]

grand_psnr = []
grand_ssim = []
moyen_psnr = []
moyen_ssim = []
petit_psnr = []
petit_ssim = []
tres_petit_psnr = []
tres_petit_ssim = []


value_epsilon_evolution="0.5"


errorlist=[]
psnrlist=[]
ssimlist=[] 
for index_image in range(1,6):
    #error_before=[]
    file = '../../res/epsilon_evolution/value%s/simul_data/tres_petit%d/epsilon_evolution%s/epsilon_evolution%s' %(value_epsilon_evolution,index_image,value_epsilon_evolution,value_epsilon_evolution)
    petit = '../../res/epsilon_evolution/value%s/simul_data/tres_petit%d/epsilon_evolution%s/res_test_epsilon_evolution%s.joblib.gz' %(value_epsilon_evolution,index_image,value_epsilon_evolution,value_epsilon_evolution)
    directory = '../../simu/tres_petit%d/' %(index_image)
    print(petit)
    
    if os.path.exists(petit): 
           
        print(petit)
        res = joblib.load(petit)
        #print(res)
        key=[p[0] for p in res]
        element=[p[1] for p in res]
        listSlice=element[key.index('listSlice')]
        parameters_slices = element[key.index('EvolutionParameters')][-1,:,:]
        rejected_slices = element[key.index('RejectedSlices')]
        nb_rejected_slices = np.shape(rejected_slices)[0]
        nb_slices = len(listSlice)
        listFeature=element[key.index('ListError')]
        #proportion_rejected_slices_reallysmall[i-1] = nb_rejected_slices/nb_slices

        transfo_axial =  directory + 'transfoAx_trespetit%d.npy' %(index_image)
        transfo_sagittal  = directory + 'transfoSag_trespetit%d.npy' %(index_image)
        transfo_coronal = directory + 'transfoCor_trespetit%d.npy' %(index_image)
        transfo = np.array([transfo_axial,transfo_coronal,transfo_sagittal])
        
        axial_without_movment = directory + 'LrAxNifti_nomvt.nii.gz'
        img_axial_without_movment = nib.load(axial_without_movment)
        axial_mask_without_movment = directory + 'brain_mask/LrAxNifti_trespetit%d.nii.gz' %(index_image)
        img_axial_mask_without_movment = nib.load(axial_mask_without_movment)
        coronal_without_movment = directory + 'LrCorNifti_nomvt.nii.gz'
        img_coronal_without_movment = nib.load(coronal_without_movment)
        coronal_mask_without_movment = directory + 'brain_mask/LrCorNifti_trespetit%d.nii.gz' %(index_image)
        img_coronal_mask_without_movment = nib.load(coronal_mask_without_movment)
        sagittal_without_movment = directory + 'LrSagNifti_nomvt.nii.gz'
        img_sagittal_without_movment = nib.load(sagittal_without_movment)
        sagittal_mask_without_movment = directory + 'brain_mask/LrSagNifti_trespetit%d.nii.gz' %(index_image)
        img_sagittal_mask_without_movment = nib.load(sagittal_mask_without_movment)
        
        listnomvt = []
        output=convert2Slices(img_axial_without_movment,img_axial_mask_without_movment,[],0,0)
        listnomvt.extend(output)
        output=convert2Slices(img_coronal_without_movment,img_coronal_mask_without_movment,[],1,1)
        listnomvt.extend(output)
        output=convert2Slices(img_sagittal_without_movment,img_sagittal_mask_without_movment,[],2,2)
        listnomvt.extend(output)
        
        #print(len(listSlice),len(listnomvt))
        img,msk=separate_slices_in_stacks(listSlice)
        img2,msk2=separate_slices_in_stacks(listnomvt)
        print(len(img[0]),len(img2[0]),len(img[1]),len(img2[1]),len(img[2]),len(img2[2]))
        listerrorimg1img2_after = [feature.get_error() for feature in listFeature]     
        errorlist.extend(listerrorimg1img2_after) 
   
        for i_slice in listSlice:
                i_slice.set_parameters([0,0,0,0,0,0])
        tre_for_each_slices(listnomvt,listSlice,listFeature,transfo,rejected_slices)
        error_before=[feature.get_error() for feature in listFeature]
        error_before_correction_reallysmall.append(error_before)
            
            
        
            #register image to a reference image to compute PSNR, using ants
    if (os.path.exists(file+'/recon_subject_space/srr_subject.nii.gz') and os.path.exists(file+'/recon_subject_space/srr_subject_mask.nii.gz')):
                
            reference_image = ants.image_read('~/Chloe/DHCP/image_%d.nii.gz' %(index_image)) #je récupère l'image originale qui correspond à mon image reconstruite
            reference_mask = ants.image_read('~/Chloe/DHCP/binmask_%d.nii.gz' %(index_image)) #le mask de l'image originale

            moving_image = ants.image_read(file+'/recon_subject_space/srr_subject.nii.gz')

            moving_mask = ants.image_read(file+'/recon_subject_space/srr_subject_mask.nii.gz')

            transform = ants.registration(reference_image,moving_image,'Rigid')
            moving_image_registered = ants.apply_transforms(reference_image,moving_image,transformlist=transform['fwdtransforms'])
            mask_registered = ants.apply_transforms(reference_mask,moving_mask,transformlist=transform['fwdtransforms'])
            
                #convert image from ants to numpy, in order to compute PSNR
            numpy_moving_image = moving_image_registered.numpy()
            numpy_moving_mask = mask_registered.numpy()
            numpy_reference_image = reference_image.numpy()
            numpy_reference_mask = reference_mask.numpy()
                
            numpy_reference_image = numpy_reference_image*numpy_reference_mask
            numpy_moving_image = numpy_moving_image*numpy_moving_mask
            
                #Compute PSNR and SSIM
            psnrval = PSNR(numpy_reference_image,numpy_moving_image)
            psnrlist.append(psnrval)
            
                #psnrlist.append(psnr)
            ssim_res = SSIM(numpy_reference_image,numpy_moving_image)
            ssimlist.append(ssim_res)
                #list_ssim_inter.append(ssim_res)
            
tres_petit_psnr.append(psnrlist)
tres_petit_ssim.append(ssimlist)
error_after_correction_reallysmall.append(errorlist)
    
errorlist=[]
psnrlist=[]
ssimlist=[] 
error_before=[]
for index_image in range(1,6):
        
    file='../../res/epsilon_evolution/value%s/simul_data/Petit%d/epsilon_evolution%s/epsilon_evolution%s' %(value_epsilon_evolution,index_image,value_epsilon_evolution,value_epsilon_evolution)
    petit = '../../res/epsilon_evolution/value%s/simul_data/Petit%d/epsilon_evolution%s/res_test_epsilon_evolution%s.joblib.gz' %(value_epsilon_evolution,index_image,value_epsilon_evolution,value_epsilon_evolution)
    #directory1 = '/home/mercier/Documents/donnee/test/Grand%d/' %(i)
    directory = '../../simu/Petit%d/' %(index_image)
    #print(petit)
    
    if os.path.exists(petit)  : 
            
        print(petit)
    
        res = joblib.load(petit)
        key=[p[0] for p in res]
        element=[p[1] for p in res]
        listSlice=element[key.index('listSlice')]
        parameters_slices = element[key.index('EvolutionParameters')][-1,:,:]
        listFeature=element[key.index('ListError')]
        
        rejected_slices = element[key.index('RejectedSlices')]
     
        nb_rejected_slices = np.shape(rejected_slices)[0]
        nb_slices = len(listSlice)
        #proportion_rejected_slices_small[i-1] = nb_rejected_slices/nb_slices
    
        transfo_axial =  directory + 'transfoAx_petit%d.npy' %(index_image)
        transfo_sagittal  = directory + 'transfoSag_petit%d.npy' %(index_image)
        transfo_coronal = directory + 'transfoCor_petit%d.npy' %(index_image)
        transfo = np.array([transfo_axial,transfo_coronal,transfo_sagittal])
        
        axial_without_movment = directory + 'LrAxNifti_nomvt.nii.gz'
        img_axial_without_movment = nib.load(axial_without_movment)
        axial_mask_without_movment = directory + 'brain_mask/LrAxNifti_petit%d.nii.gz' %(index_image)
        img_axial_mask_without_movment = nib.load(axial_mask_without_movment)
        coronal_without_movment = directory + 'LrCorNifti_nomvt.nii.gz'
        img_coronal_without_movment = nib.load(coronal_without_movment)
        coronal_mask_without_movment = directory + 'brain_mask/LrCorNifti_petit%d.nii.gz' %(index_image)
        img_coronal_mask_without_movment = nib.load(coronal_mask_without_movment)
        sagittal_without_movment = directory + 'LrSagNifti_nomvt.nii.gz'
        img_sagittal_without_movment = nib.load(sagittal_without_movment)
        sagittal_mask_without_movment = directory + 'brain_mask/LrSagNifti_petit%d.nii.gz' %(index_image)
        img_sagittal_mask_without_movment = nib.load(sagittal_mask_without_movment)
        
        listnomvt = []
        output=convert2Slices(img_axial_without_movment,img_axial_mask_without_movment,[],0,0)
        listnomvt.extend(output)
        output=convert2Slices(img_coronal_without_movment,img_coronal_mask_without_movment,[],1,1)
        listnomvt.extend(output)
        output=convert2Slices(img_sagittal_without_movment,img_sagittal_mask_without_movment,[],2,2)
        listnomvt.extend(output)

        listerrorimg1img2_after = [feature.get_error() for feature in listFeature]
        errorlist.extend(listerrorimg1img2_after)

        
        for i_slice in listSlice:
                i_slice.set_parameters([0,0,0,0,0,0])
        tre_for_each_slices(listnomvt,listSlice,listFeature,transfo,rejected_slices)
        error_before=[feature.get_error() for feature in listFeature]
        error_before_correction_small.append(error_before)

        reference_image = ants.image_read('~/Chloe/DHCP/image_%d.nii.gz' %(index_image)) #je récupère l'image originale qui correspond à mon image reconstruite
        reference_mask = ants.image_read('~/Chloe/DHCP/binmask_%d.nii.gz' %(index_image)) #le mask de l'image originale
        
        #register image to a reference image to compute PSNR, using ants            
        if os.path.exists(file+'/recon_subject_space/srr_subject.nii.gz') and os.path.exists(file+'/recon_subject_space/srr_subject_mask.nii.gz'):
                
            moving_image = ants.image_read(file+'/recon_subject_space/srr_subject.nii.gz')

            moving_mask = ants.image_read(file+'/recon_subject_space/srr_subject_mask.nii.gz')
        
            transform = ants.registration(reference_image,moving_image,'Rigid')
            moving_image_registered = ants.apply_transforms(reference_image,moving_image,transformlist=transform['fwdtransforms'])
            mask_registered = ants.apply_transforms(reference_mask,moving_mask,transformlist=transform['fwdtransforms'])
            
                #convert image from ants to numpy, in order to compute PSNR
            numpy_moving_image = moving_image_registered.numpy()
            numpy_moving_mask = mask_registered.numpy()
            numpy_reference_image = reference_image.numpy()
            numpy_reference_mask = reference_mask.numpy()
                
            numpy_reference_image = numpy_reference_image*numpy_reference_mask
            numpy_moving_image = numpy_moving_image*numpy_moving_mask
            
                #Compute PSNR and SSIM
            psnr = PSNR(numpy_reference_image,numpy_moving_image)
            psnrlist.append(psnr)
            
            ssim_res = SSIM(numpy_reference_image,numpy_moving_image)
            ssimlist.append(ssim_res)
        
petit_psnr.append(psnrlist)
petit_ssim.append(ssimlist)
error_after_correction_small.append(errorlist)   
    

errorlist=[]
psnrlist=[]
ssimlist=[] 
for index_image in range(1,6):
        
    file = '../../res/epsilon_evolution/value%s/simul_data/Moyen%d/epsilon_evolution%s/epsilon_evolution%s' %(value_epsilon_evolution,index_image,value_epsilon_evolution,value_epsilon_evolution)
    petit = '../../res/epsilon_evolution/value%s/simul_data/Moyen%d/epsilon_evolution%s/res_test_epsilon_evolution%s.joblib.gz' %(value_epsilon_evolution,index_image,value_epsilon_evolution,value_epsilon_evolution)
    #directory1 = '/home/mercier/Documents/donnee/test/Grand%d/' %(i)
    directory = '../../simu/Moyen%d/' %(index_image)
    print(os.path.exists(petit))
    
    if os.path.exists(petit) : 
          
            
        print(petit)
    
        res = joblib.load(open(petit,'rb'))
        key=[p[0] for p in res]
        element=[p[1] for p in res]
        listSlice=element[key.index('listSlice')]
        listFeature=element[key.index('ListError')]
        parameters_slices = element[key.index('EvolutionParameters')][-1,:,:]
        
        rejected_slices = element[key.index('RejectedSlices')]
     
        nb_rejected_slices = np.shape(rejected_slices)[0]
        nb_slices = len(listSlice)
        #proportion_rejected_slices_small[i-1] = nb_rejected_slices/nb_slices

        transfo_axial =  directory + 'transfoAx_moyen%d.npy' %(index_image)
        transfo_sagittal  = directory + 'transfoSag_moyen%d.npy' %(index_image)
        transfo_coronal = directory + 'transfoCor_moyen%d.npy' %(index_image)
        transfo = np.array([transfo_axial,transfo_coronal,transfo_sagittal])
        
        axial_without_movment = directory + 'LrAxNifti_nomvt.nii.gz'
        img_axial_without_movment = nib.load(axial_without_movment)
        axial_mask_without_movment = directory + 'brain_mask/LrAxNifti_moyen%d.nii.gz' %(index_image)
        img_axial_mask_without_movment = nib.load(axial_mask_without_movment)
        coronal_without_movment = directory + 'LrCorNifti_nomvt.nii.gz'
        img_coronal_without_movment = nib.load(coronal_without_movment)
        coronal_mask_without_movment = directory + 'brain_mask/LrCorNifti_moyen%d.nii.gz' %(index_image)
        img_coronal_mask_without_movment = nib.load(coronal_mask_without_movment)
        sagittal_without_movment = directory + 'LrSagNifti_nomvt.nii.gz'
        img_sagittal_without_movment = nib.load(sagittal_without_movment)
        sagittal_mask_without_movment = directory + 'brain_mask/LrSagNifti_moyen%d.nii.gz' %(index_image)
        img_sagittal_mask_without_movment = nib.load(sagittal_mask_without_movment)
        
        listnomvt = []
        output=convert2Slices(img_axial_without_movment,img_axial_mask_without_movment,[],0,0)
        listnomvt.extend(output)
        output=convert2Slices(img_coronal_without_movment,img_coronal_mask_without_movment,[],1,1)
        listnomvt.extend(output)
        output=convert2Slices(img_sagittal_without_movment,img_sagittal_mask_without_movment,[],2,2)
        listnomvt.extend(output)
        
        listerrorimg1img2_after = [feature.get_error() for feature in listFeature]
        errorlist.extend(listerrorimg1img2_after) 

        
        for i_slice in listSlice:
                i_slice.set_parameters([0,0,0,0,0,0])

        tre_for_each_slices(listnomvt,listSlice,listFeature,transfo,rejected_slices)
        error_before=[feature.get_error() for feature in listFeature]
        error_before_correction_medium.append(error_before)
            
        reference_image = ants.image_read('~/Chloe/DHCP/image_%d.nii.gz' %(index_image)) #je récupère l'image originale qui correspond à mon image reconstruite
        reference_mask = ants.image_read('~/Chloe/DHCP/binmask_%d.nii.gz' %(index_image)) #le mask de l'image originale
        
        #register image to a reference image to compute PSNR, using ants
            
        if os.path.exists(file+'/recon_subject_space/srr_subject.nii.gz') and os.path.exists(file+'/recon_subject_space/srr_subject.nii.gz'):
            
            moving_image = ants.image_read(file+'/recon_subject_space/srr_subject.nii.gz')

            moving_mask = ants.image_read(file+'/recon_subject_space/srr_subject_mask.nii.gz')
        
            transform = ants.registration(reference_image,moving_image,'Rigid')
            moving_image_registered = ants.apply_transforms(reference_image,moving_image,transformlist=transform['fwdtransforms'])
            mask_registered = ants.apply_transforms(reference_mask,moving_mask,transformlist=transform['fwdtransforms'])
            
            #convert image from ants to numpy, in order to compute PSNR
            numpy_moving_image = moving_image_registered.numpy()
            numpy_moving_mask = mask_registered.numpy()
            numpy_reference_image = reference_image.numpy()
            numpy_reference_mask = reference_mask.numpy()
                
            numpy_reference_image = numpy_reference_image*numpy_reference_mask
            numpy_moving_image = numpy_moving_image*numpy_moving_mask
            
            #Compute PSNR and SSIM
            psnr = PSNR(numpy_reference_image,numpy_moving_image)
            psnrlist.append(psnr)
            
            ssim_res = SSIM(numpy_reference_image,numpy_moving_image)
            ssimlist.append(ssim_res)
        
moyen_psnr.append(psnrlist)
moyen_ssim.append(ssimlist)
error_after_correction_medium.append(errorlist)   


errorlist=[]
psnrlist=[]
ssimlist=[] 
for index_image in range(1,6):

    file = '../../res/epsilon_evolution/value%s/simul_data/Grand%d/epsilon_evolution%s/epsilon_evolution%s' %(value_epsilon_evolution,index_image,value_epsilon_evolution,value_epsilon_evolution)
    petit = '../../res/epsilon_evolution/value%s/simul_data/Grand%d/epsilon_evolution%s/res_test_epsilon_evolution%s.joblib.gz' %(value_epsilon_evolution,index_image,value_epsilon_evolution,value_epsilon_evolution)
    #directory1 = '/home/mercier/Documents/donnee/test/Grand%d/' %(i)
    directory = '../../simu/Grand%d/' %(index_image)
    print(petit,os.path.exists(petit))
    
    if os.path.exists(petit): 
          
            
        #print(petit)
    
        res = joblib.load(open(petit,'rb'))
        key=[p[0] for p in res]
        element=[p[1] for p in res]
        listSlice=element[key.index('listSlice')]
        listFeature=element[key.index('ListError')]

        parameters_slices = element[key.index('EvolutionParameters')][-1,:,:]
        
        rejected_slices = element[key.index('RejectedSlices')]
     
        nb_rejected_slices = np.shape(rejected_slices)[0]
        nb_slices = len(listSlice)
        #proportion_rejected_slices_small[i-1] = nb_rejected_slices/nb_slices

        transfo_axial =  directory + 'transfoAx_grand%d.npy' %(index_image)
        transfo_sagittal  = directory + 'transfoSag_grand%d.npy' %(index_image)
        transfo_coronal = directory + 'transfoCor_grand%d.npy' %(index_image)
        transfo = np.array([transfo_axial,transfo_coronal,transfo_sagittal])
        
        axial_without_movment = directory + 'LrAxNifti_nomvt.nii.gz'
        img_axial_without_movment = nib.load(axial_without_movment)
        axial_mask_without_movment = directory + 'brain_mask/LrAxNifti_grand%d.nii.gz' %(index_image)
        img_axial_mask_without_movment = nib.load(axial_mask_without_movment)
        coronal_without_movment = directory + 'LrCorNifti_nomvt.nii.gz'
        img_coronal_without_movment = nib.load(coronal_without_movment)
        coronal_mask_without_movment = directory + 'brain_mask/LrCorNifti_grand%d.nii.gz' %(index_image)
        img_coronal_mask_without_movment = nib.load(coronal_mask_without_movment)
        sagittal_without_movment = directory + 'LrSagNifti_nomvt.nii.gz'
        img_sagittal_without_movment = nib.load(sagittal_without_movment)
        sagittal_mask_without_movment = directory + 'brain_mask/LrSagNifti_grand%d.nii.gz' %(index_image)
        img_sagittal_mask_without_movment = nib.load(sagittal_mask_without_movment)
        
        listnomvt = []
        output = convert2Slices(img_axial_without_movment,img_axial_mask_without_movment,[],0,0)
        listnomvt.extend(output)
        output =convert2Slices(img_coronal_without_movment,img_coronal_mask_without_movment,[],1,1)
        listnomvt.extend(output)
        output = convert2Slices(img_sagittal_without_movment,img_sagittal_mask_without_movment,[],2,2)
        listnomvt.extend(output)
        
        
        listerrorimg1img2_after = [feature.get_error() for feature in listFeature]
        errorlist.extend(listerrorimg1img2_after) 

        
        for i_slice in listSlice:
                i_slice.set_parameters([0,0,0,0,0,0])
        tre_for_each_slices(listnomvt,listSlice,listFeature,transfo,rejected_slices)
        error_before=[feature.get_error() for feature in listFeature]
        error_before_correction_grand.append(error_before)
        #error_before_correction_grand.append(error_before)

        reference_image = ants.image_read('~/Chloe/DHCP/image_%d.nii.gz' %(index_image)) #je récupère l'image originale qui correspond à mon image reconstruite
        reference_mask = ants.image_read('~/Chloe/DHCP/binmask_%d.nii.gz' %(index_image)) #le mask de l'image originale

        if os.path.exists(file+'/recon_subject_space/srr_subject.nii.gz'):
                
                #register image to a reference image to compute PSNR, using ants
            moving_image = ants.image_read(file+'/recon_subject_space/srr_subject.nii.gz')

            moving_mask = ants.image_read(file+'/recon_subject_space/srr_subject_mask.nii.gz')
            
            transform = ants.registration(reference_image,moving_image,'Rigid')
            moving_image_registered = ants.apply_transforms(reference_image,moving_image,transformlist=transform['fwdtransforms'])
            mask_registered = ants.apply_transforms(reference_mask,moving_mask,transformlist=transform['fwdtransforms'])
            
                #convert image from ants to numpy, in order to compute PSNR
            numpy_moving_image = moving_image_registered.numpy()
            numpy_moving_mask = mask_registered.numpy()
            numpy_reference_image = reference_image.numpy()
            numpy_reference_mask = reference_mask.numpy()
                
            numpy_reference_image = numpy_reference_image*numpy_reference_mask
            numpy_moving_image = numpy_moving_image*numpy_moving_mask
            
            #Compute PSNR and SSIM
            psnr = PSNR(numpy_reference_image,numpy_moving_image)
            psnrlist.append(psnr)
            
            ssim_res = SSIM(numpy_reference_image,numpy_moving_image)
            ssimlist.append(ssim_res)
        
    grand_psnr.append(psnrlist)
    grand_ssim.append(ssimlist)
    error_after_correction_grand.append(errorlist)

#fig,axs = plt.subplots(1,1, figsize=(40, 15)) 
fig = plt.figure(figsize=(10,15))
ax1 = fig.add_subplot(111)
color = ['blue','orange','green','red']
#couleur=['green','red','blue','yellow']  
motionRange=['small','medium','large','extra-large']     


print('test :',value_epsilon_evolution)
        
for mvt in range(0,4):
    print("motion",mvt)
    if mvt==0:
        datalist=error_after_correction_reallysmall[0]
        before=np.concatenate(error_before_correction_reallysmall)
    elif mvt==1:
        datalist=error_after_correction_small[0]
        before=np.concatenate(error_before_correction_small)
    elif mvt==2:
        datalist=error_after_correction_medium[0]
        before=np.concatenate(error_before_correction_medium)
    else :
        datalist=error_after_correction_grand[0]
        before=np.concatenate(error_before_correction_grand) 
            
            #data=datalist[value_epsilon_evolution]
        
                #print(len(before))
                #print(len(data))
                
                #add error from each epsilon_evolution test
    nous_before = np.array(before)
    data = np.array(datalist)
    print(len(nous_before),len(data))
    if len(data)==len(nous_before):
                    
        #axs[value_epsilon_evolution].
        ax1.scatter(data,nous_before,marker='.',s=170,alpha=0.1,c=color[mvt])
                #axs[value_epsilon_evolution].scatter(ebner_petit, ebner_before, c='blue', marker='.', s=170)
                 #axs[mvt].set_yticks(np.arange(min, max),fontsize=100) 
                #axs[value_epsilon_evolution].
        plt.ylabel('avant recalage',fontsize=30)
                #axs[value_epsilon_evolution].
        plt.xlabel('après recalage',fontsize=30)
                #axs[value_epsilon_evolution].
        plt.title('epsilon=0.5/8',fontsize=30) 
                #axs[value_epsilon_evolution]
        ax1.set_ylim(0,16)
                #axs[value_epsilon_evolution]
        ax1.set_xlim(0,5)

        #for tick in axs[value_epsilon_evolution].xaxis.get_majorticklabels():  # example for xaxis
                #plt.set_fontsize(15) 

        #for tick in axs[value_epsilon_evolution].yaxis.get_majorticklabels():  # example for xaxis
                #plt.set_fontsize(15) 

plt.tight_layout()

plt.legend(["small","medium","large","extra-large"], fontsize=30, loc ="lower right")

plt.savefig('epsilon_evolution_test.png')

for i in range(0,len(value_epsilon_evolution)):
    
    print('epsilon_evolution :',value_epsilon_evolution[i])
    print('PSNR')
    print('mean :',np.mean(tres_petit_psnr[i]))
    print('std :',np.std(tres_petit_psnr[i]))
    print('mean :',np.mean(petit_psnr[i]))
    print('std :',np.std(petit_psnr[i]))
    print('mean :',np.mean(moyen_psnr[i]))
    print('std :',np.std(moyen_psnr[i]))
    print('mean :',np.mean(grand_psnr[i]))
    print('std :',np.std(grand_psnr[i]))

    print('SSIM')
    print('mean :',np.mean(tres_petit_ssim[i]))
    print('std :',np.std(tres_petit_ssim[i]))
    print('mean :',np.mean(petit_ssim[i]))
    print('std :',np.std(petit_ssim[i]))
    print('mean :',np.mean(moyen_ssim[i]))
    print('std :',np.std(moyen_ssim[i]))
    print('mean :',np.mean(grand_ssim[i]))
    print('std :',np.std(grand_ssim[i]))