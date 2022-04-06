#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:29:22 2022

@author: mercier
""" 
import numpy as np
import time
import os
import argparse
from data_simulation import createMvt,findCommonPointbtw2V,  ErrorOfRegistrationBtw2Slice, ChamferDistance, createArrayOfChamferDistance, createAnErrorImage
from registration import loadSlice,loadimages, normalization,computeCostBetweenAll2Dimages,costFromMatrix,global_optimization
from input_argparser import InputArgparser
import joblib
from os import getcwd
from tools import createVolumesFromAlist

if __name__ == '__main__':
    
    root=getcwd()
    
    input_parser = InputArgparser()
    
    #load images and masks (in the order, axial, sagittal, coronnal)
    input_parser.add_filenames(required=True)
    input_parser.add_filenames_masks()
    input_parser.add_simulation_angle()
    input_parser.add_simulation_translation()
    input_parser.add_output(required=True)
    
    args = input_parser.parse_args()
    
    start = time.time()
    costGlobal = np.zeros(3)
    
    #loading image
    listSlice = []
    
    #Create a list of slices from the images
    for i in range(len(args.filenames)):
        print(i)
        im, inmask = loadimages(args.filenames[i], args.filenames_masks[i])
        loadSlice(im, inmask, listSlice, i)

    #normalize the data with a standart distribution
    listSlice = normalization(listSlice)
    listSlicessmvt=listSlice.copy()
    images,mask = createVolumesFromAlist(listSlice.copy())
    listptimg1img2_img1=[];listptimg1img2_img2=[]
    for i1 in range(len(images)):
        for i2 in range(len(images)):
            if i1 < i2:
               print('i1 :', i1, 'i2 :', i2)
               ptimg1img2_img1, ptimg1img2_img2 = findCommonPointbtw2V(images[i1],images[i2]) #list of point between Axial and Sagittal
               listptimg1img2_img1.append(ptimg1img2_img1)
               listptimg1img2_img2.append(ptimg1img2_img2)
    
    nbimages=len(images)
    print(nbimages)
    print(len(listSlice))
    
    
    
    #Simulated data, before motion correction
    listWithMvt = listSlice.copy() 
    m1 = args.simulation_angle
    mvtAngle = [-m1,m1]
    m2 = args.simulation_translation
    mvtTrans = [-m2,m2]
    motion_parameters = createMvt(listWithMvt, mvtAngle,mvtTrans) #Simulate a mvt of the slices
   
    
    imagesmvt, mskmvt = createVolumesFromAlist(listWithMvt) #Create 3 list of slices that represents the volumes, axial, coronal and sagittal from the listWithMvt (list with simulated motion)
    
    listSlice = normalization(listSlice)
    images,mask = createVolumesFromAlist(listSlice.copy())
    listerrorimg1img2_before=[];
    i=0
    for i1 in range(len(images)):
        for i2 in range(len(images)):
            if i1 < i2:
               errorimg1img2_before = ErrorOfRegistrationBtw2Slice(listptimg1img2_img1[i],listptimg1img2_img2[i],imagesmvt[i1],imagesmvt[i2])
               print('error_image:', errorimg1img2_before)
               listerrorimg1img2_before.append(errorimg1img2_before)
               i=i+1
    
    #Simulated data and Motion Correction
    ErrorEvolution,DiceEvolution,EvolutionGridError,EvolutionGridNbpoint,EvolutionGridInter,EvolutionGridUnion,EvolutionParameters,EvolutionTransfo = global_optimization(listWithMvt) #Algorithm of motion correction
    
    file = args.output
    if not os.path.exists(file):
        os.makedirs(file)
        
    nbit = len(ErrorEvolution)
    nbSlice=len(listSlice)
    print(nbSlice)
    
    #strEE = file + '/ErrorEvolution'
    #np.savez_compressed(strEE,ErrorEvolution)
    
    #strED = file + '/DiceEvolution'
    #np.savez_compressed(strED,DiceEvolution)
    
    #strEGE = file + '/EvolutionGridError'
    EvolutionGridError = np.reshape(EvolutionGridError,[nbit,nbSlice,nbSlice])
    #np.savez_compressed(strEGE,EvolutionGridError)
    
    #strEGN = file + '/EvolutionGridNbpoint'
    EvolutionGridNbpoint = np.reshape(EvolutionGridNbpoint,[nbit,nbSlice,nbSlice])
    #np.savez_compressed(strEGN,EvolutionGridNbpoint)
    
    #strEGI = file + '/EvolutionGridInter'
    EvolutionGridInter = np.reshape(EvolutionGridInter,[nbit,nbSlice,nbSlice])
    #np.savez_compressed(strEGI,EvolutionGridInter)
    
    #strEGU = file + '/EvolutionGridUnion'
    EvolutionGridUnion = np.reshape(EvolutionGridUnion,[nbit,nbSlice,nbSlice])
    #np.savez_compressed(strEGU,EvolutionGridUnion)
    
    #strEP = file + '/EvolutionParameters'
    EvolutionParameters = np.reshape(EvolutionParameters,[nbit,nbSlice,6])
    #np.savez_compressed(strEP,EvolutionParameters)
    
    #strET = file + '/EvolutionTransfo'
    EvolutionTransfo = np.reshape(EvolutionTransfo,[nbit,nbSlice,4,4])
    #np.savez_compressed(strET,EvolutionTransfo)
    
    listCorrected=[]
    for i_slice in range(nbSlice):
        s=listSlice[i_slice]
        x=EvolutionParameters[nbit-1,i_slice,:]
        s.set_parameters(x)
        listCorrected.append(s)
        
    images_corrected, msk_corrected = createVolumesFromAlist(listCorrected) #Create 3 list of slices that represents the volume Axial, Coronal and Sagittal with simulation motion corrected by the algorithm
    
    #Error of registration after motion correction. It is expected to be smaller than the one before correction.
    
    #Compute the Chamfer distance between the points considered when computed the error of registration. The chamfer distance corresponds to the distance between the center of an image and each points.
    
    
    
    #Compute a list of point that will be used to validate the quality of the registration
    #Create a 3 list from of images corresponding to the three volumes
    
    i=0;listNameErrorBefore=[];listNameErrorAfter=[];listNameColorMap=[];listErrorAfter=[];listColorMap=[]
    for i1 in range(len(images)):
        for i2 in range(len(images)):
            if i1 < i2:
               
               #ptimg1img2_img1, ptimg1img2_img2 = findCommonPointbtw2V(images[i1],images[i2]) #list of point between Axial and Sagittal
              
               errorimg1img2_after = ErrorOfRegistrationBtw2Slice(listptimg1img2_img1[i],listptimg1img2_img2[i],images_corrected[i1],images_corrected[i2])
               listErrorAfter.append(errorimg1img2_after)
               print('error_image:', errorimg1img2_after)
               
               im, inmask = loadimages(args.filenames[i1], args.filenames_masks[i1])
               imgChamferDistancei1 = ChamferDistance(inmask)
               cimg1img2 = createArrayOfChamferDistance(imgChamferDistancei1,listptimg1img2_img1[i])
               listColorMap.append(cimg1img2)
               print('error_image:', listerrorimg1img2_before[i])
               
               strEB='ErrorBefore%d%d' %(i1,i2)
               tupleEB=(strEB,listerrorimg1img2_before[i])
               listNameErrorBefore.append(tupleEB)
               #np.savez_compressed(strEB,listerrorimg1img2_before[i])
               
               strEA='ErrorAfter%d%d' %(i1,i2)
               tupleEA=(strEA,errorimg1img2_after)
               listNameErrorAfter.append(tupleEA)
               #np.savez_compressed(strEA,errorimg1img2_after)
               
               strC ='colormap%d%d' %(i1,i2)
               tupleC=(strC,cimg1img2)
               listNameColorMap.append(tupleC)
               #np.savez_compressed(strC,cimg1img2)
               i=i+1

    res_obj = [('listSlice',listSlicessmvt),('ErrorEvolution',ErrorEvolution), ('DiceEvolution',DiceEvolution), ('EvolutionGridError',EvolutionGridError), ('EvolutionGridNbpoint',EvolutionGridNbpoint), ('EvolutionGridInter',EvolutionGridInter), ('EvolutionGridUnion',EvolutionGridUnion), ('EvolutionParameters',EvolutionParameters),('EvolutionTransfo',EvolutionTransfo)]
    res_obj.extend(listNameColorMap);res_obj.extend(listNameErrorBefore);res_obj.extend(listNameErrorAfter)
    print('key :', [p[0] for p in res_obj])
    joblib_name = root + '/' + args.output + '.joblib' + '.gz' 
    joblib.dump(res_obj,open(joblib_name,'wb'), compress=True)
    end = time.time()
    elapsed = end - start
    print(f'Temps d\'exécution : {elapsed}')