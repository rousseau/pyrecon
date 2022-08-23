#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:29:22 2022

@author: mercier
""" 
import numpy as np
import time
from data_simulation import findCommonPointbtw2V,  ErrorOfRegistrationBtw2Slice, ChamferDistance, createArrayOfChamferDistance
from registration import normalization,global_optimization
from load import loadSlice,loadimages
from input_argparser import InputArgparser
import joblib
from os import getcwd
from tools import createVolumesFromAlist



if __name__ == '__main__':
        
    root=getcwd()
    
    input_parser = InputArgparser()
    
    
    input_parser.add_filenames(required=True) #load images
    input_parser.add_filenames_masks() #load masks
    input_parser.add_simulation() #load simulated transformation
    input_parser.add_output(required=True) #output name
    input_parser.add_nomvt() #load images with no movement
    
    args = input_parser.parse_args()
    
    start = time.time()
    costGlobal = np.zeros(3)
    
    
    listSlice = [] 
    listnomvt = []
    
    #Create a list of slices from the images
    for i in range(len(args.filenames)):
        im, inmask = loadimages(args.filenames[i], args.filenames_masks[i]) 
        loadSlice(im, inmask, listSlice, i)
        
        im, inmask = loadimages(args.nomvt[i], args.filenames_masks[i])
        loadSlice(im, inmask, listnomvt, i)

    #normalize the data with a standart distribution
    listSlice = normalization(listSlice)
    
    #Algorithm of motion correction
    dicRes, rejectedSlices = global_optimization(listSlice) 
    
    #result of registration
    ErrorEvolution=dicRes["evolutionerror"]
    DiceEvolution=dicRes["evolutiondice"]
    nbit = len(ErrorEvolution)
    nbSlice=len(listSlice)
    EvolutionGridError = np.reshape(dicRes["evolutiongriderror"],[nbit,nbSlice,nbSlice])
    EvolutionGridNbpoint = np.reshape(dicRes["evolutiongridnbpoint"],[nbit,nbSlice,nbSlice])
    EvolutionGridInter = np.reshape(dicRes["evolutiongridinter"],[nbit,nbSlice,nbSlice])
    EvolutionGridUnion = np.reshape(dicRes["evolutiongridunion"],[nbit,nbSlice,nbSlice])
    EvolutionParameters = np.reshape(dicRes["evolutionparameters"],[nbit,nbSlice,6])
    EvolutionTransfo = np.reshape(dicRes["evolutiontransfo"],[nbit,nbSlice,4,4])
    
    images,mask = createVolumesFromAlist(listnomvt.copy())
    listptimg1img2_img1=[];listptimg1img2_img2=[]
    for i1 in range(len(images)):
        for i2 in range(len(images)):
            if i1 < i2:
               ptimg1img2_img1, ptimg1img2_img2 = findCommonPointbtw2V(images[i1],images[i2],rejectedSlices) #common points between volumes when no movement
               listptimg1img2_img1.append(ptimg1img2_img1)
               listptimg1img2_img2.append(ptimg1img2_img2)
    
    transfo = args.simulation #the simulated transformation
    
    images,mask = createVolumesFromAlist(listnomvt.copy()) 
    
    listerrorimg1img2_before=[];
    i=0
    for i1 in range(len(images)):
        for i2 in range(len(images)):
            if i1 < i2:
               transfo1 = np.load(transfo[i1])
               transfo2 = np.load(transfo[i2])
               #error of registration between volumes before registration
               errorimg1img2_before = ErrorOfRegistrationBtw2Slice(listptimg1img2_img1[i],listptimg1img2_img2[i],images[i1],images[i2],transfo1,transfo2)
               listerrorimg1img2_before.append(errorimg1img2_before)
               i=i+1

    listCorrected=[]
    for i_slice in range(nbSlice):
        s=listSlice[i_slice]
        x=EvolutionParameters[-1,i_slice,:]
        s.set_parameters(x)
        listCorrected.append(s) #list with the corrected parameters
        
    images_corrected, msk_corrected = createVolumesFromAlist(listCorrected) 
    
    

    
    i=0;listNameErrorBefore=[];listNameErrorAfter=[];listNameColorMap=[];listErrorAfter=[];listColorMap=[]
    for i1 in range(len(images)):
        for i2 in range(len(images)):
            if i1 < i2:
               transfo1 = np.load(transfo[i1])
               transfo2 = np.load(transfo[i2])
               
               #error of registration between volumes after registration
               errorimg1img2_after = ErrorOfRegistrationBtw2Slice(listptimg1img2_img1[i],listptimg1img2_img2[i],images_corrected[i1],images_corrected[i2],np.linalg.inv(transfo1),np.linalg.inv(transfo2))
               listErrorAfter.append(errorimg1img2_after)

               
               im, inmask = loadimages(args.filenames[i1], args.filenames_masks[i1])
               imgChamferDistancei1 = ChamferDistance(inmask)
               cimg1img2 = createArrayOfChamferDistance(imgChamferDistancei1,listptimg1img2_img1[i]) 
               listColorMap.append(cimg1img2) #the color of points depend of chamfer distance

               strEB='ErrorBefore%d%d' %(i1,i2)
               tupleEB=(strEB,listerrorimg1img2_before[i])
               listNameErrorBefore.append(tupleEB)
               
               strEA='ErrorAfter%d%d' %(i1,i2)
               tupleEA=(strEA,errorimg1img2_after)
               listNameErrorAfter.append(tupleEA)
               
               strC ='colormap%d%d' %(i1,i2)
               tupleC=(strC,cimg1img2)
               listNameColorMap.append(tupleC)
               
               i=i+1

    #save result in a joblib
    res_obj = [('listSlice',listSlice),('ErrorEvolution',ErrorEvolution), ('DiceEvolution',DiceEvolution), ('EvolutionGridError',EvolutionGridError), ('EvolutionGridNbpoint',EvolutionGridNbpoint), ('EvolutionGridInter',EvolutionGridInter), ('EvolutionGridUnion',EvolutionGridUnion), ('EvolutionParameters',EvolutionParameters),('EvolutionTransfo',EvolutionTransfo),('RejectedSlices',rejectedSlices)]
    res_obj.extend(listNameColorMap);res_obj.extend(listNameErrorBefore);res_obj.extend(listNameErrorAfter)
    joblib_name = root + '/' + args.output + '.joblib' + '.gz' 
    joblib.dump(res_obj,open(joblib_name,'wb'), compress=True)
    end = time.time()
    elapsed = end - start
    