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
from data_simulation import createMvt,findCommonPointbtw2V, create3VolumeFromAlist, ErrorOfRegistrationBtw2Slice, ChamferDistance, createArrayOfChamferDistance, createAnErrorImage
from registration import loadSlice,loadimages, normalization,computeCostBetweenAll2Dimages,costFromMatrix,optimization

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    #load images and masks (in the order, axial, sagittal, coronnal)
    parser.add_argument('-i','--input',help='Input Image',type=str,action='append')
    parser.add_argument('--imask',help='Mask of the images',type=str,action='append')
    parser.add_argument('--imvt',help='Range of motion simulation',type=int,action='append')
    parser.add_argument('-o','--output',help='Output Image',type=str,required = True)
    args = parser.parse_args()
    
    start = time.time()
    costGlobal = np.zeros(3)
    
    #loading image
    listSlice = []
    
    #Create a list of slices from the images
    Orientation = ['Axial','Coronal','Sagittal']
    for i in range(len(args.input)):
        im, inmask = loadimages(args.input[i], args.imask[i])
        loadSlice(im, inmask, listSlice, Orientation[i])
    
    #normalize the data with a standart distribution
    listSlice = normalization(listSlice)
    
    #Create a 3 list from of images corresponding to the three volumes
    sliceAx,sliceCor,sliceSag = create3VolumeFromAlist(listSlice)
    
    #Compute a list of point that will be used to validate the quality of the registration
    ptAxSag_Ax,ptAxSag_Sag = findCommonPointbtw2V(sliceAx,sliceSag) #list of point between Axial and Sagittal
    ptAxCor_Ax, ptAxCor_Cor = findCommonPointbtw2V(sliceAx,sliceCor) #list of point between Axial and Coronnal
    ptCorSag_Cor, ptCorSag_Sag =  findCommonPointbtw2V(sliceCor,sliceSag) #list of point Between Coronal and Sagittal
    
    
    #Simulated data, before motion correction
    listWithMvt = listSlice.copy() 
    m1 = args.imvt[0]
    mvtAngle = [-m1,m1]
    mvtTrans = [-m1,m1]
    motion_parameters = createMvt(listWithMvt, mvtAngle,mvtTrans) #Simulate a mvt of the slices
   
    
    SliceAxBeforeReg,SliceCorBeforeReg,SliceSagBeforeReg = create3VolumeFromAlist(listWithMvt) #Create 3 list of slies that represents the volumes, axial, coronal and sagittal from the listWithMvt (list with simulated motion)
    
    #Compute the error of registration before motion correction. It will be used for the grafical representation of the reconstruction error.
    #This errror represents the distance, in mm, between two point that are the same in the original data (ie the data without simulation)
    #It is expected to be higher distance computed after registration
    errorAxCor_before = ErrorOfRegistrationBtw2Slice(ptAxCor_Ax,ptAxCor_Cor,SliceAxBeforeReg,SliceCorBeforeReg)  #Error of registration between axial and coronal
    errorAxSag_before = ErrorOfRegistrationBtw2Slice(ptAxSag_Ax,ptAxSag_Sag,SliceAxBeforeReg,SliceSagBeforeReg)  #Error of registration between axial and sagittal
    errorCorSag_before = ErrorOfRegistrationBtw2Slice(ptCorSag_Cor,ptCorSag_Sag,SliceCorBeforeReg,SliceSagBeforeReg) #Error of registration between coronal and sagittal
    
    #Simulated data and Motion Correction
    ErrorEvolution,DiceEvolution,EvolutionGridError,EvolutionGridNbpoint,EvolutionGridInter,EvolutionGridUnion,EvolutionParameters,EvolutionTransfo = optimization(listWithMvt) #Algorithm of motion correction

    SliceAxAfterReg,SliceCorAfterReg,SliceSagAfterReg = create3VolumeFromAlist(listWithMvt) #Create 3 list of slices that represents the volume Axial, Coronal and Sagittal with simulation motion corrected by the algorithm
    
    #Error of registration after motion correction. It is expected to be smaller than the one before correction.
    errorAxCor_after = ErrorOfRegistrationBtw2Slice(ptAxCor_Ax,ptAxCor_Cor,SliceAxAfterReg,SliceCorAfterReg) #Error of registration between Axial and Coronal
    errorAxSag_after = ErrorOfRegistrationBtw2Slice(ptAxSag_Ax,ptAxSag_Sag,SliceAxAfterReg,SliceSagAfterReg) #Error of registration between Axial and Sagittal
    errorCorSag_after = ErrorOfRegistrationBtw2Slice(ptCorSag_Cor,ptCorSag_Sag,SliceCorAfterReg,SliceSagAfterReg) #Error of Registration between Coronal and Sagittal
    
    #Compute the Chamfer distance between the points considered when computed the error of registration. The chamfer distance corresponds to the distance between the center of an image and each points.
    axial = args.input[0] 
    axmask = args.imask[0]
    imax, inmask = loadimages(axial,axmask)
    imgChamferDistanceAx = ChamferDistance(inmask)
    caxcor = createArrayOfChamferDistance(imgChamferDistanceAx,ptAxCor_Ax)
    caxsag = createArrayOfChamferDistance(imgChamferDistanceAx,ptAxSag_Ax)
    
    coronal = args.input[1]
    cormask = args.imask[1]
    imcor, inmask = loadimages(coronal,cormask)
    imgChamferDistanceCor = ChamferDistance(inmask)
    ccorsag = createArrayOfChamferDistance(imgChamferDistanceCor,ptCorSag_Cor)
        
    file = args.output
    if not os.path.exists(file):
        os.makedirs(file)
    
    nbit = len(ErrorEvolution)
    nbSlice=len(listSlice)
    
    strEE = file + '/ErrorEvolution'
    #ErrorEvolution.tofile(strEE)
    np.savez_compressed(strEE,ErrorEvolution)
    
    strED = file + '/DiceEvolution'
    #DiceEvolution.tofile(strED)
    np.savez_compressed(strED,DiceEvolution)
    
    strEGE = file + '/EvolutionGridError'
    #EvolutionGridError.tofile(strEGE)
    EvolutionGridError = np.reshape(EvolutionGridError,[nbit,nbSlice,nbSlice])
    np.savez_compressed(strEGE,EvolutionGridError)
    
    strEGN = file + '/EvolutionGridNbpoint'
    #EvolutionGridNbpoint.tofile(strEGN)
    EvolutionGridNbpoint = np.reshape(EvolutionGridNbpoint,[nbit,nbSlice,nbSlice])
    np.savez_compressed(strEGN,EvolutionGridNbpoint)
    
    strEGI = file + '/EvolutionGridInter'
    #EvolutionGridInter.tofile(strEGI)
    EvolutionGridInter = np.reshape(EvolutionGridInter,[nbit,nbSlice,nbSlice])
    np.savez_compressed(strEGI,EvolutionGridInter)
    
    strEGU = file + '/EvolutionGridUnion'
    #EvolutionGridUnion.tofile(strEGU)
    EvolutionGridUnion = np.reshape(EvolutionGridUnion,[nbit,nbSlice,nbSlice])
    np.savez_compressed(strEGU,EvolutionGridUnion)
    
    strEP = file + '/EvolutionParameters'
    #EvolutionParameters.tofile(strEP)
    EvolutionParameters = np.reshape(EvolutionParameters,[nbit,nbSlice,6])
    np.savez_compressed(strEP,EvolutionParameters)
    
    strET = file + '/EvolutionTransfo'
    #EvolutionTransfo.tofile(strET)
    EvolutionTransfo = np.reshape(EvolutionTransfo,[nbit,nbSlice,4,4])
    np.savez_compressed(strET,EvolutionTransfo)
    
    strEBac = file + '/ErrorBeforeAxCor'
    #errorAxCor_before.tofile(strEBac)
    np.savez_compressed(strEBac,errorAxCor_before)
    
    strEBas = file + '/ErrorBeforeAxSag'
    #errorAxSag_before.tofile(strEBas)
    np.savez_compressed(strEBas,errorAxSag_before)
    
    strEBcs = file + '/ErrorBeforeCorSag'
    #errorCorSag_before.tofile(strEBcs)
    np.savez_compressed(strEBcs,errorCorSag_before)
    
    
    strEAac = file + '/ErrorAfterAxCor'
    #errorAxCor_after.tofile(strEAac)
    np.savez_compressed(strEAac,errorAxCor_after)
    
    strEAas = file + '/ErrorAfterAxSag'
    #errorAxSag_after.tofile(strEAas)
    np.savez_compressed(strEAas,errorAxSag_after)
    
    strEAcs = file + '/ErrorAfterCorSag'
    #errorCorSag_after.tofile(strEAcs)
    np.savez_compressed(strEAcs,errorCorSag_after)

    strCac = file + '/colormapAC'
    #caxcor.tofile(strCac)
    np.savez_compressed(strCac,caxcor)
    
    strCas = file +'/colormapAS'
    #caxsag.tofile(strCas)
    np.savez_compressed(strCas,caxsag)
    
    strCcs = file + '/colormapCS'
    #ccorsag.tofile(strCcs)
    np.savez_compressed(strCcs,ccorsag)
    
    end = time.time()
    elapsed = end - start
    print(f'Temps d\'exécution : {elapsed}')