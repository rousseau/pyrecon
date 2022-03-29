#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:24:49 2022

@author: mercier
"""
import numpy as np
import time
import os
import argparse
from registration import loadSlice,loadimages, normalization,computeCostBetweenAll2Dimages,costFromMatrix,global_optimization
import nibabel as nib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    #load images and masks (in the order, axial, sagittal, coronnal)
    parser.add_argument('-i','--input',help='Input Image',type=str,action='append')
    parser.add_argument('--imask',help='Mask of the images',type=str,action='append')
    parser.add_argument('--imvt',help='Range of motion simulation',type=int,action='append')
    parser.add_argument('-o','--output',help='Output Image',type=str,required = True)
    args = parser.parse_args()
    
    start = time.time()
    costGlobal = np.zeros(2)
    
    #loading image
    listSlice = []
    
    #Create a list of slices from the images
    #Orientation = ['Axial','Coronal','Sagittal']
    for i in range(len(args.input)):
        im, inmask = loadimages(args.input[i],args.imask[i])
        datamask = inmask.get_fdata().squeeze()
        mask = nib.Nifti1Image(datamask, inmask.affine)
        loadSlice(im, mask, listSlice, i)
    
    #normalize the data with a standart distribution
    listSlice = normalization(listSlice)
    
    gridError, gridNbpoint, gridInter, gridUnion = computeCostBetweenAll2Dimages(listSlice)
    cost = costFromMatrix(gridError,gridNbpoint)
    costGlobal[0] = cost
    
    #Simulated data and Motion Correction
    ErrorEvolution,DiceEvolution,EvolutionGridError,EvolutionGridNbpoint,EvolutionGridInter,EvolutionGridUnion,EvolutionParameters,EvolutionTransfo = global_optimization(listSlice) #Algorithm of motion correction
    ge_mvtCorrected,gn_mvtCorrected,gi_mvtCorrected,gu_mvtCorrected = computeCostBetweenAll2Dimages(listSlice) #Compute 2 grid for the MSE
    cost = costFromMatrix(ge_mvtCorrected,gn_mvtCorrected)
    costGlobal[1] = cost
    
    file = args.output
    if not os.path.exists(file):
        os.makedirs(file)
        
    strEE = file + '/ErrorEvolution.dat'
    ErrorEvolution.tofile(strEE)
    
    strED = file + '/DiceEvolution.dat'
    DiceEvolution.tofile(strED)
    
    strEGE = file + '/EvolutionGridError.dat'
    EvolutionGridError.tofile(strEGE)
    
    strEGN = file + '/EvolutionGridNbpoint.dat'
    EvolutionGridNbpoint.tofile(strEGN)
    
    strEGI = file + '/EvolutionGridInter.dat'
    EvolutionGridInter.tofile(strEGI)
    
    strEGU = file + '/EvolutionGridUnion.dat'
    EvolutionGridUnion.tofile(strEGU)
    
    strEP = file + '/EvolutionParameters.dat'
    EvolutionParameters.tofile(strEP)
    
    strET = file + '/EvolutionTranfo.dat'
    EvolutionTransfo.tofile(strET)
    
    strCG = file + '/CostGlobal.dat'
    costGlobal.tofile(strCG)
    
    end = time.time()
    elapsed = end - start
    print(f'Temps d\'exécution : {elapsed}')
