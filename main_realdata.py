#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:24:49 2022

@author: mercier
"""
import numpy as np
import time
import os
from registration import loadSlice,loadimages, normalization,computeCostBetweenAll2Dimages,costFromMatrix,global_optimization
import nibabel as nib
from input_argparser import InputArgparser

if __name__ == '__main__':
    
    #input arguments :
    input_parser = InputArgparser()
    input_parser.add_filenames(required=True)
    input_parser.add_filenames_masks()
    input_parser.add_dir_output(required=True)
    args = input_parser.parse_args()
    
    start = time.time()
    costGlobal = np.zeros(2)
    
    #loading image
    listSlice = []
    #Load images, mask and create a list of slices from the images
    for i in range(len(args.filenames)):
        im, inmask = loadimages(args.filenames[i],args.filenames_masks[i])
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
    
    file = args.dir_output
    if not os.path.exists(file):
        os.makedirs(file)
    
    nbit = len(ErrorEvolution)
    nbSlice=len(listSlice)
    
    strEE = file + '/ErrorEvolution.npz'
    np.savez_compressed(strEE,ErrorEvolution)
    
    strED = file + '/DiceEvolution.npz'
    np.savez_compressed(strED,DiceEvolution)
    
    strEGE = file + '/EvolutionGridError.npz'
    EvolutionGridError = np.reshape(EvolutionGridError,[nbit,nbSlice,nbSlice])
    np.savez_compressed(strEGE,EvolutionGridError)
    
    strEGN = file + '/EvolutionGridNbpoint.npz'
    EvolutionGridNbpoint = np.reshape(EvolutionGridNbpoint,[nbit,nbSlice,nbSlice])
    np.savez_compressed(strEGN,EvolutionGridNbpoint)
    
    strEGI = file + '/EvolutionGridInter.npz'
    EvolutionGridInter = np.reshape(EvolutionGridInter,[nbit,nbSlice,nbSlice])
    np.savez_compressed(strEGI,EvolutionGridInter)
    
    strEGU = file + '/EvolutionGridUnion.npz'
    EvolutionGridUnion = np.reshape(EvolutionGridUnion,[nbit,nbSlice,nbSlice])
    np.savez_compressed(strEGU,EvolutionGridUnion)
    
    strEP = file + '/EvolutionParameters.npz'
    EvolutionParameters = np.reshape(EvolutionParameters,[nbit,nbSlice,6])
    np.savez_compressed(strEP,EvolutionParameters)
    
    strET = file + '/EvolutionTransfo.npz'
    EvolutionTransfo = np.reshape(EvolutionTransfo,[nbit,nbSlice,4,4])
    np.savez_compressed(strET,EvolutionTransfo)
    strCG = file + '/CostGlobal.npz'
    costGlobal.tofile(strCG)
    
    end = time.time()
    elapsed = end - start
    print(f'Temps d\'exécution : {elapsed}')
