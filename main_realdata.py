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
import joblib
from os import getcwd

if __name__ == '__main__':
    
    root=getcwd()
    
    #input arguments :
    input_parser = InputArgparser()
    input_parser.add_filenames(required=True)
    input_parser.add_filenames_masks()
    input_parser.add_output(required=True)
    args = input_parser.parse_args()
    
    start = time.time()
    costGlobal = np.zeros(2)
    
    #loading image
    listSlice = []
    #Load images, mask and create a list of slices from the images
    for i in range(len(args.filenames)):
        im, inmask = loadimages(args.filenames[i],args.filenames_masks[i]) 
        Affine = im.affine
       
        datamask = inmask.get_fdata().squeeze()
        mask = nib.Nifti1Image(datamask, inmask.affine)
        # if  i==0:
        #     nx_img1=Affine[0:3,0].copy()
        #     ny_img1=Affine[0:3,1].copy()
        #     nz_img1=np.cross(nx_img1,ny_img1)
        #     nx_img1_norm=nx_img1/np.linalg.norm(nx_img1)
        #     ny_img1_norm=ny_img1/np.linalg.norm(ny_img1)
        #     nz_img1_norm=nz_img1/np.linalg.norm(nz_img1)
        #     loadSlice(im,mask,listSlice,0)
        #     print('nx_img1 :', nx_img1, 'ny_img1 :', ny_img1, 'nz_img1 :', nz_img1)
        #     print('nx_img1_norm :', nx_img1_norm,'ny_img1_norm', ny_img1_norm,'nz_img1_norm', nz_img1_norm)
        #     print(i,' : Axial')
            
        # else:
        #     nx=Affine[0:3,0].copy()
        #     ny=Affine[0:3,1].copy()
        #     nz=np.cross(nx,ny)
        #     nz_norm=nz/np.linalg.norm(nz)
        #     orx=np.sqrt((nz_norm[0]-nx_img1_norm[0])**2+(nz_norm[1]-nx_img1_norm[1])**2+(nz_norm[2]-nx_img1_norm[2])**2)
        #     ory=np.sqrt((nz_norm[0]-ny_img1_norm[0])**2+(nz_norm[1]-ny_img1_norm[1])**2+(nz_norm[2]-ny_img1_norm[2])**2)
        #     orz=np.sqrt((nz_norm[0]-nz_img1_norm[0])**2+(nz_norm[1]-nz_img1_norm[1])**2+(nz_norm[2]-nz_img1_norm[2])**2)
        #     if min(orx,ory,orz)==orx:
        #         loadSlice(im,mask,listSlice,1)
        #         print('orx :', orx, 'ory :', ory, 'orz :', orz)
        #         print(i, ' : Coronal')
                
        #     elif min(orx,ory,orz)==ory:
        #         loadSlice(im,mask,listSlice,2)
        #         print('orx :', orx, 'ory :', ory, 'orz :', orz)
        #         print(i ,' : Sagittal')
                
        #     else:
        #         loadSlice(im,mask,listSlice,0)
        #         print('orx :', orx, 'ory :', ory, 'orz :', orz)
        #         print(i , ' : Axial')
        loadSlice(im,mask,listSlice,i//2)
        print(i,i//2)
            
    
    #normalize the data with a standart distribution
    listSlice = normalization(listSlice)
    listSlicessMvt=listSlice.copy()
    
    gridError, gridNbpoint, gridInter, gridUnion = computeCostBetweenAll2Dimages(listSlice)
    cost = costFromMatrix(gridError,gridNbpoint)
    costGlobal[0] = cost
    
    #Simulated data and Motion Correction
    dicRes = global_optimization(listSlice) #Algorithm of motion correction
    ge_mvtCorrected,gn_mvtCorrected,gi_mvtCorrected,gu_mvtCorrected = computeCostBetweenAll2Dimages(listSlice) #Compute 2 grid for the MSE
    cost = costFromMatrix(ge_mvtCorrected,gn_mvtCorrected)
    costGlobal[1] = cost
    
    
    ErrorEvolution=dicRes["evolutionerror"]
    DiceEvolution=dicRes["evolutiondice"]
    nbit = len(ErrorEvolution)
    nbSlice=len(listSlice)
    
    #strEE = file + '/ErrorEvolution.npz'
    #np.savez_compressed(strEE,ErrorEvolution)
    
    #strED = file + '/DiceEvolution.npz'
    #np.savez_compressed(strED,DiceEvolution)
    
    #strEGE = file + '/EvolutionGridError.npz'
    EvolutionGridError = np.reshape(dicRes["evolutiongriderror"],[nbit,nbSlice,nbSlice])
    #np.savez_compressed(strEGE,EvolutionGridError)
    
    #strEGN = file + '/EvolutionGridNbpoint.npz'
    EvolutionGridNbpoint = np.reshape(dicRes["evolutiongridnbpoint"],[nbit,nbSlice,nbSlice])
    #np.savez_compressed(strEGN,EvolutionGridNbpoint)
    
    #strEGI = file + '/EvolutionGridInter.npz'
    EvolutionGridInter = np.reshape(dicRes["evolutiongridinter"],[nbit,nbSlice,nbSlice])
    #np.savez_compressed(strEGI,EvolutionGridInter)
    
    #strEGU = file + '/EvolutionGridUnion.npz'
    EvolutionGridUnion = np.reshape(dicRes["evolutiongridunion"],[nbit,nbSlice,nbSlice])
    #np.savez_compressed(strEGU,EvolutionGridUnion)
    
    #strEP = file + '/EvolutionParameters.npz'
    EvolutionParameters = np.reshape(dicRes["evolutionparameters"],[nbit,nbSlice,6])
    #np.savez_compressed(strEP,EvolutionParameters)
    
    #strET = file + '/EvolutionTransfo.npz'
    EvolutionTransfo = np.reshape(dicRes["evolutiontransfo"],[nbit,nbSlice,4,4])
    #np.savez_compressed(strET,EvolutionTransfo)
    #strCG = file + '/CostGlobal.npz'
    #costGlobal.tofile(strCG)
    
    res_obj = [('listSlice',listSlicessMvt),('ErrorEvolution',ErrorEvolution), ('DiceEvolution',DiceEvolution), ('EvolutionGridError',EvolutionGridError), ('EvolutionGridNbpoint',EvolutionGridNbpoint), ('EvolutionGridInter',EvolutionGridInter), ('EvolutionGridUnion',EvolutionGridUnion), ('EvolutionParameters',EvolutionParameters),('EvolutionTransfo',EvolutionTransfo)]

    joblib_name = root + '/' + args.output + '.joblib' + '.gz' 
    joblib.dump(res_obj,open(joblib_name,'wb'), compress=True)
    
    end = time.time()
    elapsed = end - start
    print(f'Temps d\'exécution : {elapsed}')
