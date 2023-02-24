#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:24:49 2022

@author: mercier
"""
import numpy as np
import time
from registration import normalization,computeCostBetweenAll2Dimages,costFromMatrix,global_optimisation
from load import loadSlice,loadimages
import nibabel as nib
from input_argparser import InputArgparser
import joblib
from os import getcwd, path, mkdir
from rec_ebner import convert2EbnerParam
from tools import image_center,center_image_2_ref

if __name__ == '__main__':
    
    root=getcwd()
    
    #input arguments :
    input_parser = InputArgparser()
    input_parser.add_filenames(required=True)
    input_parser.add_filenames_masks()
    input_parser.add_output(required=True)
    input_parser.add_ablation(required=True)
    input_parser.add_hyperparameters()
    args = input_parser.parse_args()
    
    
    
    start = time.time()
    costGlobal = np.zeros(2)
    
    list_prefixImage = []
    for string_name in args.filenames:
        name_file = string_name.split('/')[-1]
        name = name_file.replace('.nii.gz','')
        list_prefixImage.append(name)
    print('list of images :',list_prefixImage)
    
    
    
    #loading image
    listSlice = []
    #Load images, mask and create a list of slices from the images
    image_pre=[]
    for i in range(len(args.filenames)):
        im, inmask = loadimages(args.filenames[i],args.filenames_masks[i]) 
        Affine = im.affine
       
        datamask = inmask.get_fdata().squeeze()
        mask = nib.Nifti1Image(datamask, inmask.affine)
        
        if  i==0:
            loadSlice(im,mask,listSlice,0,i) 
            image_pre.append(im)
            center_ref = image_center(mask.get_fdata())
            ref_affine = im.affine
        else:
            new_affine = center_image_2_ref(im.get_fdata(),im.affine,center_ref,ref_affine)
            translation=new_affine[0:3,3]
            loadSlice(im,mask,listSlice,i,i)

           
   
    #normalize the data with a standart distribution
    listSlice = normalization(listSlice)
    listSlicessMvt=listSlice.copy()
    
    gridError, gridNbpoint, gridInter, gridUnion = computeCostBetweenAll2Dimages(listSlice)
    cost = costFromMatrix(gridError,gridNbpoint)
    costGlobal[0] = cost
    
    #Simulated data and Motion Correction
    ablation = args.ablation
    #Apply motion correction
    dicRes,rejectedSlices = global_optimisation(args.hyperparameters,listSlice,ablation) #Algorithm of motion correction
    ge_mvtCorrected,gn_mvtCorrected,gi_mvtCorrected,gu_mvtCorrected = computeCostBetweenAll2Dimages(listSlice) #Compute 2 grid for the MSE
    cost = costFromMatrix(ge_mvtCorrected,gn_mvtCorrected)
    costGlobal[1] = cost
    
    
    ErrorEvolution=dicRes["evolutionerror"]
    DiceEvolution=dicRes["evolutiondice"]
    nbit = len(ErrorEvolution)
    nbSlice=len(listSlice)
    
    
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

    res_obj = [('listSlice',listSlicessMvt),('ErrorEvolution',ErrorEvolution), ('DiceEvolution',DiceEvolution), ('EvolutionGridError',EvolutionGridError), ('EvolutionGridNbpoint',EvolutionGridNbpoint), ('EvolutionGridInter',EvolutionGridInter), ('EvolutionGridUnion',EvolutionGridUnion), ('EvolutionParameters',EvolutionParameters),('EvolutionTransfo',EvolutionTransfo),('RejectedSlices',rejectedSlices)]

    joblib_name = root + '/' + args.output + '.joblib' + '.gz' 
    joblib.dump(res_obj,open(joblib_name,'wb'), compress=True)
    
    end = time.time()
    elapsed = end - start
    print(f'Temps d\'exécution : {elapsed}')

    res = joblib.load(open(joblib_name,'rb'))
    key=[p[0] for p in res]
    element=[p[1] for p in res]
    listSlice=element[key.index('listSlice')]
    

    
    #save results for NiftyMIC
    parent_dir = getcwd() + '/'
    directory = args.output + '_mvt'
    path = path.join(parent_dir, directory)
    mkdir(path) 
    convert2EbnerParam(res,list_prefixImage,path)