#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:23:38 2023

@author: mercier
"""

from input_argparser import InputArgparser
from os import getcwd, path, mkdir
import numpy as np
import registration_copy as re
import pickle
import joblib
from data_simulation import findCommonPointbtw2V,  ErrorOfRegistrationBtw2Slice,  createArrayOfChamferDistance, error_for_each_slices
import load
from outliers_detection_intersection import ErrorSlice
import nibabel as nib

if __name__ == '__main__':
    
    root=getcwd()
    
    #input arguments :
    input_parser = InputArgparser()
    input_parser.add_filenames(required=True)
    input_parser.add_output(required=True)
    input_parser.add_simulation() #load simulated transformation
    input_parser.add_nomvt() #load images with no movement
    input_parser.add_filenames_masks()
    args = input_parser.parse_args()
    
    # listnomvt=[] 
    # for i in range(len(args.filenames)):
    #     im, inmask = load.loadimages(args.nomvt[i], args.filenames_masks[i])
    #     load.loadSlice(im, inmask, listnomvt, i,i)
    
    transfo = args.simulation
    nomvt = args.nomvt
    mask = args.filenames_masks
    listnomvt=[]
    load.loadSlice(nib.load(nomvt[0]),nib.load(mask[0]),listnomvt,0,0)
    load.loadSlice(nib.load(nomvt[1]),nib.load(mask[1]),listnomvt,1,1)
    load.loadSlice(nib.load(nomvt[2]),nib.load(mask[2]),listnomvt,2,2)

    res = joblib.load(open(args.filenames[0],'rb'))
    key = [p[0] for p in res]
    element = [p[1] for p in res]
    
    loaded_model = pickle.load(open('my_newest_model.pickle', "rb"))
    
    ge = element[key.index('EvolutionGridError')][-1,:,:]
    gn = element[key.index('EvolutionGridNbpoint')][-1,:,:]
    gi = element[key.index('EvolutionGridInter')][-1,:,:]
    gu = element[key.index('EvolutionGridUnion')][-1,:,:]

    listSliceError = element[key.index('ListError')]
    e = [error.get_error() for error in listSliceError]
    #e = element[key.index('Error')]

    grid_slices = np.array([ge,gn,gi,gu])
   
    listSlice = element[key.index('listSlice')]  
    nbSlice = len(listSlice)
    
    dicRes={}
    dicRes["evolutionparameters"] =np.reshape(element[key.index('EvolutionParameters')][-1,:,:],-1).tolist() 
    dicRes["evolutiontransfo"] = np.reshape(element[key.index('EvolutionTransfo')][-1,:,:],(4*nbSlice,4)).tolist() 
    dicRes["evolutiongriderror"] = element[key.index('EvolutionGridError')][-1,:,:].tolist()  
    dicRes["evolutiongridnbpoint"] = element[key.index('EvolutionGridNbpoint')][-1,:,:].tolist() 
    dicRes["evolutiongridinter"] = element[key.index('EvolutionGridInter')][-1,:,:].tolist()  
    dicRes["evolutiongridunion"] = element[key.index('EvolutionGridUnion')][-1,:,:].tolist() 
    dicRes["evolutionerror"] = element[key.index('EvolutionGridError')][-1,:,:].tolist()
    dicRes["evolutionnbpoint"] = element[key.index('EvolutionGridNbpoint')][-1,:,:].tolist()
    dicRes["evolutionGridInter"] = element[key.index('EvolutionGridInter')][-1,:,:].tolist()
    dicRes["evolutionGridUnion"] = element[key.index('EvolutionGridUnion')][-1,:,:].tolist()
    costMse=re.costFromMatrix(ge, gn)
    print("Cost Before we start multi-start",costMse)
    costDice=re.costFromMatrix(gi,gu)
    dicRes["evolutionerror"] = [] 
    dicRes["evolutiondice"] = []
    dicRes["evolutionerror"].append(costMse)
    dicRes["evolutiondice"].append(costDice)
    
    re.update_feature(listSlice,listSliceError,ge,gn,gi,gu)
    
    hyperparameters = np.array([4,0.25,1e-10,2,1,0])
    
    new_hyperparameters = np.array([4,0.25,hyperparameters[2],np.sqrt(6*(hyperparameters[3]/8)**2),1,0])
    nbSlice=len(listSlice)
    set_o = np.zeros(nbSlice)
    
    #if ablation!='no_multistart':
    set_o1,set_o2 = re.detect_misregistered_slice(listSlice, grid_slices, loaded_model)
    set_r = np.logical_or(set_o1,set_o2)
    index = np.where(set_o1)
    index2=np.where(set_o2)
    e=np.array([e])
    e=e[0]
    print(e)
    print(index)
    print(e[index])
    print(e[index2])
    print(sum(set_o1),sum(set_o2),sum(set_r))
    print("badly register : ",set_r)
    #print(set_o)
    #ge,gn,gi,gu,dicRes=re.algo_optimisation(new_hyperparameters,listSlice,set_r,set_r,grid_slices,dicRes)  
    #grid_slices=np.array([ge,gn,gi,gu])
    #print(set_o)
    before = re.removeBadSlice(listSlice,set_o)
    #new_hyperparameters = np.array([hyperparameters[0],hyperparameters[1],hyperparameters[2],np.sqrt(6*hyperparameters[3]**2),hyperparameters[4],0])
    #grid_slices,set_r,dicRes=re.correction_out_images(listSlice,new_hyperparameters,set_o2.copy(),set_r,grid_slices,dicRes,thresholds)
    #set_o1,set_o2,thresholds = re.detect_misregistered_slice(listSlice, grid_slices, loaded_model)
    #set_r = np.logical_or(set_o1,set_o2) 
    #ge,gn,gi,gu=re.computeCostBetweenAll2Dimages(listSlice)
    #grid_slices=np.array([ge,gn,gi,gu])
    #ge,gn,gi,gu,dicRes=re.algo_optimisation(new_hyperparameters,listSlice,set_o1,set_o1,grid_slices,dicRes)  
    #grid_slices=np.array([ge,gn,gi,gu])
    grid_slices,set_r,dicRes=re.correction_out_images(listSlice,new_hyperparameters,set_o1.copy(),set_r,grid_slices,dicRes,thresholds,listSliceError)
    rejectedSlices=re.removeBadSlice(listSlice, set_r)
    set_o1,set_o2 = re.detect_misregistered_slice(listSlice, grid_slices, loaded_model)
    set_r = np.logical_or(set_o1,set_o2) 
    ge,gn,gi,gu=re.computeCostBetweenAll2Dimages(listSlice)
    grid_slice=np.array([ge,gn,gi,gu])
    ge,gn,gi,gu,dicRes=re.algo_optimisation(new_hyperparameters,listSlice,set_r,set_r,grid_slices,dicRes)  
    grid_slices=np.array([ge,gn,gi,gu])
    
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
    
    #transfo = args.simulation
    
    res_obj = [('listSlice',listSlice),('ErrorEvolution',ErrorEvolution), ('DiceEvolution',DiceEvolution), ('EvolutionGridError',EvolutionGridError), ('EvolutionGridNbpoint',EvolutionGridNbpoint), ('EvolutionGridInter',EvolutionGridInter), ('EvolutionGridUnion',EvolutionGridUnion), ('EvolutionParameters',EvolutionParameters),('EvolutionTransfo',EvolutionTransfo),('RejectedSlices',rejectedSlices)]
    
    joblib_name = root + '/../res/' + args.output + '.joblib' + '.gz' 
    joblib.dump(res_obj,open(joblib_name,'wb'), compress=True)
    
    res = joblib.load(open(joblib_name,'rb'))
    key=[p[0] for p in res]
    element=[p[1] for p in res]
    listSlice=element[key.index('listSlice')]
    

