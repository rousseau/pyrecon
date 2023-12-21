#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:23:38 2023

@author: mercier
"""

from input_argparser import InputArgparser
from os import getcwd, path, mkdir
import numpy as np
import joblib
from rosi.registration.load import convert2Slices
import nibabel as nib
from rosi.registration.outliers_detection.feature import update_features
from rosi.registration.outliers_detection.multi_start import correct_slice_with_theorical_error, removeBadSlice
from rosi.simulation.validation import same_order, tre_for_each_slices
from rosi.registration.intersection import compute_cost_matrix, compute_cost_from_matrix
from rosi.registration.tools import computeMaxVolume



if __name__ == '__main__':
    
    root=getcwd()
    
    #input arguments :
    input_parser = InputArgparser()
    input_parser.add_filenames(required=True)
    input_parser.add_output(required=True)
    input_parser.add_simulation() #load simulated transformation
    input_parser.add_nomvt() #load images with no movement
    input_parser.add_filenames_masks()
    input_parser.add_ablation()
    input_parser.add_hyperparameters()
    args = input_parser.parse_args()
    
    
    transfo_str = np.array(args.simulation)
    print(type(transfo_str))
    nomvt = args.nomvt
    mask = args.filenames_masks
    opti = args.ablation
    hyperparameters = np.array(args.hyperparameters).astype(float)

    print('hyperparameters :',hyperparameters)
    print(type(hyperparameters[0]))
    listnomvt=[]
    output = convert2Slices(nib.load(nomvt[0]),nib.load(mask[0]),[],0,0)
    listnomvt+=output
    output = convert2Slices(nib.load(nomvt[1]),nib.load(mask[1]),[],1,1)
    listnomvt+=output
    output=convert2Slices(nib.load(nomvt[2]),nib.load(mask[2]),[],2,2)
    listnomvt+=output

    res = joblib.load(open(args.filenames[0],'rb'))
    key = [p[0] for p in res]
    element = [p[1] for p in res]

    listOfSlice = element[0]
    listFeatures = element[-1]
    

    image,nomvt,features,transfolist=same_order(listOfSlice,listnomvt,listFeatures,transfo_str)
    listOfSlice=np.concatenate(image)
    listNomvt=np.concatenate(nomvt)
    listFeatures=np.concatenate(features)
    number_slice = len(listOfSlice)
    print([len(image[i]) for i in range(0,3)],[len(nomvt[i]) for i in range(0,3)],[len(features[i]) for i in range(0,3)])
    
    squarre_error,nbpoint_matrix,intersection_matrix,union_matrix=compute_cost_matrix(listOfSlice)
    matrix = np.array([squarre_error,nbpoint_matrix,intersection_matrix,union_matrix])
    update_features(listOfSlice,listFeatures,squarre_error,nbpoint_matrix,intersection_matrix,union_matrix)
    
    #image,theorique,features,transfolist=same_order(listOfSlice,listNomvt,listFeatures,transfo_str)
    tre_for_each_slices(listNomvt,listOfSlice,listFeatures,transfolist,[])
    original_error = [e.get_error() for e in listFeatures]
    

    
    dicRes={}
    dicRes["evolutionparameters"] =np.reshape(element[key.index('EvolutionParameters')][-1,:,:],-1).tolist() 
    dicRes["evolutiontransfo"] = np.reshape(element[key.index('EvolutionTransfo')][-1,:,:],(4*number_slice,4)).tolist() 
    dicRes["evolutiongriderror"] = element[key.index('EvolutionGridError')][-1,:,:].tolist()  
    dicRes["evolutiongridnbpoint"] = element[key.index('EvolutionGridNbpoint')][-1,:,:].tolist() 
    dicRes["evolutiongridinter"] = element[key.index('EvolutionGridInter')][-1,:,:].tolist()  
    dicRes["evolutiongridunion"] = element[key.index('EvolutionGridUnion')][-1,:,:].tolist() 
    dicRes["evolutionerror"] = element[key.index('EvolutionGridError')][-1,:,:].tolist()
    dicRes["evolutionnbpoint"] = element[key.index('EvolutionGridNbpoint')][-1,:,:].tolist()
    dicRes["evolutionGridInter"] = element[key.index('EvolutionGridInter')][-1,:,:].tolist()
    dicRes["evolutionGridUnion"] = element[key.index('EvolutionGridUnion')][-1,:,:].tolist()
    costMse=compute_cost_from_matrix(squarre_error, nbpoint_matrix)
    print("Cost Before we start multi-start",costMse)
    costDice=compute_cost_from_matrix(intersection_matrix,union_matrix)
    dicRes["evolutionerror"] = [] 
    dicRes["evolutiondice"] = []
    dicRes["evolutionerror"].append(costMse)
    dicRes["evolutiondice"].append(costDice)
    
       
    #if ablation!='no_multistart':
    set_r = np.array(original_error)>3
    index = np.where(set_r==1)
    print('Slice not well registered before multistart')
    print(index)
    Vmx = computeMaxVolume(listOfSlice)
    print(len(listnomvt))
    tre_for_each_slices(listNomvt,listOfSlice,listFeatures,transfolist,[])
    print(opti[0])
    set_r,tre_new = correct_slice_with_theorical_error(set_r,listOfSlice,hyperparameters,opti[0],Vmx,matrix,listFeatures,transfolist,listNomvt)
    print('Slice not well registered after multistart')
    print(np.where(set_r==1))
    
    bad_slices = removeBadSlice(listOfSlice,set_r)
    
    
    ErrorEvolution=dicRes["evolutionerror"]
    DiceEvolution=dicRes["evolutiondice"]
    nbit = len(ErrorEvolution)
 
    
    #strEGE = file + '/EvolutionGridError.npz'
    EvolutionGridError = np.reshape(dicRes["evolutiongriderror"],[nbit,number_slice,number_slice])
    #np.savez_compressed(strEGE,EvolutionGridError)
    
    #strEGN = file + '/EvolutionGridNbpoint.npz'
    EvolutionGridNbpoint = np.reshape(dicRes["evolutiongridnbpoint"],[nbit,number_slice,number_slice])
    #np.savez_compressed(strEGN,EvolutionGridNbpoint)
    
    #strEGI = file + '/EvolutionGridInter.npz'
    EvolutionGridInter = np.reshape(dicRes["evolutiongridinter"],[nbit,number_slice,number_slice])
    #np.savez_compressed(strEGI,EvolutionGridInter)
    
    #strEGU = file + '/EvolutionGridUnion.npz'
    EvolutionGridUnion = np.reshape(dicRes["evolutiongridunion"],[nbit,number_slice,number_slice])
    #np.savez_compressed(strEGU,EvolutionGridUnion)
    
    #strEP = file + '/EvolutionParameters.npz'
    EvolutionParameters = np.reshape(dicRes["evolutionparameters"],[nbit,number_slice,6])
    #np.savez_compressed(strEP,EvolutionParameters)
    
    #strET = file + '/EvolutionTransfo.npz'
    EvolutionTransfo = np.reshape(dicRes["evolutiontransfo"],[nbit,number_slice,4,4])
    #np.savez_compressed(strET,EvolutionTransfo)
    #strCG = file + '/CostGlobal.npz'
    #costGlobal.tofile(strCG)
    
    #transfo = args.simulation
    tre_for_each_slices(listNomvt,listOfSlice,listFeatures,transfolist,[])
    #tre_new = np.array([e.get_error() for e in listFeatures])

    res_obj = [('listSlice',listOfSlice),('ErrorEvolution',ErrorEvolution), ('DiceEvolution',DiceEvolution), ('EvolutionGridError',EvolutionGridError), ('EvolutionGridNbpoint',EvolutionGridNbpoint), ('EvolutionGridInter',EvolutionGridInter), ('EvolutionGridUnion',EvolutionGridUnion), ('EvolutionParameters',EvolutionParameters),('EvolutionTransfo',EvolutionTransfo),('RejectedSlices',bad_slices),('ListError',listFeatures)]
    
    joblib_name = args.output + '_v2.joblib' + '.gz' 
    joblib.dump(res_obj,open(joblib_name,'wb'), compress=True)
    
    res = joblib.load(open(joblib_name,'rb'))
    key=[p[0] for p in res]
    element=[p[1] for p in res]
    listSlice=element[key.index('listSlice')]

    
    

