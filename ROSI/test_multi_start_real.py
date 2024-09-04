#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:23:38 2023

@author: mercier
"""

import shutil
from input_argparser import InputArgparser
from os import getcwd, path, mkdir
import numpy as np
import joblib
from rosi.registration.outliers_detection.outliers import sliceFeature
from rosi.reconstruction.rec_ebner import convert2EbnerParam
from rosi.registration.load import convert2Slices
import nibabel as nib
from rosi.registration.outliers_detection.feature import update_features, detect_misregistered_slice
from rosi.registration.outliers_detection.multi_start import correct_slice, removeBadSlice
from rosi.simulation.validation import same_order, tre_for_each_slices
from rosi.registration.intersection import compute_cost_matrix, compute_cost_from_matrix
from rosi.registration.tools import computeMaxVolume
from rosi.registration.optimisation import algo_optimisation
 


if __name__ == '__main__':
    
    root=getcwd()
    
    #input arguments :
    input_parser = InputArgparser()
    input_parser.add_filenames(required=True)
    input_parser.add_output(required=True)
    input_parser.add_filenames_masks()
    input_parser.add_classifier()


    args = input_parser.parse_args()

    mask = args.filenames_masks
    classifier = args.classifier


    list_prefixImage = []
    for string_name in mask:
        name_file = string_name.split('/')[-1]
        name = name_file.replace('.nii.gz','')
        list_prefixImage.append(name)

    print(list_prefixImage)

    #print('hyperparameters :',hyperparameters)
    #print(type(hyperparameters[0]))


    res = joblib.load(open(args.filenames[0],'rb'))
    key = [p[0] for p in res]
    element = [p[1] for p in res]

    listOfSlice = element[0]
    number_slice = len(listOfSlice)
    
    listFeatures = [sliceFeature(s.get_stackIndex(),s.get_indexSlice()) for s in listOfSlice]
    squarre_error,nbpoint_matrix,intersection_matrix,union_matrix=compute_cost_matrix(listOfSlice)
    matrix = np.array([squarre_error,nbpoint_matrix,intersection_matrix,union_matrix])
    update_features(listOfSlice,listFeatures,squarre_error,nbpoint_matrix,intersection_matrix,union_matrix)
    
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

    import pickle
    load_model = pickle.load(open(classifier,'rb'))

    #if ablation!='no_multistart':
    set_r = detect_misregistered_slice(listOfSlice,matrix,load_model,0.8)
    #before08=np.sum(set_r)
    set_o = detect_misregistered_slice(listOfSlice,matrix,load_model,0.5)
    #before05=np.sum(set_o)
    #index = np.where(set_r==1)
    #print('Slice not well registered before multistart')
    #print(index)
    #Vmx = computeMaxVolume(listOfSlice)
    hyperparameters = np.array([4, 0.0001, 2000, 0.25, 1, 0])
    Vmx = computeMaxVolume(listOfSlice)
    set_r = correct_slice(set_r,set_o,listOfSlice,hyperparameters,'Nelder-Mead',Vmx,matrix,listFeatures,load_model,dicRes,0.8)
    squarre_error,nbpoint_matrix,intersection_matrix,union_matrix=compute_cost_matrix(listOfSlice)
    matrix = np.array([squarre_error,nbpoint_matrix,intersection_matrix,union_matrix])
    #set_r = detect_misregistered_slice(listOfSlice,matrix,load_model,0.8)
    #after08=np.sum(set_r)
    set_o = detect_misregistered_slice(listOfSlice,matrix,load_model,0.5)
    after05=np.sum(set_o)
    #ge,gn,gi,gu,dicRes=algo_optimisation(hyperparameters,listOfSlice,set_o,set_r,matrix,dicRes,Vmx,1,'Nelder-Mead')
    #print('Slice not well registered after multistart')
    #print(np.where(set_r==1))
    
    # set_r = np.zeros(len(listOfSlice))
    bad_slices = removeBadSlice(listOfSlice,set_o)
    
    
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
    #multi_start = [('Before at 0.5',before05),('Before at 0.8',before08),('After at 0.5',after05),('After at 08',after08)]
    #transfo = args.simulation
    #tre_new = np.array([e.get_error() for e in listFeatures])
    res_obj = [('listSlice',listOfSlice),('ErrorEvolution',ErrorEvolution), ('DiceEvolution',DiceEvolution), ('EvolutionGridError',EvolutionGridError), ('EvolutionGridNbpoint',EvolutionGridNbpoint), ('EvolutionGridInter',EvolutionGridInter), ('EvolutionGridUnion',EvolutionGridUnion), ('EvolutionParameters',EvolutionParameters),('EvolutionTransfo',EvolutionTransfo),('RejectedSlices',bad_slices),('ListError',listFeatures)]
    #,('multi_start',multi_start)]
    
    joblib_name = args.output + '.joblib' + '.gz' 
    joblib.dump(res_obj,open(joblib_name,'wb'), compress=True)
    
    res = joblib.load(open(joblib_name,'rb'))
    key=[p[0] for p in res]
    element=[p[1] for p in res]
    listSlice=element[key.index('listSlice')]
    
    print('-----Save Results-----')
    parent_dir = getcwd() + '/'
    directory = args.output + '_mvt'
    path_dir = path.join(parent_dir, directory)
    if not path.isdir(path_dir):
        mkdir(path_dir) 
    else:
        shutil.rmtree(path_dir)
        mkdir(path_dir)
    convert2EbnerParam(res,list_prefixImage,path_dir)