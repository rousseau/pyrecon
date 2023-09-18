#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:29:22 2022

@author: mercier
""" 
import numpy as np
import time
from data_simulation import findCommonPointbtw2V,  ErrorOfRegistrationBtw2Slice,  createArrayOfChamferDistance, error_for_each_slices
from faster_code.registration_all_algo import normalization,global_optimisation
from faster_code.load import loadSlice,loadimages
from input_argparser import InputArgparser
import joblib
from os import getcwd, path, mkdir
from tools import createVolumesFromAlist
from rec_ebner import convert2EbnerParam
from outliers_detection_intersection import ErrorSlice
import shutil



if __name__ == '__main__':
        
    root=getcwd()
    
    input_parser = InputArgparser()
    
    
    input_parser.add_filenames(required=True) #load images
    input_parser.add_filenames_masks() #load masks
    input_parser.add_simulation() #load simulated transformation
    input_parser.add_output(required=True) #output name
    input_parser.add_nomvt() #load images with no movement
    input_parser.add_ablation(required=True)
    input_parser.add_hyperparameters()
    
    args = input_parser.parse_args()
    
    start = time.time()
    costGlobal = np.zeros(3)
    
    
    listSlice = [] 
    listnomvt = []
    
    #Create a list of slices from the images
    for i in range(len(args.filenames)):
        print('----load images-----')
        im, inmask = loadimages(args.filenames[i], args.filenames_masks[i]) 
        
        print(args.filenames[i])
        print(im.shape)
        loadSlice(im, inmask, listSlice, i,i)
        
        im, inmask = loadimages(args.nomvt[i], args.filenames_masks[i])
        loadSlice(im, inmask, listnomvt, i,i)

    #normalize the data with a standart distribution
    image,mask = createVolumesFromAlist(listSlice)
    listSliceNorm = []
    
    print(len(image))
    for m in image:
        listSliceNorm = listSliceNorm + normalization(m)
    
    listSlice = listSliceNorm
    
    #Algorithm of motion correction
    ablation = args.ablation
    print('hyperparameters :',args.hyperparameters)
    print('ablation :', ablation)
    print('----Start Registration----')
    dicRes, rejectedSlices = global_optimisation(args.hyperparameters,listSlice,ablation) 
    print('-----End Registration-----')
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
        
    transfo = args.simulation #the simulated transformation
    
    
    listErrorSlice = [ErrorSlice(slicei.get_orientation(),slicei.get_index_slice()) for slicei in listSlice]
    error_for_each_slices(listnomvt,listSlice,listErrorSlice,transfo,rejectedSlices)
    #update_feature(listSlice,listErrorSlice,EvolutionGridError[-1,:,:],EvolutionGridNbpoint[-1,:,:],EvolutionGridInter[-1,:,:],EvolutionGridUnion[-1,:,:])

    
    
    #save result in a joblib
    res_obj = [('listSlice',listSlice),('ErrorEvolution',ErrorEvolution), ('DiceEvolution',DiceEvolution), ('EvolutionGridError',EvolutionGridError), ('EvolutionGridNbpoint',EvolutionGridNbpoint), ('EvolutionGridInter',EvolutionGridInter), ('EvolutionGridUnion',EvolutionGridUnion), ('EvolutionParameters',EvolutionParameters),('EvolutionTransfo',EvolutionTransfo), ('RejectedSlices',rejectedSlices),('ListError',listErrorSlice)]
    joblib_name = root + '/' + args.output + '.joblib' + '.gz' 
    joblib.dump(res_obj,open(joblib_name,'wb'), compress=True)
    end = time.time()
    elapsed = end - start
    
    res = joblib.load(open(joblib_name,'rb'))
    key=[p[0] for p in res]
    element=[p[1] for p in res]
    listSlice=element[key.index('listSlice')]
    

    list_prefixImage = []
    for string_name in args.filenames:
        name_file = string_name.split('/')[-1]
        name = name_file.replace('.nii.gz','')
        list_prefixImage.append(name)
        
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
    
    