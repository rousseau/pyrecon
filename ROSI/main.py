#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:29:22 2022

@author: mercier
""" 
import numpy as np
import time
from rosi.simulation.validation import tre_indexes,  slice_tre,  distance_from_mask_edges, tre_for_each_slices
from rosi.registration.tools import normalization
from rosi.registration.motion_correction import global_optimisation
from rosi.registration.load import convert2Slices,loadStack
from input_argparser import InputArgparser
import joblib
from os import getcwd, path, mkdir
from rosi.registration.tools import separate_slices_in_stacks
from rosi.NiftyMIC.rec_ebner import convert2EbnerParam
from rosi.registration.outliers_detection.outliers import sliceFeature
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
        im, inmask = loadStack(args.filenames[i], args.filenames_masks[i]) 
        
        print(args.filenames_masks[i])#,print('i',i))
        print(im.shape)
        output = convert2Slices(im, inmask, [], i,i)
        listSlice+=output

        im, inmask = loadStack(args.nomvt[i], args.filenames_masks[i])
        nomt = convert2Slices(im, inmask, [], i,i)
        listnomvt+=nomt

    print('len :',len(listSlice))
    #normalize the data with a standart distribution
    image,mask = separate_slices_in_stacks(listSlice.copy())
    listSliceNorm = []
    
    print(len(image))
    for m in image:
        listSliceNorm = listSliceNorm + normalization(m)
    
    listSlice = listSliceNorm
    print('taille list :',len(listSlice))
    
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
    transfo=np.array(transfo,dtype=str)
    print(type(transfo))
    print('truc')
    print(transfo)
    listsliceFeatures = [sliceFeature(slicei.get_stackIndex(),slicei.get_indexSlice()) for slicei in listSlice]
    tre_for_each_slices(listnomvt,listSlice,listsliceFeatures,transfo,rejectedSlices)
    #update_feature(listSlice,listsliceFeatures,EvolutionGridError[-1,:,:],EvolutionGridNbpoint[-1,:,:],EvolutionGridInter[-1,:,:],EvolutionGridUnion[-1,:,:])

    
    
    #save result in a joblib
    res_obj = [('listSlice',listSlice),('ErrorEvolution',ErrorEvolution), ('DiceEvolution',DiceEvolution), ('EvolutionGridError',EvolutionGridError), ('EvolutionGridNbpoint',EvolutionGridNbpoint), ('EvolutionGridInter',EvolutionGridInter), ('EvolutionGridUnion',EvolutionGridUnion), ('EvolutionParameters',EvolutionParameters),('EvolutionTransfo',EvolutionTransfo), ('RejectedSlices',rejectedSlices),('ListError',listsliceFeatures)]
    
    #joblib_name =  args.output 
    import os

    dirname = os.path.dirname(__file__)
    joblib_name = os.path.join(dirname,'../'+ args.output + '.joblib.gz')
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
    
    