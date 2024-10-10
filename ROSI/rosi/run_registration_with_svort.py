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
from rosi.registration.load import convert2Slices,loadStack,loadFromdir
from input_argparser import InputArgparser
import joblib
from os import getcwd, path, mkdir
from rosi.registration.tools import separate_slices_in_stacks
from rosi.reconstruction.rec_ebner import convert2EbnerParam
from rosi.registration.outliers_detection.outliers import sliceFeature
import shutil
import os
import nibabel as nib



def main():
    
    print('test')
    input_parser = InputArgparser() 
    input_parser.add_filenames(required=True) #load images
    input_parser.add_output(required=True) #output 
    input_parser.add_tre() #compute TRE for each slice
    input_parser.add_isimplex() #initial size of the simplex, default value is 4
    input_parser.add_fsimplex() #final size of the simplex, default value is 0.25
    input_parser.add_localConvergence() #local convergence Threshold, default value is 2
    input_parser.add_omega() # intersection weight, default value is 0
    input_parser.add_no_mutlistart() #by default, algorithm use multistart
    input_parser.add_optimisation() #Nedler-Mead
    input_parser.add_classifier() #random forest classifier

    args = input_parser.parse_args()
    args.tre

    if args.tre==1 : 
        input_parser.add_nomvt(required=True) #load images with no movement
        input_parser.add_nomvt_mask(required=True)
        input_parser.add_transformation(required=True)

     
    args = input_parser.parse_args()
    print(args.classifier)
    costGlobal = np.zeros(3)
    
    
    listSlice = [] #list of the slices

    if args.tre==1:
        listnomvt = [] #list of the slices without simulated motion : use to compute the tre
    

    dir_input = args.filenames[0]
    print(dir_input)
    listSlice = loadFromdir(dir_input)

    print('\n')
    #normalize the data with z-score
    image,mask = separate_slices_in_stacks(listSlice.copy()) #create n list of slices and n mask, corresponding to n stack 
    
    listSliceNorm = [] 
    
    for m in image:
        listSliceNorm = listSliceNorm + normalization(m)
    
    listSlice = listSliceNorm
    print(len(listSlice))
    
    #Algorithm of motion correction
    print("--Method use for optimisation is :", args.optimisation)
    print(args.optimisation)
    if args.optimisation=="Nelder-Mead" : 
        print("--Parameters value for optimisation are : ")  
        print("ds = ", args.initial_simplex)
        print("fs = ", args.final_simplex)
    else : 
        print("Default value from scipy are used")
    print('\n')
    print("--Other Parameters :")
    print("th = ",args.local_convergence)
    print("Omega = ",args.omega)
    print("Multi-start :",args.no_multistart)
    print("Path to classifier is :",args.classifier)
    print('\n')



    print('----Start Registration :')
    #dicRes, rejectedSlices = global_optimisation(listSlice,optimisation=args.optimisation,classifier=args.classifier,multi_start=args.no_multistart,hyperparameters={'ds':args.initial_simplex,'fs':args.final_simplex,'T':args.local_convergence,'omega':args.omega}) 
    print('\n')
    print('-----End Registration :')
    
    
    #reshape results for saving : 
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
    
    if args.tre==1:
        transfo = args.simulation #the simulated transformation
        transfo=np.array(transfo,dtype=str)
        listsliceFeatures = [sliceFeature(slicei.get_stackIndex(),slicei.get_indexSlice()) for slicei in listSlice]
        tre_for_each_slices(listnomvt,listSlice,listsliceFeatures,transfo,rejectedSlices)
        #save result in a joblib
        res_obj = [('listSlice',listSlice),('ErrorEvolution',ErrorEvolution), ('DiceEvolution',DiceEvolution), ('EvolutionGridError',EvolutionGridError), ('EvolutionGridNbpoint',EvolutionGridNbpoint), ('EvolutionGridInter',EvolutionGridInter), ('EvolutionGridUnion',EvolutionGridUnion), ('EvolutionParameters',EvolutionParameters),('EvolutionTransfo',EvolutionTransfo), ('RejectedSlices',rejectedSlices),('ListError',listsliceFeatures)]
    else :
        res_obj = [('listSlice',listSlice),('ErrorEvolution',ErrorEvolution), ('DiceEvolution',DiceEvolution), ('EvolutionGridError',EvolutionGridError), ('EvolutionGridNbpoint',EvolutionGridNbpoint), ('EvolutionGridInter',EvolutionGridInter), ('EvolutionGridUnion',EvolutionGridUnion), ('EvolutionParameters',EvolutionParameters),('RejectedSlices',rejectedSlices)]
   
    
    #save results
    print('-----Save Results : ')
    dirname = os.path.dirname(__file__)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    joblib_name = os.path.join(args.output + '/res.joblib.gz')
    #joblib_name = os.path.join(dirname,'../../'+ args.output + '.joblib.gz')
    joblib.dump(res_obj,open(joblib_name,'wb'), compress=True)
    
    
    #Create directory of transformations for NiftyMIC
    #print('-----Save Results for NiftyMIC')
    #res = joblib.load(open(joblib_name,'rb'))
    #key=[p[0] for p in res]
    #element=[p[1] for p in res]
    #listSlice=element[key.index('listSlice')]
    
    #list_prefixImage = []
    #for string_name in args.filenames:
    #    name_file = string_name.split('/')[-1]
    #    name = name_file.replace('.nii.gz','')
    #    list_prefixImage.append(name)
        
    #parent_dir = getcwd() + '/'
    #path_dir = os.path.join(args.output + '/niftimic_mvt')
    #path_dir = path.join(parent_dir, directory)
    #if not path.isdir(path_dir):
    #    mkdir(path_dir) 
    #else:
    #    shutil.rmtree(path_dir)
    #    mkdir(path_dir)
    #convert2EbnerParam(res,list_prefixImage,path_dir)
    
if __name__ == '__main__':
    main()   