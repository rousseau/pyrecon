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
from os import getcwd, path, mkdir, makedirs
from rosi.registration.tools import separate_slices_in_stacks
from rosi.reconstruction.link_to_reconstruction import convert2EbnerParam
from rosi.registration.outliers_detection.outliers import sliceFeature
import shutil
import os
import nibabel as nib



def main():
    
    print('test')
    input_parser = InputArgparser() 
    input_parser.add_filenames(required=True) #load images
    input_parser.add_filenames_masks(required=True) #load masks
    input_parser.add_output(required=True) #output 
    input_parser.add_tre() #compute TRE for each slice
    input_parser.add_isimplex() #initial size of the simplex, default value is 4
    input_parser.add_fsimplex() #final size of the simplex, default value is 0.25
    input_parser.add_localConvergence() #local convergence Threshold, default value is 2
    input_parser.add_omega() # intersection weight, default value is 0
    input_parser.add_no_mutlistart() #by default, algorithm use multistart
    input_parser.add_optimisation() #Nedler-Mead
    input_parser.add_classifier() #random forest classifier
    input_parser.add_nomvt(required=False) #load images with no movement
    input_parser.add_nomvt_mask(required=False)
    input_parser.add_transformation(required=False)
    add_transformation

    args = input_parser.parse_args()
    args.tre

    if args.tre and (args.nomvt is None or args.nomvt_mask is None or args.transformation is None):
        input_parser.error("--tre requires --nomvt and --nomvt_mask and --transformation.")

     
    print(args.classifier)
    costGlobal = np.zeros(3)
    
    
    listSlice = [] #list of the slices

    if args.tre==1:
        listnomvt = [] #list of the slices without simulated motion : use to compute the tre
    

    i_image=0
    nb_remove=0
    i_prefix=0
    for i in range(len(args.filenames)):
        print(args.filenames[i])
        print('------------load images--------------------')
        im, inmask = loadStack(args.filenames[i],args.filenames_masks[i]) 
        Affine = im.affine

        datamask = inmask.get_fdata().squeeze()
        ##check mask and image size : 
        if datamask.shape==im.get_fdata().shape:
            mask = nib.Nifti1Image(datamask, inmask.affine)
        
            if  i==0:
              nx_img1=Affine[0:3,0].copy()
              ny_img1=Affine[0:3,1].copy()
              nz_img1=np.cross(nx_img1,ny_img1)
              nx_img1_norm=nx_img1/np.linalg.norm(nx_img1)
              ny_img1_norm=ny_img1/np.linalg.norm(ny_img1)
              nz_img1_norm=nz_img1/np.linalg.norm(nz_img1)
              output = convert2Slices(im,mask,[],i_image,i_image)
              listSlice+=output
              i_image=i_image+1
        
            else:
              nx=Affine[0:3,0].copy()
              ny=Affine[0:3,1].copy()
              nz=np.cross(nx,ny)
              nz_norm=nz/np.linalg.norm(nz)
              
              orz=np.abs(np.dot(nz_norm,nz_img1_norm))
              ory=np.abs(np.dot(nz_norm,ny_img1_norm))
              orx=np.abs(np.dot(nz_norm,nx_img1_norm))
              
              if max(orx,ory,orz)==orx:
                  output = convert2Slices(im,mask,[],1,i_image)
                  listSlice+=output
                  print('orx :', orx, 'ory :', ory, 'orz :', orz)
                  print(i, ' : Coronal')
                  i_image=i_image+1
        
              elif max(orx,ory,orz)==ory:
                  output = convert2Slices(im,mask,[],2,i_image)
                  listSlice+=output
                  print('orx :', orx, 'ory :', ory, 'orz :', orz)
                  print(i ,' : Sagittal')
                  i_image=i_image+1
        
              else:
                  output = convert2Slices(im,mask,[],0,i_image)
                  listSlice+=output
                  print('orx :', orx, 'ory :', ory, 'orz :', orz)
                  print(i , ' : Axial')
                  i_image=i_image+1
              
            print('i_image',i_image)
             
              
        else :
            i_prefix = i - nb_remove
            del list_prefixImage[i_prefix]
            print(list_prefixImage)
            nb_remove=nb_remove+1

        if args.tre==1:
            im, inmask = loadStack(args.nomvt[i], args.filenames_masks[i])
            nomt = convert2Slices(im, inmask, [], i,i_image)
            listnomvt+=nomt

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
    dicRes, rejectedSlices = global_optimisation(listSlice,optimisation=args.optimisation,classifier=args.classifier,multi_start=args.no_multistart,hyperparameters={'ds':args.initial_simplex,'fs':args.final_simplex,'T':args.local_convergence,'omega':args.omega}) 
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
        transfo = args.transformation #the simulated transformation
        transfo=np.array(transfo,dtype=str)
        listsliceFeatures = [sliceFeature(slicei.get_stackIndex(),slicei.get_indexSlice()) for slicei in listSlice]
        tre_for_each_slices(listnomvt,listSlice,listsliceFeatures,transfo,[])
        tre = [errori.get_error() for errori in listsliceFeatures]
        #save result in a joblib
        res_obj = [('listSlice',listSlice),('ErrorEvolution',ErrorEvolution), ('DiceEvolution',DiceEvolution), ('EvolutionGridError',EvolutionGridError), ('EvolutionGridNbpoint',EvolutionGridNbpoint), ('EvolutionGridInter',EvolutionGridInter), ('EvolutionGridUnion',EvolutionGridUnion), ('EvolutionParameters',EvolutionParameters),('EvolutionTransfo',EvolutionTransfo), ('RejectedSlices',rejectedSlices),('ListError',listsliceFeatures),('tre',tre)]
    else :
        res_obj = [('listSlice',listSlice),('ErrorEvolution',ErrorEvolution), ('DiceEvolution',DiceEvolution), ('EvolutionGridError',EvolutionGridError), ('EvolutionGridNbpoint',EvolutionGridNbpoint), ('EvolutionGridInter',EvolutionGridInter), ('EvolutionGridUnion',EvolutionGridUnion), ('EvolutionParameters',EvolutionParameters),('RejectedSlices',rejectedSlices)]
   
    
    #save results
    print('-----Save Results : ')
    dirname = os.path.dirname(__file__)
    if not os.path.exists(args.output):
        makedirs(args.output)
    
    joblib_name = os.path.join(args.output + '/res.joblib.gz')
    #joblib_name = os.path.join(dirname,'../../'+ args.output + '.joblib.gz')
    joblib.dump(res_obj,open(joblib_name,'wb'), compress=True)
    
    
    #Create directory of transformations for NiftyMIC
    print('-----Save Results for NiftyMIC')
    res = joblib.load(open(joblib_name,'rb'))
    key=[p[0] for p in res]
    element=[p[1] for p in res]
    listSlice=element[key.index('listSlice')]
    
    list_prefixImage = []
    for string_name in args.filenames:
        name_file = string_name.split('/')[-1]
        name = name_file.replace('.nii.gz','')
        list_prefixImage.append(name)
        
    #parent_dir = getcwd() + '/'
    path_dir = os.path.join(args.output + '/niftimic_mvt')
    #path_dir = path.join(parent_dir, directory)
    if not path.isdir(path_dir):
        makedirs(path_dir) 
    else:
        shutil.rmtree(path_dir)
        makedirs(path_dir)
    convert2EbnerParam(res,list_prefixImage,path_dir)
    
if __name__ == '__main__':
    main()   
