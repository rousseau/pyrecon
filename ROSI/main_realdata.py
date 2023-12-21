#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:24:49 2022

@author: mercier
"""
import numpy as np
import time
from rosi.registration.motion_correction import compute_cost_matrix,compute_cost_from_matrix,global_optimisation
from rosi.registration.tools import normalization
from rosi.registration.load import convert2Slices,loadStack
import nibabel as nib
from input_argparser import InputArgparser
import joblib
from os import getcwd, path, mkdir, rmdir
from rosi.NiftyMIC.rec_ebner import convert2EbnerParam
import shutil
import os

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
    
    #loaded_model = pickle.load(open('my_model.pickle', "rb"))
    
    
    start = time.time()
    costGlobal = np.zeros(2)
    
    list_prefixImage = []
    for string_name in args.filenames:
        name_file = string_name.split('/')[-1]
        name = name_file.replace('.nii.gz','')
        list_prefixImage.append(name)
    #print('list of images :',list_prefixImage)
    
    
    
    #loading image
    listSlice = []
    #Load images, mask and create a list of slices from the images
    image_pre=[]
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
                #print('nx_img1 :', nx_img1, 'ny_img1 :', ny_img1, 'nz_img1 :', nz_img1)
                #print('nx_img1_norm :', nx_img1_norm,'ny_img1_norm', ny_img1_norm,'nz_img1_norm', nz_img1_norm)
                #print(i,' : Axial')
              i_image=i_image+1
        
            else:
              nx=Affine[0:3,0].copy()
              ny=Affine[0:3,1].copy()
              nz=np.cross(nx,ny)
              nz_norm=nz/np.linalg.norm(nz)
              #orx=np.sqrt((nz_norm[0]-nx_img1_norm[0])**2+(nz_norm[1]-nx_img1_norm[1])**2+(nz_norm[2]-nx_img1_norm[2])**2)
              orz=np.abs(np.dot(nz_norm,nz_img1_norm))
              ory=np.abs(np.dot(nz_norm,ny_img1_norm))
              orx=np.abs(np.dot(nz_norm,nx_img1_norm))
              #ory=np.sqrt((nz_norm[0]-ny_img1_norm[0])**2+(nz_norm[1]-ny_img1_norm[1])**2+(nz_norm[2]-ny_img1_norm[2])**2)
              #=np.sqrt((nz_norm[0]-nz_img1_norm[0])**2+(nz_norm[1]-nz_img1_norm[1])**2+(nz_norm[2]-nz_img1_norm[2])**2)
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
                  (im,mask,[],0,i_image)
                  print('orx :', orx, 'ory :', ory, 'orz :', orz)
                  print(i , ' : Axial')
                  i_image=i_image+1
              
            print('i_image',i_image)
              #i_image=i_image+1
              
        else :
            i_prefix = i - nb_remove
            del list_prefixImage[i_prefix]
            print(list_prefixImage)
            nb_remove=nb_remove+1
    print(list_prefixImage)
    #normalize the data with a standart distribution
    print('---Data Normalization------')
    listSlice = normalization(listSlice)
    listSlicessMvt=listSlice.copy()
    
    print('---Compute Initial Cost----')
    gridError, gridNbpoint, gridInter, gridUnion = compute_cost_matrix(listSlice)
    cost = compute_cost_from_matrix(gridError,gridNbpoint)
    costGlobal[0] = cost
    
    #Simulated data and Motion Correction
    ablation = args.ablation
    #Apply motion correction
    print('----Start Motion Correction-----')
    dicRes,rejectedSlices = global_optimisation(args.hyperparameters,listSlice,ablation) #Algorithm of motion correction
    ge_mvtCorrected,gn_mvtCorrected,gi_mvtCorrected,gu_mvtCorrected = compute_cost_matrix(listSlice) #Compute 2 grid for the MSE
    cost = compute_cost_from_matrix(ge_mvtCorrected,gn_mvtCorrected)
    costGlobal[1] = cost
    print('----End of Motion Correction-----')
    
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
        
    res_obj = [('listSlice',listSlice),('ErrorEvolution',ErrorEvolution), ('DiceEvolution',DiceEvolution), ('EvolutionGridError',EvolutionGridError), ('EvolutionGridNbpoint',EvolutionGridNbpoint), ('EvolutionGridInter',EvolutionGridInter), ('EvolutionGridUnion',EvolutionGridUnion), ('EvolutionParameters',EvolutionParameters),('EvolutionTransfo',EvolutionTransfo),('RejectedSlices',rejectedSlices)]
    
    dirname = os.path.dirname(__file__)
    joblib_name = os.path.join(dirname,'../'+ args.output + '.joblib.gz')
    joblib.dump(res_obj,open(joblib_name,'wb'), compress=True)
    
    end = time.time()
    elapsed = end - start
    print(f'Temps d\'exécution : {elapsed}')

    res = joblib.load(open(joblib_name,'rb'))
    key=[p[0] for p in res]
    element=[p[1] for p in res]
    listSlice=element[key.index('listSlice')]
    

    print('----Saved Results-------')
    #save results for NiftyMIC
    parent_dir = getcwd() + '/'
    directory = args.output + '_mvt'
    path_dir = path.join(parent_dir, directory)
    if not path.isdir(path_dir):
        mkdir(path_dir) 
    else:
        shutil.rmtree(path_dir)
        mkdir(path_dir)
    convert2EbnerParam(res,list_prefixImage,path_dir)
print('---Registration Done----')
    