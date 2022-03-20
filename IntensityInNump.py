#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from registration import loadimages, loadSlice, normalization
import nibabel as nib
import numpy as np
from data_simulation import create3VolumeFromAlist

"""
Created on Mon Mar  7 10:51:47 2022

@author: mercier
"""

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('-i','--input',help='Input Image',type=str,action='append')
    parser.add_argument('--imask',help='Mask of the images',type=str,action='append')
    parser.add_argument('--idata',help='',type=str,action='append')
    args = parser.parse_args()
    
    listSlice = []
    
    Orientation = ['Axial','Coronal','Sagittal']
    
    for i in range(len(args.input)):
        im, inmask = loadimages(args.input[i],args.imask[i])
        datamask = inmask.get_fdata().squeeze()
        mask = nib.Nifti1Image(datamask, inmask.affine)
        loadSlice(im, mask, listSlice, Orientation[i])
    
    sizeList = len(listSlice)
    #normalize the data with a standart distribution
    
    data = args.idata[0]
    ErrorEvolution = np.fromfile(data + 'ErrorEvolution.dat') #0
    nit = len(ErrorEvolution)
        
    EPA = np.fromfile(data + 'EvolutionParameters.dat')
    PreviousParameters = np.reshape(EPA,[nit,sizeList,6])
    
    
     
    for i_slice in range(len(listSlice)):
        
        s = listSlice[i_slice]
        s.set_parameters(PreviousParameters[nit-1,i_slice,:])

    Axial,Coronal,Sagital = create3VolumeFromAlist(listSlice)
    

    X,Y,Z=Axial[0].get_slice().get_fdata().shape
    Z=len(Axial)
    npAxial=[]
    X,Y,Z=Coronal[0].get_slice().get_fdata().shape
    Z=len(Coronal)
    npCoronal=[]
    X,Y,Z=Sagital[0].get_slice().get_fdata().shape
    Z=len(Sagital)
    npSagital=[]

    
    Vol=[Axial,Coronal,Sagital]
    strVol=['Axial','Coronal','Sagital']
    VolSave=[npAxial,npCoronal,npSagital]
    for i_vol in range(len(Vol)):
        print(strVol[i_vol])
        for i_slice in range(len(Vol[i_vol])):
            nump=VolSave[i_vol]
            slicei=Vol[i_vol][i_slice]
            X,Y,Z=slicei.get_slice().get_fdata().shape
            M=slicei.get_slice().affine
            for x in range(X):
                for y in range(Y):
                    if slicei.get_mask()[x,y,0]>0:
                            valArray=np.zeros((1,4))
                            valReal=M @ np.array([x,y,0,1])
                            valArray[0,0:3]=valReal[0:3]
                            valArray[0,3]=slicei.get_slice().get_fdata()[x,y,0]
                            nump.extend(valArray)
        array=np.asarray(nump)
        print(array.shape)
        datasave='%sresult_%s'  %(data,strVol[i_vol])
        np.savez_compressed(datasave,nump)