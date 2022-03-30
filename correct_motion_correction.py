#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 14:08:00 2022

@author: mercier
"""

import os
import nibabel as nib
import argparse
from registration import loadimages,loadSlice, normalization, computeCostBetweenAll2Dimages, matrixOfWeight, cost_fct, updateCostBetweenAllImageAndOne, updateMatrixOfWeight, costFromMatrix
import numpy as np
from data_simulation import create3VolumeFromAlist, ErrorOfRegistrationBtw2Slice, findCommonPointbtw2V, ChamferDistance, createArrayOfChamferDistance
from scipy.optimize import minimize

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
    listSlice = normalization(listSlice)
    
    sliceAx,sliceCor,sliceSag = create3VolumeFromAlist(listSlice)
    
    #Compute a list of point that will be used to validate the quality of the registration
    ptAxSag_Ax,ptAxSag_Sag = findCommonPointbtw2V(sliceAx,sliceSag) #list of point between Axial and Sagittal
    ptAxCor_Ax, ptAxCor_Cor = findCommonPointbtw2V(sliceAx,sliceCor) #list of point between Axial and Coronnal
    ptCorSag_Cor, ptCorSag_Sag =  findCommonPointbtw2V(sliceCor,sliceSag) #list of point Between Coronal and Sagittal
    
    data = args.idata[0]
    print('data',data)
    ErrorEvolution = np.fromfile(data + 'ErrorEvolution.dat') #0
    #self.EvolutionLocalCost =  #0
        
    nit = len(ErrorEvolution)
        
    EGE = np.fromfile(data + 'EvolutionGridError.dat')
    EvolutionGridError = np.reshape(EGE,[nit,sizeList,sizeList])#np.zeros((1,sizeList,sizeList))
        
    EGN = np.fromfile(data + 'EvolutionGridNbpoint.dat')
    EvolutionGridNbpoint = np.reshape(EGN,[nit,sizeList,sizeList]) #np.zeros((1,sizeList,sizeList))
        
        #self.costAlreadyComputed = False
        
    EPA = np.fromfile(data + 'EvolutionParameters.dat')
    PreviousParameters = np.reshape(EPA,[nit,sizeList,6])
        
    for i_slice in range(len(listSlice)):
        
        s = listSlice[i_slice]
        s.set_parameters(PreviousParameters[0,i_slice,:])
    
    
    SliceAxBeforeReg,SliceCorBeforeReg,SliceSagBeforeReg = create3VolumeFromAlist(listSlice) #Create 3 list of slies that represents the volumes, axial, coronal and sagittal from the listWithMvt (list with simulated motion)
    
    #Compute the error of registration before motion correction. It will be used for the grafical representation of the reconstruction error.
    #This errror represents the distance, in mm, between two point that are the same in the original data (ie the data without simulation)
    #It is expected to be higher distance computed after registration
    errorAxCor_before = ErrorOfRegistrationBtw2Slice(ptAxCor_Ax,ptAxCor_Cor,SliceAxBeforeReg,SliceCorBeforeReg)  #Error of registration between axial and coronal
    errorAxSag_before = ErrorOfRegistrationBtw2Slice(ptAxSag_Ax,ptAxSag_Sag,SliceAxBeforeReg,SliceSagBeforeReg)  #Error of registration between axial and sagittal
    errorCorSag_before = ErrorOfRegistrationBtw2Slice(ptCorSag_Cor,ptCorSag_Sag,SliceCorBeforeReg,SliceSagBeforeReg)
    
    for i_slice in range(len(listSlice)):
        
        s = listSlice[i_slice]
        s.set_parameters(PreviousParameters[nit-1,i_slice,:])
    
    gridError,gridNbpoint,gridInter,gridUnion = computeCostBetweenAll2Dimages(listSlice)

    
    vectMse = np.zeros([gridError.shape[0]])
    
    for i in range(gridError.shape[0]):
        mse = sum(gridError[i,:]) + sum(gridError[:,i])
        point =  sum(gridNbpoint[i,:]) + sum(gridNbpoint[:,i])
        vectMse[i] = mse/point
                            
    valMse = np.median(vectMse[~np.isnan(vectMse)])
    
    threshold = 1.25*valMse
    print('threshold :', threshold)        
    gridWeight = matrixOfWeight(gridError, gridNbpoint, threshold)

    randomIndex= np.arange(sizeList)
    np.random.shuffle(randomIndex)                        

    delta = 0.05
    initial_s = np.zeros((7,6))
    
    delta = 5
    vectd = np.linspace(delta,1,delta,dtype=int)
    
    nbSlice = len(listSlice)
    
    EvolutionParameters = []
    ErrorEvolution = []
    EvolutionGridInter=[]
    EvolutionGridUnion=[]
    EvolutionGridError = []
    EvolutionGridNbpoint = []
    costGlobal = []
    
    costMse = costFromMatrix(gridError,gridNbpoint)
    cost = costMse
    costGlobal.append(cost)
    
    
    badRegistration = 0
    for i_slice in range(sizeList):
        if gridWeight[0,i_slice] == 0:
            slicei = listSlice[i_slice] 
            slicei.set_parameters(PreviousParameters[0,i_slice,:])
            badRegistration = badRegistration + 1
    print(badRegistration)
    
    gridError,gridNbpoint,gridInter,gridUnion = computeCostBetweenAll2Dimages(listSlice)
    
    for d in vectd:
           
            delta = d
        
        
            for i in range(1):
                    
                
                randomIndex= np.arange(nbSlice)
                np.random.shuffle(randomIndex)
                previous_cost = costFromMatrix(gridError,gridNbpoint)

                    
                for i_slice in randomIndex:
                        #for i in range(3):
                        if gridWeight[0,i_slice] == 0:
                            #if (gridWeight[0,i_slice] == True and test == 0) or (gridWeight[0,i_slice] == False and test == 1) :
                                
                            slicei = listSlice[i_slice] 
                            print('index slice: ',i_slice)
                            print('Mse of the slice', vectMse[i_slice])
                            
                            x = slicei.get_parameters()
                            P0 = x
                            P1 = x + np.array([delta,0,0,0,0,0])
                            P2 = x + np.array([0,delta,0,0,0,0])
                            P3 = x + np.array([0,0,delta,0,0,0])
                            P4 = x + np.array([0,0,0,delta,0,0])
                            P5 = x + np.array([0,0,0,0,delta,0])
                            P6 = x + np.array([0,0,0,0,0,delta])
                            
                            initial_s[0,:]=P0
                            initial_s[1,:]=P1
                            initial_s[2,:]=P2
                            initial_s[3,:]=P3
                            initial_s[4,:]=P4
                            initial_s[5,:]=P5
                            initial_s[6,:]=P6
                                        
                            X,Y = gridError.shape
                            NM = minimize(cost_fct,x,args=(i_slice,listSlice,gridError.copy(),gridNbpoint.copy(),gridInter.copy(),gridUnion.copy(),gridWeight.copy(),threshold),method='Nelder-Mead',options={"disp" : True,"maxiter" : 20, "maxfev":1e6, "xatol" : 1e-2, "fatol" : 1e-4, "initial_simplex" : initial_s , "adaptive" :  False})
                                
                            #print(listSlice[0].get_orientation())
                            x_opt = NM.x
                            slicei.set_parameters(x_opt)
                            updateCostBetweenAllImageAndOne(i_slice, listSlice, gridError, gridNbpoint,gridInter,gridUnion)
   
                            costMse = costFromMatrix(gridError,gridNbpoint)
                            current_cost = costMse
                            print('curent_cost:',current_cost)
            
                gridError,gridNbpoint,gridInter,gridUnion = computeCostBetweenAll2Dimages(listSlice)
                costMse = costFromMatrix(gridError,gridNbpoint)
                cost = costMse
                for i_slice in range(nbSlice):
                    s = listSlice[i_slice]
                    EvolutionParameters.extend(s.get_parameters())
            
                ErrorEvolution.append(cost)
                EvolutionGridError.append(gridError)
                EvolutionGridNbpoint.append(gridNbpoint)
                EvolutionGridInter.append(gridInter)
                EvolutionGridUnion.append(gridUnion)
    
    # for i_slice in randomIndex:
    #        if gridWeight[0,i_slice] == 0:
                  
    #               print('index slice: ',i_slice)
    #               print('Mse of the slice', vectMse[i_slice])
                                 
    #               slicei = listSlice[i_slice] 
    #               x = slicei.get_parameters()
    #               P0 = x
    #               P1 = x + np.array([delta,0,0,0,0,0])
    #               P2 = x + np.array([0,delta,0,0,0,0])
    #               P3 = x + np.array([0,0,delta,0,0,0])
    #               P4 = x + np.array([0,0,0,delta,0,0])
    #               P5 = x + np.array([0,0,0,0,delta,0])
    #               P6 = x + np.array([0,0,0,0,0,delta])
                  
    #               initial_s[0,:]=P0
    #               initial_s[1,:]=P1
    #               initial_s[2,:]=P2
    #               initial_s[3,:]=P3
    #               initial_s[4,:]=P4
    #               initial_s[5,:]=P5
    #               initial_s[6,:]=P6
                            
    #               X,Y = gridError.shape
    #               NM = minimize(cost_fct,x,args=(slicei,i_slice,listSlice,gridError.copy(),gridNbpoint.copy(),np.ones((X,Y)),threshold),method='Nelder-Mead',options={"disp" : True, "maxiter": 20,"maxfev":1e6,"xatol" : 1e-2, "fatol" : 1e-4, "initial_simplex" : initial_s , "adaptive" :  True})
    #               #print(listSlice[0].get_orientation())
    #               x_opt = NM.x
    #               slicei.set_parameters(x_opt)
    #               updateCostBetweenAllImageAndOne(slicei, i_slice, listSlice, gridError, gridNbpoint,'MSE')
    #               #gridWeight = updateMatrixOfWeight(gridError, gridNbpoint, gridWeight, i_slice, threshold)  
    #               # vectMse = np.zeros([gridError.shape[0]])
    #               # for i in range(gridError.shape[0]):
    #               #      mse = sum(gridError[i,:]) + sum(gridError[:,i])
    #               #      point =  sum(gridNbpoint[i,:]) + sum(gridNbpoint[:,i])
    #               #      vectMse[i] = mse/point
                            
    #               # valMse = np.median(vectMse[~np.isnan(vectMse)])
    
    #               # threshold = 1.25*valMse
    #               #print('threshold :', threshold)        
    #               #gridWeight = matrixOfWeight(gridError, gridNbpoint, threshold)
           
    #               costMse = costFromMatrix(gridError,gridNbpoint)
    #               current_cost = costMse
    
    gridError,gridNbpoint,gridInter,gridUnion = computeCostBetweenAll2Dimages(listSlice)
    costMse = costFromMatrix(gridError,gridNbpoint)
    cost = costMse
    costGlobal.append(cost)
    
    for i_slice in range(nbSlice):
        s = listSlice[i_slice]
        EvolutionParameters.extend(s.get_parameters())
            
    ErrorEvolution.append(cost)
    EvolutionGridInter.append(gridInter)
    EvolutionGridUnion.append(gridUnion)
    EvolutionGridError.append(gridError)
    EvolutionGridNbpoint.append(gridNbpoint)

          
    SliceAxAfterReg,SliceCorAfterReg,SliceSagAfterReg = create3VolumeFromAlist(listSlice) #Create 3 list of slices that represents the volume Axial, Coronal and Sagittal with simulation motion corrected by the algorithm
    
    #Error of registration after motion correction. It is expected to be smaller than the one before correction.
    errorAxCor_after = ErrorOfRegistrationBtw2Slice(ptAxCor_Ax,ptAxCor_Cor,SliceAxAfterReg,SliceCorAfterReg) #Error of registration between Axial and Coronal
    errorAxSag_after = ErrorOfRegistrationBtw2Slice(ptAxSag_Ax,ptAxSag_Sag,SliceAxAfterReg,SliceSagAfterReg) #Error of registration between Axial and Sagittal
    errorCorSag_after = ErrorOfRegistrationBtw2Slice(ptCorSag_Cor,ptCorSag_Sag,SliceCorAfterReg,SliceSagAfterReg) #Error of Registration between Coronal and Sagittal
    
    axial = args.input[0] 
    axmask = args.imask[0]
    imax, inmask = loadimages(axial,axmask)
    imgChamferDistanceAx = ChamferDistance(inmask)
    caxcor = createArrayOfChamferDistance(imgChamferDistanceAx,ptAxCor_Ax)
    caxsag = createArrayOfChamferDistance(imgChamferDistanceAx,ptAxSag_Ax)
    
    coronal = args.input[1]
    cormask = args.imask[1]
    imcor, inmask = loadimages(coronal,cormask)
    imgChamferDistanceCor = ChamferDistance(inmask)
    ccorsag = createArrayOfChamferDistance(imgChamferDistanceCor,ptCorSag_Cor)
    
    
    EvolutionParameters = np.array(EvolutionParameters)
    ErrorEvolution = np.array(ErrorEvolution)
    EvolutionGridInter=np.array(EvolutionGridInter)
    EvolutionGridUnion=np.array(EvolutionGridUnion)
    EvolutionGridError = np.array(EvolutionGridError)
    EvolutionGridNbpoint = np.array(EvolutionGridNbpoint)
    costGlobal = np.array(costGlobal)
    
    
    file = data + 'v2'
    if not os.path.exists(file):
        os.makedirs(file)
        
    strEE = file + '/ErrorEvolution.dat'
    ErrorEvolution.tofile(strEE)
    
    strEGE = file + '/EvolutionGridError.dat'
    EvolutionGridError.tofile(strEGE)
    
    strEGN = file + '/EvolutionGridNbpoint.dat'
    EvolutionGridNbpoint.tofile(strEGN)
    
    strEGI = file + '/EvolutionGridInter.dat'
    EvolutionGridInter.tofile(strEGI)
    
    strEGU = file + '/EvolutionGridUnion.dat'
    EvolutionGridUnion.tofile(strEGU)
    
    strEP = file + '/EvolutionParameters.dat'
    EvolutionParameters.tofile(strEP)
    
    strCG = file + '/CostGlobal.dat'
    costGlobal.tofile(strCG)
    
    strEBac = file + '/ErrorBeforeAxCor.dat'
    errorAxCor_before.tofile(strEBac)
    
    strEBas = file + '/ErrorBeforeAxSag.dat'
    errorAxSag_before.tofile(strEBas)
    
    strEBcs = file + '/ErrorBeforeCorSag.dat'
    errorCorSag_before.tofile(strEBcs)
    
    strEAac = file + '/ErrorAfterAxCor.dat'
    errorAxCor_after.tofile(strEAac)
    
    strEAas = file + '/ErrorAfterAxSag.dat'
    errorAxSag_after.tofile(strEAas)
    
    strEAcs = file + '/ErrorAfterCorSag.dat'
    errorCorSag_after.tofile(strEAcs)

    strCac = file + '/colormapAC.dat'
    caxcor.tofile(strCac)
    
    strCas = file +'/colormapAS.dat'
    caxsag.tofile(strCas)
    
    strCcs = file + '/colormapCS.dat'
    ccorsag.tofile(strCcs)
    
    