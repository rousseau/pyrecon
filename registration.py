#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:32:32 2017

@author: faisan
"""


from criterium import criteriumToMinimize,translateToRealParameter,translateToFalseParameter,dataRegistration
from transformation import createRotationMatrix,matriceCoordHomogene,sequenceofundersampling,underSample,createTransfoInmmWord,computeBarycentre
from algoFSL import iterateOverCoarseGrid,doubleTaille,computeOnFinerGrid,searchMaxima,createSimplex
from numba import jit
from math import cos,sin,pi
import numpy as np
import nibabel as nib 
from scipy.ndimage.interpolation import affine_transform,map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import minimize
from sklearn.metrics import mutual_info_score
from numpy.linalg import inv
import sys

if __name__ == '__main__':  
    imgref  = nib.load(sys.argv[1])
    imgtr = nib.load(sys.argv[2])
    filenameres    = sys.argv[3]
    filenameres2    = sys.argv[4]
    echelleBasseenmm = float(sys.argv[5])

    volumePixel = np.zeros(6)
    volumePixel[0:3]=np.asarray(imgtr.header.get_zooms())
    volumePixel[3:6]=np.asarray(imgref.header.get_zooms())
    s = sequenceofundersampling(volumePixel,echelleBasseenmm)
    
    
    size_tr = s[0,0:3]
    size_ref = s[0,3:6]
    imgtr_down = underSample(imgtr,size_tr)    
    imgref_down = underSample(imgref,size_ref)
    
    #calcul des matrices pour se retrouver dans un espace millimétrique    
    Mtr = createTransfoInmmWord(size_tr)           
    Mref = createTransfoInmmWord(size_ref)           
    Ktr = 2/min(np.min(size_tr),np.min(size_ref))/1000
    Kth = max(np.max(imgtr_down.shape),np.max(imgref_down.shape))*3.1415/360/1000
    Nbin = pow(imgref_down.shape[0]*imgref_down.shape[1]*imgref_down.shape[2],1/3.)
    

    imageARecaler = dataRegistration(size_ref,size_tr,imgref_down,imgtr_down,Mref,Mtr,0,Kth,Ktr,np.round(Nbin)) 
    
    
    print ("--------------------------------------------------------------------------")
    print ("-----------------------------   scale 0  ---------------------------------")
    print ("------------   size pixels image reference  : " + str(size_ref) + "  ------------ ")
    print ("------------   size pixels image transforme : " + str(size_tr) + "   ------------ ")
    print ("--------------------------------------------------------------------------")    
    
    
    
    
    
    centreTr = computeBarycentre(imgtr_down) 
    centreRef = computeBarycentre(imgref_down) 
    centreTr = np.append(centreTr,1)                                 
    centreRef = np.append(centreRef,1) 
    
    M = 10
    minimal = -45
    maximal = 45
    cyclic = True
    tx,ty,tz,euler1,euler2,euler3 = iterateOverCoarseGrid(M,imageARecaler,centreTr,centreRef,cyclic,minimal,maximal)
    txx,tyy,tzz,euler1,euler2,euler3 = doubleTaille(tx,ty,tz,cyclic,minimal,maximal,M)
    criterium = computeOnFinerGrid(txx,tyy,tzz,euler1,euler2,euler3,imageARecaler)    
    
    
    
    
    listeMaximum,listeCriterium = searchMaxima(criterium,cyclic)
    print listeMaximum
    print listeCriterium
    if len(listeMaximum) == 0:
        print "pas de maximum local !!"
        exit(0)
        
    listeParameter = []
    listeCriterium2 = []
    for i in range(len(listeMaximum)):        
        indices = listeMaximum[i]        
        x,y,z=indices
        x0 = np.zeros(6)
        x0[0] = euler1[x]
        x0[1] = euler2[y]
        x0[2] = euler3[z]
        x0[3] = txx[x][y][z]
        x0[4] = tyy[x][y][z]
        x0[5] = tzz[x][y][z]
        
        print ("initialisation "+ str(i) +'/' + str(len(listeMaximum)-1))
        print ("before")        
        print ("  rotation: " + str(x0[0:3])) 
        print ("  translation: " + str(x0[3:6]) ) 
        
        x0 = translateToFalseParameter(x0,imageARecaler)
        
        print ("  cost :  " + str( criteriumToMinimize(x0,imageARecaler)) + ' ' + str(listeCriterium[i]))
                    
        sizeSimplex = [euler1[1]-euler1[0],euler2[1]-euler2[0],euler3[1]-euler3[0],0.01,0.01,0.01]
        simplex = createSimplex(x0,sizeSimplex,imageARecaler) 
        res = minimize(criteriumToMinimize, x0, imageARecaler, method='Nelder-Mead', options={'initial_simplex' : simplex, 'xtol': 0.0005, 'disp': False})    
        #res = minimize(criteriumToMinimize, x0, imageARecaler, method='Powell',options={'xtol':0.01})
        
        value = criteriumToMinimize(res.x,imageARecaler) 
        
        res.x[0:3] = (res.x[0:3]-imageARecaler.offset)/imageARecaler.Ktheta
        res.x[3:6] = (res.x[3:6]-imageARecaler.offset)/imageARecaler.Ktrans
                     
        listeParameter.append(res.x)
        listeCriterium2.append(value)
        print ("after")
        print ("  rotation: " + str(res.x[0:3])) 
        print ("  translation: " + str(res.x[3:6]) )                
        print ("  cost:  " + str(value) )
        
    best = np.argsort(listeCriterium2)

    #on passe a la resolution suivante
    
    resolution = 1
    size_tr = s[resolution,0:3]
    size_ref = s[resolution,3:6]
    imgtr_down = underSample(imgtr,size_tr)    
    imgref_down = underSample(imgref,size_ref)
    
    #calcul des matrices pour se retrouver dans un espace millimétrique    
    Mtr = createTransfoInmmWord(size_tr)           
    Mref = createTransfoInmmWord(size_ref)           
    Ktr = 2/min(np.min(size_tr),np.min(size_ref))/1000
    Kth = max(np.max(imgtr_down.shape),np.max(imgref_down.shape))*3.1415/360/1000
    Nbin = pow(imgref_down.shape[0]*imgref_down.shape[1]*imgref_down.shape[2],1/3.)
    imageARecaler = dataRegistration(size_ref,size_tr,imgref_down,imgtr_down,Mref,Mtr,0,Kth,Ktr,np.round(Nbin)) 
    
    print ("--------------------------------------------------------------------------")
    print ("-----------------------------   scale 1  ---------------------------------")
    print ("------------   size pixels image reference  : " + str(size_ref) + "  ------------ ")
    print ("------------   size pixels image transforme : " + str(size_tr) + "   ------------ ")
    print ("--------------------------------------------------------------------------")
    
    nbreMultistart = min(3,len(best))
    minimalValue = 100000
    for i in range(nbreMultistart):
        x0 = listeParameter[best[i]]
        print ("maximum "+ str(i) +'/' + str(nbreMultistart-1))
        print("before")
        print ("  rotation: " + str(x0[0:3])) 
        print ("  translation: " + str(x0[3:6]) )        

        x0[0:3] = x0[0:3] * imageARecaler.Ktheta + imageARecaler.offset
        x0[3:6] = x0[3:6] * imageARecaler.Ktrans + imageARecaler.offset

        print ("  cost:  " + str( criteriumToMinimize(x0,imageARecaler)) )
        
        
        
        simplex = createSimplex(x0,np.full(6,0.005),imageARecaler) 
        res = minimize(criteriumToMinimize, x0, imageARecaler, method='Nelder-Mead', options={'initial_simplex' : simplex, 'xtol': 0.0005, 'disp': False})    
#      res = minimize(criteriumToMinimize, x0, imageARecaler, method='Powell',options={'xtol':0.01})
           
        better = res
        bestofIteration = criteriumToMinimize(better.x,imageARecaler) 
        
                
        simplex2 = createSimplex(res.x,np.full(6,0.002),imageARecaler) #taille de 50mm
        for multistart in range(simplex2.shape[0]):
             simplex3 = createSimplex(simplex2[i],np.full(6,0.002),imageARecaler) #taille de 50mm
             res = minimize(criteriumToMinimize, x0, imageARecaler, method='Nelder-Mead', options={'initial_simplex' : simplex3, 'xtol':  0.0005, 'disp': False})    
             value = criteriumToMinimize(res.x,imageARecaler) 
             if value<bestofIteration:
                 value = bestofIteration
                 better = res
                 
                
        res = better
        
        res.x[0:3] = (res.x[0:3]-imageARecaler.offset)/imageARecaler.Ktheta
        res.x[3:6] = (res.x[3:6]-imageARecaler.offset)/imageARecaler.Ktrans
        print("after")
        print ("  rotation: " + str(res.x[0:3])) 
        print ("  translation: " + str(res.x[3:6]) ) 
        print ("  cost:  " + str( bestofIteration) )
                
        if value<minimalValue:
            xresponse = res.x
            minimalValue = bestofIteration
            

    #on itere sur les resolutions suivantes
    for resolution in range(s.shape[0]-2):        
        size_tr = s[resolution+2,0:3]
        size_ref = s[resolution+2,3:6]
        if (resolution+2) == (s.shape[0]-1): #on est a la derniere resolution
           imgtr_down = imgtr.get_data().copy()
           imgref_down = imgref.get_data().copy()
        else:
            imgtr_down = underSample(imgtr,size_tr)    
            imgref_down = underSample(imgref,size_ref)

        print ("--------------------------------------------------------------------------")
        print ("-----------------------------   scale" + str(resolution+2) + "  ---------------------------------")
        print ("------------   size pixels image reference  : " + str(size_ref) + "  ------------ ")
        print ("------------   size pixels image transforme : " + str(size_tr) + "   ------------ ")
        print ("--------------------------------------------------------------------------")    


    
        #calcul des matrices pour se retrouver dans un espace millimétrique    
        Mtr = createTransfoInmmWord(size_tr)           
        Mref = createTransfoInmmWord(size_ref)  
        
        Ktr = 2/min(np.min(size_tr),np.min(size_ref))/1000
        Kth = max(np.max(imgtr_down.shape),np.max(imgref_down.shape))*3.1415/360/1000
        Nbin = pow(imgref_down.shape[0]*imgref_down.shape[1]*imgref_down.shape[2],1/3.)
        imageARecaler = dataRegistration(size_ref,size_tr,imgref_down,imgtr_down,Mref,Mtr,0,Kth,Ktr,np.round(Nbin)) 
        print("before")
        print ("  rotation: " + str(xresponse[0:3])) 
        print ("  translation: " + str(xresponse[3:6]))
        xresponse[0:3] = xresponse[0:3] * imageARecaler.Ktheta + imageARecaler.offset
        xresponse[3:6] = xresponse[3:6] * imageARecaler.Ktrans + imageARecaler.offset

        print ("  cost:  " + str( criteriumToMinimize(xresponse,imageARecaler)) )
    
        
        
        
        simplex = createSimplex(xresponse,np.full(6,0.004),imageARecaler) #taille de 50mm
        res = minimize(criteriumToMinimize, xresponse, imageARecaler, method='Nelder-Mead', options={'initial_simplex' : simplex, 'xtol': 0.0005, 'disp': False})            
        #res = minimize(criteriumToMinimize, x0, imageARecaler, method='Powell',options={'xtol':0.01})

        value = criteriumToMinimize(res.x,imageARecaler)
        
        
        xresponse = res.x
        xresponse[0:3] = (xresponse[0:3]-imageARecaler.offset)/imageARecaler.Ktheta
        xresponse[3:6] = (xresponse[3:6]-imageARecaler.offset)/imageARecaler.Ktrans
        print("after")
        print ("  rotation: " + str(xresponse[0:3])) 
        print ("  translation: " + str(xresponse[3:6]))
        print ("  cost:  " + str(value) )




   
    imageARecaler = dataRegistration(size_ref,size_tr,imgref.get_data(),imgtr.get_data(),Mref,Mtr,0,Kth,Ktr,np.round(Nbin)) 
    
    x0 = xresponse    
    M = createRotationMatrix(x0[0:3])
    T = x0[3:6]    
    M = matriceCoordHomogene(M,T)    
    M = imageARecaler.invMtr.dot(M).dot(imageARecaler.Mref)    
    Msave = M.copy()
    Msave = inv(Msave)
    
    T = M[0:3,3]/M[3,3]
    Mf = M[0:3,0:3]/M[3][3] 
    
    res = affine_transform(imageARecaler.Imtr, Mf, offset=T, output_shape=imageARecaler.ImRef.shape,  order=3, mode='constant', cval=0, prefilter=False)
            
    img = nib.Nifti1Image(res, imgref.affine, header=imgref.header)
    nib.save(img, filenameres) 

    T = Msave[0:3,3]/Msave[3,3]
    Mf = Msave[0:3,0:3]/Msave[3][3] 
    
    res = affine_transform(imageARecaler.ImRef, Mf, offset=T, output_shape=imageARecaler.Imtr.shape,  order=3, mode='constant', cval=0, prefilter=False)
    img = nib.Nifti1Image(res, imgtr.affine, header=imgtr.header)
    nib.save(img, filenameres2) 



