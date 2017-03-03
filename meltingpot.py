#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:32:32 2017

@author: faisan
"""




#from numba import jit
from math import cos,sin,pi
import numpy as np
import nibabel as nib 
from scipy.ndimage.interpolation import affine_transform
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import minimize
from sklearn.metrics import mutual_info_score
from numpy.linalg import inv

def underSample(im1,desiredVolumePixel):
    volumePixel =  np.asarray(im1.header.get_zooms())
    zoom = desiredVolumePixel / volumePixel;
    M = np.diag(zoom)    
    filtered_data = gaussian_filter(im1.get_data(),zoom*0.8);
    res = affine_transform(filtered_data, M, offset=0, output_shape=None,  order=3, mode='constant', cval=0.0, prefilter=False)
    
    MaxCoord = np.ceil(im1.get_data().shape/zoom)
    MaxCoord = MaxCoord.astype(int)
    res2 = res[0:MaxCoord[0],0:MaxCoord[1],0:MaxCoord[2]]     
    return res2
    

def sequenceofundersampling(volumePixel,largestSize):
    numberMaximalOfResolutions = 100

    sequence = np.zeros((numberMaximalOfResolutions,6))
    sequence[0,:] = np.full(6,largestSize)    
    
    for i in range(numberMaximalOfResolutions-1):        
        sequence[i+1,:] = sequence[i,:]/2        
    
    for j in range(6):
        sequence[:,j] = np.clip(sequence[:,j], volumePixel[j], 1000000)
    
    take = 0
    for i in range(numberMaximalOfResolutions):        
        if np.isclose(sequence[i,:],volumePixel,rtol=0.001, atol=0.1).all() == True:     
            sequence[i,:] = volumePixel
            take = i+1
            break
            
    
    
    sequenceFinal = sequence[0:take,:] 
    return sequenceFinal
    

#critere a minimiser pour le recalage
#transfoRigide : les 3 premiers elements sont les angles d'Euler et les trois suivant la translation
#args : un objet de type dataRegistration qui contient les deux images 
def criteriumToMinimize(transfoRigide,args):
    M = createRotationMatrix(transfoRigide[0:3])
    T = transfoRigide[3:6]    
    M = matriceCoordHomogene(M,T)    
    M = args.invMref.dot(M).dot(args.Mtr) 
    
    T = M[0:3,3]/M[3,3]
    Mf = M[0:3,0:3]/M[3][3] 
    
    res = affine_transform(args.Imtr, Mf, offset=T, output_shape=args.ImRef.shape,  order=3, mode='constant', cval=-10001.0, prefilter=False)    
    
    value1 = res[res>-10000]
    value2 = args.ImRef[res>-10000]
    critere = calc_MI(value1,value2, 64)      
    return -critere

class dataRegistration:
   def __init__(self, imref,imtr,invmref,mtr):
      self.ImRef    = imref      
      self.Imtr     = imtr
      self.invMref  = invmref
      self.Mtr      = mtr      
      


def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


#critere L2 entre les images im1 et im2
def l2Criterium(im1,im2):
    difference = im2-im1
    return np.sum(difference*difference)


#calcule la matrice de rotation a partir des angles d Euler
# cf http://www.fil.ion.ucl.ac.uk/spm/doc/books/hbf2/pdfs/Ch2.pdf
def createRotationMatrix(eulers):
    angles = eulers.copy()
    angles = angles / 360 * pi
    c4 = cos(angles[0])
    s4 = sin(angles[0])
    c5 = cos(angles[1])
    s5 = sin(angles[1])
    c6 = cos(angles[2])
    s6 = sin(angles[2])    
    M = np.array([[c5*c6,c5*s6,s5],[-s4*s5*c6-c4*s6,-s4*s5*s6+c4*c6,s4*c5],[-c4*s5*c6+s4*s6,-c4*s5*s6-s4*c6,c4*c5]])    
    return M


#calcule le barycentre d'une image
#@jit
def computeBarycentre(data):
    barycentre = np.zeros(3)
    compteur = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                barycentre[0] +=  data[i][j][k]*i
                barycentre[1] +=  data[i][j][k]*j
                barycentre[2] +=  data[i][j][k]*k                             
                compteur += data[i][j][k]
    return barycentre/compteur                
      

#cree le simplex a partir d'une initialisation et de dx (grandeur du simplex pour chaque dimension)
#l algo pourrait etre ameliore de maniere a ce que pour chaque point du simplex, les centres de gravites soient alignés
def createSimplex(x,dx):
    simplex = np.zeros((  len(x)+1,len(x) ))
    for i in range(len(x)+1):    
        simplex[i,:] = x.copy()
    for i in range(len(x)):    
        simplex[i+1][i] = simplex[i+1][i] + dx[i]
    return simplex


def matriceCoordHomogene(M,T):
    Mres = np.zeros((4,4))
    Mres[0:3,0:3]=M
    Mres[3,3]=1
    Mres[0:3,3]=T
    return Mres
    

imgref  = nib.load("/home/miv/faisan/patient16_exam02_T1GADO.nii.gz")
imgtr = nib.load("/home/miv/faisan/patient16_exam02_T2.nii.gz")

volumePixel1 =  np.asarray(imgtr.header.get_zooms())
volumePixel2 =  np.asarray(imgref.header.get_zooms())
volumePixel = np.zeros(6)
volumePixel[0:3]=volumePixel1
volumePixel[3:6]=volumePixel2

s = sequenceofundersampling(volumePixel,4)

for resolution in range(s.shape[0]):
    size_tr = s[resolution,0:3]
    size_ref = s[resolution,3:6]
    imgtr_down = underSample(imgtr,size_tr)    
    imgref_down = underSample(imgref,size_ref)
    
   
    #calcul des matrices pour se retrouver dans un espace millimétrique    
    M = np.diag(size_tr)
    T = size_tr/2
    Mtr = matriceCoordHomogene(M,T)       
    
    M = np.diag(size_ref)
    T = size_ref/2
    Mref = matriceCoordHomogene(M,T)       
    
    imageARecaler = dataRegistration(imgref_down,imgtr_down,inv(Mref),Mtr)    
    
# 
    #initialisation
    if resolution == 0:
        centreTr = computeBarycentre(imgtr_down) 
        centreRef = computeBarycentre(imgref_down) 
                                  
        centreTr = np.append(centreTr,1)                                 
        centreRef = np.append(centreRef,1)            
        
        T = Mref.dot(centreRef) - Mtr.dot(centreTr)        
        print T
        x0  = np.zeros(6)
        x0[3:6] = T[0:3]
        
        M = np.eye(4)
        M[0:3,3] = T[0:3]
        invMref = inv(Mref)
        titi = invMref.dot(M).dot(Mtr).dot(centreTr)                         
        toto =  centreRef                                            
        print (" (ca doit etre nul)" + str( titi-toto ))
        
    
    if resolution == 0:
        dx  = np.ones(6) * 100;
        
    else:
        dx  = np.ones(6) * 0.1;
        dx[3] = 3
        dx[4] = 3
        dx[5] = 3
          
    simplex = createSimplex(x0,dx)
    print ("--------------------------------------------------------------------------")
    print ("-----------------------------   scale" + str(resolution) + "  ---------------------------------")
    print ("------------   size pixels image reference  : " + str(size_ref) + "  ------------ ")
    print ("------------   size pixels image transforme : " + str(size_tr) + "   ------------ ")
    print ("--------------------------------------------------------------------------")    
    print ("cost before:  " + str( criteriumToMinimize(x0,imageARecaler)) )
    print ("  rotation: " + str(x0[0:3])) 
    print ("  translation: " + str(x0[3:6]))
    res = minimize(criteriumToMinimize, x0, imageARecaler, method='Nelder-Mead', options={'initial_simplex' : simplex, 'xtol': 1e-1, 'disp': False})    
    x0 = res.x    
    print ("cost after:  " + str( criteriumToMinimize(x0,imageARecaler)) )
    print ("  rotation: " + str(x0[0:3])) 
    print ("  translation: " + str(x0[3:6]) )
                                         
    
    
M = createRotationMatrix(x0[0:3])
T = x0[3:6]    
M = matriceCoordHomogene(M,T)    
M = imageARecaler.invMref.dot(M).dot(imageARecaler.Mtr) 
T = M[0:3,3]/M[3,3]
Mf = M[0:3,0:3]/M[3][3] 
res = affine_transform(imageARecaler.Imtr, Mf, offset=T, output_shape=imageARecaler.ImRef.shape,  order=3, mode='constant', cval=0, prefilter=False)
            
img = nib.Nifti1Image(res, imgtr.affine, header=imgtr.header)
nib.save(img, '/home/miv/faisan/Rousseau/res.nii.gz')
