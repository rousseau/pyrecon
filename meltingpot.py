#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:32:32 2017

@author: faisan
"""




from numba import jit
from math import cos,sin
import numpy as np
import nibabel as nib 
from scipy.ndimage.interpolation import affine_transform
from scipy.optimize import minimize

#critere a minimiser pour le recalage
#transfoRigide : les 3 premiers elements sont les angles d'Euler et les trois suivant la translation
#args : un objet de type dataRegistration qui contient les deux images 

def criteriumToMinimize(transfoRigide,args):
    M = createRotationMatrix(transfoRigide[0:3])
    T = transfoRigide[3:6]    
    res = affine_transform(args.Imtr.get_data(), M, offset=T, output_shape=None,  order=3, mode='constant', cval=0.0, prefilter=False)    
    critere = l2Criterium(res,args.ImRef.get_data())
    print critere
    return critere

class dataRegistration:
   def __init__(self, imref,imtr):
      self.ImRef     = imref      
      self.Imtr     = imtr
      
#critere L2 entre les images im1 et im2
def l2Criterium(im1,im2):
    difference = im2-im1
    return np.sum(difference*difference)


#calcule la matrice de rotation a partir des angles d Euler
# cf http://www.fil.ion.ucl.ac.uk/spm/doc/books/hbf2/pdfs/Ch2.pdf
def createRotationMatrix(angles):    
    c4 = cos(angles[0])
    s4 = sin(angles[0])
    c5 = cos(angles[1])
    s5 = sin(angles[1])
    c6 = cos(angles[2])
    s6 = sin(angles[2])    
    M = np.array([[c5*c6,c5*s6,s5],[-s4*s5*c6-c4*s6,-s4*s5*s6+c4*c6,s4*c5],[-c4*s5*c6+s4*s6,-c4*s5*s6-s4*c6,c4*c5]])    
    return M


#calcule le barycentre d'une image
@jit
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



imgtr = nib.load("/home/miv/faisan/ParcellisationCorticale/c2S064_T1.nii.gz")

#creation d'une image transformée
barycentre = computeBarycentre(imgtr.get_data())
M = createRotationMatrix([0.5,0.2,0.9])
T = [10,20,5] + barycentre - M.dot(barycentre)
res = affine_transform(imgtr.get_data(), M, offset=T, output_shape=None,  order=3, mode='constant', cval=0.0, prefilter=False)
img = nib.Nifti1Image(res, imgtr.affine, header=imgtr.header)
nib.save(img, '/home/miv/faisan/Rousseau/test.nii.gz')

imgref = nib.load('/home/miv/faisan/Rousseau/test.nii.gz')

#recalage d'une image
imageARecaler = dataRegistration(imgref,imgtr)

#initialisation
centreRef = computeBarycentre(imgref.get_data());
centreTr  = computeBarycentre(imgtr.get_data());

x0  = np.zeros(6);
x0[0] = 0.5
x0[1] = 0.2
x0[2] = 0.9

M = createRotationMatrix(x0[0:3])
x0[3:6] = centreTr - M.dot(centreRef) #MXref+T = Xtr

dx  = np.ones(6) * 0.001;
dx[3] = 1
dx[4] = 1
dx[5] = 1
simplex = createSimplex(x0,dx)
res = minimize(criteriumToMinimize, x0, imageARecaler, method='Nelder-Mead', options={'initial_simplex' : simplex, 'xtol': 1e-2, 'disp': True})
print res

