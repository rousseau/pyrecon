#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:32:32 2017

@author: faisan
"""



from scipy import ndimage
from transformation import createRotationMatrix,matriceCoordHomogene
import numpy as np
from scipy.ndimage.interpolation import affine_transform
from sklearn.metrics import mutual_info_score
from numpy.linalg import inv
EPS = np.finfo(float).eps


#critere a minimiser pour le recalage
#transfoRigide : si transfo a 6 éléments : les 3 premiers elements sont les angles d'Euler et les trois suivant la translation
#si transfo a 3 élements ce sont les élements de la translation
#args : un objet de type dataRegistration qui contient les deux images, et différentes informations
def criteriumToMinimize(transfoRigide,args):      
    
    param = translateToRealParameter(transfoRigide,args) 
    
    if param.shape[0] == 3:
        M = args.Mrot
        T = param[0:3]
    else:                
        M = createRotationMatrix(param[0:3])
        T = param[3:6] 
        
    M = matriceCoordHomogene(M,T)    
    M = args.invMtr.dot(M).dot(args.Mref)     
    
    T = M[0:3,3]/M[3,3]
    Mf = M[0:3,0:3]/M[3][3] 
    
    res = affine_transform(args.Imtr, Mf, offset=T, output_shape=args.ImRef.shape,  order=1, mode='constant', cval=-10001.0, prefilter=False)    
    
    value1 = res[res>-10000] 
    value2 = args.ImRef[res>-10000]
    
    total = args.ImRef.shape [0] * args.ImRef.shape [1] * args.ImRef.shape [2] / 10
                             
    if value1.shape[0]<total:
        return 10000
    
    if np.min(value1)+1e-2 >= np.max(value1):
        return 10000
    
    if np.min(value2)+1e-2 >= np.max(value2):
        return 10000
        
    critere = mutual_information_2d(value1,value2, args.Nbin) #calc_MI(value1,value2, args.Nbin)      
    
    return -critere 

#cree le simplex a partir d'une initialisation et de dx (grandeur du simplex pour chaque dimension)
def createSimplex(x,dx,args):
    simplex = np.zeros((  len(x)+1,len(x) ))
    
    if criteriumToMinimize(x,args) > 9999:
        print("le centre du simplex n est pas bon")
        #exit(0)
    for i in range(len(x)+1):    
        simplex[i,:] = x.copy()
    for i in range(len(x)):            
         simplex[i+1][i] = simplex[i+1][i] + dx[i]
            
    return simplex


class dataRegistration:
    def __init__(self, sizeref,sizetr,imref,imtr,mref,mtr,Mr,ktheta,ktrans,nbin):
        self.sizeTr   = sizetr
        self.sizeRef  = sizeref
        self.ImRef    = imref      
        self.Imtr     = imtr
        self.Mref     = mref
        self.invMtr   = inv(mtr)
        self.Mtr      = mtr
        self.Mrot     = Mr
        self.Ktheta   = ktheta
        self.Ktrans   = ktrans
        self.offset   = 50
        self.Nbin     = nbin

                  
def calc_MI(x, y, bins):    
    try: 
        c_xy = np.histogram2d(x, y, bins)[0]
    except:
        return -10000
            
    #print c_xym
    mi = mutual_info_score(None, None, contingency=c_xy)    
    return mi

#code pris a https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429
def mutual_information_2d(x, y, bins, normalized=True):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    nmi: float
        the computed similariy measure
    """
    try: 
        jh = np.histogram2d(x, y, bins)[0]
    except:
        return -10000
    

    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    sigma = 1 #on utilise sigma = 1 si nbreBin = 1
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
                                 output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
               - np.sum(s2 * np.log(s2)))

    return mi




def translateToRealParameter(x,imageARecaler):
    if x.shape[0] == 3:       
        return (x-imageARecaler.offset)/imageARecaler.Ktrans    
    else:
        y = x.copy()
        y[0:3] = (x[0:3]-imageARecaler.offset)/imageARecaler.Ktheta
        y[3:6] = (x[3:6]-imageARecaler.offset)/imageARecaler.Ktrans
        return y

def translateToFalseParameter(x,imageARecaler):
    y = x.copy()
    if x.shape[0] == 3:
        y = y * imageARecaler.Ktrans + imageARecaler.offset    
    else:        
        y[0:3] = y[0:3] * imageARecaler.Ktheta + imageARecaler.offset
        y[3:6] = y[3:6] * imageARecaler.Ktrans + imageARecaler.offset
    return y

