#!/usr/bin/env python2
"""

  This software is governed by the CeCILL-B license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL-B
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".

  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited
  liability.

  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.

  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL-B license and that you accept its terms.

"""

import argparse
import nibabel 
from scipy.optimize import minimize
import numpy as np
from scipy.ndimage.interpolation import affine_transform
import math


#From  : https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429
from scipy import ndimage
EPS = np.finfo(float).eps

def mutual_information_2d(x, y, sigma=1, normalized=False):
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
  bins = (256, 256)

  jh = np.histogram2d(x, y, bins=bins)[0]

  # smooth the jh with a gaussian filter of given sigma
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

def compute_affine_matrix(translation=None, angles=None, scale=None, shear=None):
  """
  compute the affine matrix using a direct computation
  faster computation than numpy matrix multiplication
  """
  
  mat = np.identity(4)
  gx,gy,gz = 0.0,0.0,0.0
  sx,sy,sz = 1.0,1.0,1.0
  if translation is not None:
    mat[:3,3] = translation[:3]
  if angles is not None:
    ax = math.pi*angles[0]/180.0
    ay = math.pi*angles[1]/180.0
    az = math.pi*angles[2]/180.0
    cosx = math.cos(ax)
    cosy = math.cos(ay)
    cosz = math.cos(az)
    sinx = math.sin(ax)
    siny = math.sin(ay)
    sinz = math.sin(az)
  if shear is not None:
    gx = shear[0]
    gy = shear[1]
    gz = shear[2]
  if scale is not None:
    sx = scale[0]
    sy = scale[1]
    sz = scale[2]
    
  mat[0,0] = sx * cosy * (cosz + (gy*sinz) )
  mat[0,1] = sy * (cosy * (sinz + (gx * gy * cosz)) - (gz * siny) )
  mat[0,2] = sz * ( (gx * cosy * cosz) - siny)
  mat[1,0] = sx * (sinx * siny * (cosz + gy * sinz) - cosx * (sinz + (gy * cosz) ))
  mat[1,1] = sy * (sinx * siny * (sinz + (gx * gz * cosz) ) + cosx * (cosz - (gx * gy * sinz)) + (gz * sinx * cosy))
  mat[1,2] = sz * (sinx * cosy + (gx * (sinx * siny * cosz - cosx * sinz)))
  mat[2,0] = sx * (cosx * siny * (cosz + (gy * sinz)) + sinx * (sinz - (gy * cosz) ))
  mat[2,1] = sy * (cosx * siny * (sinz + (gx * gz * cosz)) - sinx * (cosz - (gx * gz * sinz)) + (gz * cosx * cosy) )
  mat[2,2] = sz * (cosx * cosy + (gx * ( (cosx * siny * cosz) + (sinx * sinz) )) )
  
  return mat

#Object that is used to put all the stuff needed for registration...
class dataReg(object):
  pass

def computeMref2in(x,datareg):
  #Compute transform from refimage to inputimage  
  M = compute_affine_matrix(translation=x[0:3], angles=x[3:6]) #estimated transform
  return np.dot(datareg.Mw2in,np.dot(M,datareg.Mref2w))

def createSimplex(x,dx):
  simplex = np.tile(x, (len(x)+1,1))
  for i in range(len(x)):            
    simplex[i+1][i] += dx[i]
  return simplex

def f(x,datareg):

  Mref2in = computeMref2in(x,datareg)
  
  #Apply current transform on input image
  warpedarray = affine_transform(datareg.inputarray, Mref2in[0:3,0:3], offset=Mref2in[0:3,3], output_shape=datareg.refarray.shape,  order=1, mode='constant', cval=np.nan, prefilter=False)    
  
  #Compute similarity criterion
  refval = datareg.refarray[~np.isnan(warpedarray)]
  inval  = warpedarray[~np.isnan(warpedarray)]
  
  res = np.mean(np.abs(refval-inval))
  #res = -mutual_information_2d(refval,inval, datareg.nbins)
  return res

def homogeneousMatrix(M,T):
    Mres = np.zeros((4,4))
    Mres[0:3,0:3]=M
    Mres[3,3]=1
    Mres[0:3,3]=T
    return Mres

def imageResampling(inputimage, outputspacing):
  #Need to check if the origin is computed correctly
  M1 = np.diag(outputspacing)
  T1 = outputspacing/2
  inputspacing = inputimage.header.get_zooms()[0:3]
  M2 = np.diag(np.asarray(inputspacing))
  T2 = np.asarray(inputspacing) / 2
  
  M = np.linalg.inv(homogeneousMatrix(M2,T2)).dot(homogeneousMatrix(M1,T1))
  T = M[0:3,3]
  M = M[0:3,0:3]
  
  zoom = np.diag(M[0:3,0:3])
  outputshape = np.ceil((inputimage.get_data().shape[0:3])/zoom).astype(int)
  inputarray = inputimage.get_data()
  inputarray = np.reshape(inputarray,inputarray.shape[0:3])
  outputarray = affine_transform(inputarray, M, offset=T, output_shape=outputshape,  order=3, mode='constant', cval=0.0, prefilter=False)
  
  outputaffine = inputimage.affine 
  outputaffine[0:3,0:3] = outputaffine[0:3,0:3] / inputspacing * outputspacing
  return nibabel.Nifti1Image(outputarray, outputaffine)
  
if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('-i', '--input', help='Input Image to register on the reference image', type=str, required=True)
  parser.add_argument('-r', '--ref', help='Reference Image', type=str, required=True)
  parser.add_argument('--inmask', help='Mask of the input image', type=str)
  parser.add_argument('--refmask', help='Mask of the reference image', type=str)  
  parser.add_argument('-o', '--output', help='Output Image', type=str, required = True)

  args = parser.parse_args()

  #Loading images    
  inputimage = nibabel.load(args.input)
  refimage   = nibabel.load(args.ref)
  
  if args.inmask is not None :
    inputmask = nibabel.load(args.inmask)
  if args.refmask is not None :
    refmask = nibabel.load(args.refmask)

  #TODO : If no mask is provided, we need to create one (using padding value or whole image). Masks have to been used for similarity computation
  
  #3D/3D, 2D/3D, 2D/n2D
  
  #x = [tx, ty, tz, rx, ry, rz]
  x = np.zeros((6)) 
  datareg = dataReg()
  datareg.refimage   = refimage
  datareg.inputimage = inputimage
  #Get array : using nibabel, 3D image may have 4D shape
  datareg.refarray   = np.reshape(refimage.get_data(),refimage.get_data().shape[0:3])
  datareg.inputarray = np.reshape(inputimage.get_data(),inputimage.get_data().shape[0:3])
  datareg.Mw2in      = np.linalg.inv(datareg.inputimage.affine) # world to input image
  datareg.Mref2w     = datareg.refimage.affine #reference to world

  datareg.nbins = 64
  x[0] = 5
  x[4] = 10
  dx = np.zeros((6)) 
  dx[0] = 20
  dx[1] = 20
  dx[2] = 20
  dx[3] = 10
  dx[4] = 10
  dx[5] = 10
  simplex = createSimplex(x,dx)
  
  ospacing = np.asarray([2,2,2])
  toto = imageResampling(inputimage, ospacing)
  nibabel.save(toto,args.output)

  #res = minimize(f,x,datareg, method='Nelder-Mead', options={'initial_simplex' : simplex, 'xatol': 0.01, 'disp': True})    
  #res = minimize(f,x,datareg, method='Nelder-Mead', options={'xatol': 0.01, 'disp': True})    

  #Apply estimated transform on input image
  #Mref2in = computeMref2in(res.x,datareg)
  #warpedarray = affine_transform(datareg.inputarray, Mref2in[0:3,0:3], offset=Mref2in[0:3,3], output_shape=datareg.refarray.shape,  order=1, mode='constant', cval=np.nan, prefilter=False)     
  #nibabel.save(nibabel.Nifti1Image(warpedarray, datareg.Mref2w),args.output)
