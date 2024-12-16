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
from copy import deepcopy
import argparse
import nibabel 
from scipy.optimize import minimize
import numpy as np
from scipy.ndimage.interpolation import affine_transform
import math
import multiprocessing
from numpy.random import uniform
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.interpolation import map_coordinates

#From  : https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429
from scipy import ndimage
EPS = np.finfo(float).eps


  


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


def computeMref2in(x,datareg):
  #Compute transform from refimage to inputimage  
  M = compute_affine_matrix(translation=x[0:3], angles=x[3:6]) #estimated transform
  return np.dot(datareg.Mw2in, np.dot(datareg.Mc2w, np.dot( M,np.dot(datareg.Mw2c,datareg.Mref2w) ) ) )
  



def homogeneousMatrix(M,T):
    Mres = np.zeros((4,4))
    Mres[0:3,0:3]=M
    Mres[3,3]=1
    Mres[0:3,3]=T
    return Mres

def imageResampling(inputimage, outputspacing, order=1):

  M1 = np.diag(outputspacing)
  T1 = outputspacing/2
  inputspacing = inputimage.header.get_zooms()[0:3]
  M2 = np.diag(np.asarray(inputspacing))
  T2 = np.asarray(inputspacing) / 2
  
  M = np.linalg.inv(homogeneousMatrix(M2,T2)).dot(homogeneousMatrix(M1,T1))
  outputaffine = np.dot(inputimage.affine,M)
  
  T = M[0:3,3]
  M = M[0:3,0:3]
  
  zoom = np.diag(M[0:3,0:3])
  outputshape = np.ceil((inputimage.get_data().shape[0:3])/zoom).astype(int)
  inputarray = inputimage.get_data()
  inputarray = np.reshape(inputarray,inputarray.shape[0:3])
  outputarray = affine_transform(inputarray, M, offset=T, output_shape=outputshape,  order=order, mode='constant', cval=0.0, prefilter=False)
  
  return nibabel.Nifti1Image(outputarray, outputaffine)



  
def image2slices(image):
  print('image2slices')
  slices = []
  for z in range(image.get_data().shape[2]):
    slicearray = image.get_data()[:,:,z]
    M = np.eye(4)
    M[2,3] = z
    sliceaffine = np.dot(image.affine,M)
    slices.append(nibabel.Nifti1Image(slicearray, sliceaffine))
  return slices

def slices2image(slices,refimage):  
  print('slices2image')
  outputarray = np.zeros(refimage.get_data().shape[0:3])
  for s in slices:
    M = np.dot(np.linalg.inv(s.affine),refimage.affine)
    T = M[0:3,3]
    M = M[0:3,0:3]
    #Basic, long and stupid way to put slices into reference space  
    outputarray += affine_transform(s.get_data(), M, offset=T, output_shape=outputarray.shape,  order=1, mode='constant', cval=0.0, prefilter=False)
    
    
  return nibabel.Nifti1Image(outputarray, refimage.affine)

#Object that is used to put all the stuff needed for registration...
class dataReg:
  def __init__(self,refimage,inputimage,refmask,inputmask,crefimw,s,criterium,resampling):
    if resampling == True:
      self.refimage   = imageResampling(refimage, s[3:6],order=1)
      self.inputimage = imageResampling(inputimage, s[0:3],order=1)
    else:      
      self.refimage = refimage
      self.inputimage = inputimage
    #Get array : using nibabel, 3D image may have 4D shape
    self.refarray   = np.reshape(self.refimage.get_data(),self.refimage.get_data().shape[0:3])
    self.inputarray = np.reshape(self.inputimage.get_data(),self.inputimage.get_data().shape[0:3])
    self.Mw2in      = np.linalg.inv(self.inputimage.affine) # world to input image
    self.Mref2w     = self.refimage.affine #reference to world
    
    if resampling == True:      
      self.refmask    = imageResampling(refmask, s[3:6],order=0).get_data()
      self.inputmask  = imageResampling(inputmask, s[0:3],order=0).get_data()
    else:      
      self.refmask = refmask.get_data()
      self.inputmask = inputmask.get_data()
      
      
    self.refindex   = self.refmask>0
    
    self.centerrot  = crefimw
    self.Mw2c       = np.eye(4)
    self.Mc2w       = np.eye(4)  
    self.Mw2c[0:3,3]= -self.centerrot[0:3]
    self.Mc2w[0:3,3]=  self.centerrot[0:3]

    
    #criterium
    self.criterium = criterium    
    #self.criterium.setBinNumber(self.refarray.shape)
    
    #normaization of the parameters to optimize
    Ktranslation = 2/min(np.min(s),np.min(s))  #so that precision required is 1
    Ktheta       = max(np.max(self.refarray.shape),np.max(self.refarray.shape))*3.1415/180 #so that precision required is 1
    
                      
    self.K = np.array([Ktranslation,Ktranslation,Ktranslation,Ktheta,Ktheta,Ktheta]) 
    self.Kcte = 10000





def applyTransformbyBlock(input_image, transform,  reference_image=None, order=None):
  if order is None:
    order = 1
  if reference_image is None:
    reference_image = input_image  
  output_data = np.zeros(reference_image.get_data().shape)
#on peut appliquer la transformation 2 coupes/ 2 coupes
#  for k in range(0,output_data.shape[2],2):
#    indiceMaxK = np.min((k+1,output_data.shape[2]-1))
#    toto = applyTransform(input_image, transform,reference_image, order,k,indiceMaxK)    
#    output_data[:,:,k:indiceMaxK+1] =applyTransform(input_image, transform,reference_image, order,k,indiceMaxK)
#on applique la transformation  coupe /  coupe
  for k in range(0,output_data.shape[2]):    
    toto = applyTransform(input_image, transform,reference_image, order,k,k)    
    output_data[:,:,k:k+1] =applyTransform(input_image, transform,reference_image, order,k,k)
  
  
  return output_data


def applyTransform(input_image, transform,  reference_image=None, order=None,kmin=None,kmax = None):  
  if reference_image is None:
    reference_image = input_image
  #set the interpolation order to 1 if not specified
  if order is None:
    order = 1
 
  ref_data_size = reference_image.get_data().shape 
  
  #create index for the reference space
  i = np.arange(0,ref_data_size[0])
  j = np.arange(0,ref_data_size[1])
  if kmin is None:
    k = np.arange(0,ref_data_size[2])
  else:
    k = np.arange(kmin,kmax+1)
    
  iv,jv,kv = np.meshgrid(i,j,k,indexing='ij')
  
  iv = np.reshape(iv,(-1))
  jv = np.reshape(jv,(-1))
  kv = np.reshape(kv,(-1))
  
  #compute the coordinates in the input image
  pointset = np.zeros((4,iv.shape[0]))
  pointset[0,:] = iv
  pointset[1,:] = jv
  pointset[2,:] = kv
  pointset[3,:] = np.ones((iv.shape[0]))
  
  #compute the transformation
  pointset = np.dot(transform,pointset) 
  
  #compute the interpolation
  val = np.zeros(iv.shape)            
  map_coordinates(input_image.get_data(),[pointset[0,:],pointset[1,:],pointset[2,:]],output=val,order=order,prefilter=True)
  
  if  kmin is None:
    output_data = np.reshape(val,ref_data_size)
  else:
    size = list(ref_data_size)
    size[2] = kmax-kmin+1
    output_data = np.reshape(val,tuple(size))
    
  return output_data

  
if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('-i', '--input', help='Input Image to register on the reference image', type=str, required=True)
  parser.add_argument('-r', '--ref', help='Reference Image', type=str, required=True)
  parser.add_argument('-o', '--output', help='Output Image', type=str, required = True)
  parser.add_argument('--inmask', help='Mask of the input image', type=str)
  parser.add_argument('--refmask', help='Mask of the reference image', type=str)  
  parser.add_argument('--padding', help='Padding value used when no mask is provided', type=float, default=-np.inf)
  args = parser.parse_args()
  
  #Loading images    
  inputimage = nibabel.load(args.input)
  refimage   = nibabel.load(args.ref)
  
  #manage possible 4D shape when reading 3D image using nibabel
  inputimage = nibabel.Nifti1Image(np.squeeze(inputimage.get_data()), inputimage.affine) 
  refimage = nibabel.Nifti1Image(np.squeeze(refimage.get_data()), refimage.affine) 
  
  if args.inmask is not None :
    inputmask = nibabel.load(args.inmask)
  else:
    print('Creating mask images using the following padding value:',str(args.padding))
    inputdata = np.zeros(inputimage.get_data().shape)
    inputdata[inputimage.get_data() > args.padding] = 1
    inputdata = np.reshape(inputdata,inputdata.shape[0:3])
    inputmask = nibabel.Nifti1Image(inputdata, inputimage.affine) 

  if args.refmask is not None :
    refmask = nibabel.load(args.refmask)
  else:
    print('Creating mask images using the following padding value:',str(args.padding))
    refdata = np.zeros(refimage.get_data().shape)
    refdata[refimage.get_data() > args.padding] = 1
    refdata = np.reshape(refdata,refdata.shape[0:3])
    refmask = nibabel.Nifti1Image(refdata, refimage.affine) 
  
 #Center of rotation : center of reference image expressed in world coordinate
  refcom     = np.asarray(center_of_mass(np.multiply(refimage.get_data(),refmask.get_data())))
  refcom     = np.concatenate((refcom,np.array([1])))
  refcomw    = np.dot(refimage.affine,refcom) #ref center of mass expressed in world coordinate
                       
  #Initialization of translation using center of mass    
  inputcom   = np.asarray(center_of_mass(np.multiply(inputimage.get_data(),inputmask.get_data())))    
  inputcom   = np.concatenate((inputcom,np.array([1])))    
  inputcomw  = np.dot(inputimage.affine,inputcom) #input center of mass expressed in world coordinate        
  
  currentx = np.zeros(6)
  currentx[0:3] = (inputcomw-refcomw)[0:3]
  crefimw = refcomw
  crefimw[3] = 1 
 
  currentx= currentx + [10,5,2,10,-8,6]
  
  volumePixel = np.zeros(6)
  volumePixel[0:3]=np.asarray(inputimage.header.get_zooms())
  volumePixel[3:6]=np.asarray(refimage.header.get_zooms())
  datareg = dataReg(refimage,inputimage,refmask,inputmask,crefimw,volumePixel,0,False)
  Mref2in = computeMref2in(currentx,datareg)
  
  warpedarray = affine_transform(np.reshape(inputimage.get_data(),inputimage.get_data().shape[0:3]), Mref2in[0:3,0:3], offset=Mref2in[0:3,3], output_shape=refimage.get_data().shape[0:3],  order=3, mode='constant', cval=0, prefilter=True)     
  nibabel.save(nibabel.Nifti1Image(warpedarray, refimage.affine),"titi.nii.gz")#args.output)
  
  output_data =  applyTransform(inputimage, Mref2in,  refimage, order=3)
  nibabel.save(nibabel.Nifti1Image(output_data, refimage.affine),"toto.nii.gz")#args.output)  
  
  output_data2 = applyTransformbyBlock(inputimage, Mref2in,  refimage, order=3) 
  nibabel.save(nibabel.Nifti1Image(output_data2, refimage.affine),"tata.nii.gz")#args.output)  
  
  test = warpedarray - output_data
  print np.mean(np.abs(test))

  test = warpedarray - output_data2
  print np.mean(np.abs(test))
    
 #TEST to convert image to slices  
#  slices = image2slices(inputimage)
#  inputimage = slices[20]
#  toto = slices2image(slices,refimage)
#  nibabel.save(toto,'toto.nii.gz')

  #TODO ? blur the data before downsampling
  #TODO ? add linear transform for parameters
  #TODO : define a fast and simple strategy for multiresolution registration using random inits.
  #TODO : initialization for slice to volume maybe different from 3D/3D case
  #TODO : manage slice image resampling
