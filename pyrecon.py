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

#From  : https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429
from scipy import ndimage
EPS = np.finfo(float).eps

def mutual_information_2d(x, y, sigma=1, normalized=False, nbins=128):
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
  bins = (nbins, nbins)

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

#Objects definig the criteria

#For the L1 norm, a function setBinNumber has been used so that each object defining a criterium has the same interface
class L1:
  def compute(self,refval,inval):
    return np.mean(np.abs(refval-inval))
  def setBinNumber(self,size):
    pass

class L2:
  def compute(self,refval,inval):
    diff = refval-inval
    return np.mean(diff**2)
  def setBinNumber(self,size):
    pass
  
class IM:
  def __init__(self):
    self.nbins = 128
    self.sigma = 1
  def compute(self,refval,inval):
    return -mutual_information_2d(refval,inval, sigma=self.sigma, normalized=False, nbins=self.nbins)
  def setBinNumber(self,size):
    n = np.ceil(pow(size[0]*size[1]*size[2],1/3.))
    self.nbins = n

#The ImNormalized class inherits from the IM class (we just have to change the behaviour of a function).
#It is better to have two different classes for IM, and IMNormalized... It is more simple for the creation of the objects.
class IMNormalized(IM):
  def __init__(self):
    IM.__init__(self)
  def compute(self,refval,inval):
    return -mutual_information_2d(refval,inval, sigma=self.sigma, normalized=True, nbins=self.nbins)
  


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
  


def createSimplex(x,dx):
  simplex = np.tile(x, (len(x)+1,1))
  for i in range(len(x)):            
    simplex[i+1][i] += dx[i]
  return simplex

def f(x,datareg,rot):

  if x.shape[0] == 6:
    realParameter = (x- datareg.Kcte)/datareg.K # x is the parameter of the transformation multiplied by datareg.K
  else:
    realParameter = np.zeros(6) # x is the parameter of the transformation multiplied by datareg.K
    realParameter[0:3] = np.copy(x)
    realParameter[3:6] = np.copy(rot)
    realParameter = (realParameter - datareg.Kcte)/datareg.K # x is the parameter of the transformation multiplied by datareg.K
    
    
  Mref2in = computeMref2in(realParameter,datareg)
  
  #Apply current transform on input image
  
  warpedarray = affine_transform(datareg.inputarray, Mref2in[0:3,0:3], offset=Mref2in[0:3,3], output_shape=datareg.refarray.shape,  order=1, mode='constant', cval=np.nan, prefilter=False)    
  warpedmask = affine_transform(datareg.inputmask, Mref2in[0:3,0:3], offset=Mref2in[0:3,3], output_shape=datareg.refarray.shape,  order=0, mode='constant', cval=0, prefilter=False)    
  
  #Compute similarity criterion
  index = np.multiply(~np.isnan(warpedarray),warpedmask>0)
  index = np.multiply(datareg.refindex, index)
  
  refval = datareg.refarray[index]
  inval  = warpedarray[index]
  
  res = datareg.criterium.compute(refval,inval)
  
  #Check if overlap between the two images is enough ... otherwise return infinity
  overlap = np.sum(index) * 100.0 / np.sum(datareg.refindex)
  if overlap < 5.0:
    res = 100000 # np.inf to remove warning during execution
  #print '-----------------'
  #print "compute"
  #print realParameter
  #print res
  #print '-----------------'
  return res

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

def do_minimization(container):
  if container.method == "Powell":
    print "Powell"
    xtol = 0.25 / datareg.Kcte #A ctet has been added because xtol is a relative tolerance
    #return minimize(container.f,container.x,container.datareg, method='Powell', options={'xtol': xtol, 'disp': False})    
    return minimize(container.f,container.x,args=(container.datareg,container.rot), method='Powell', options={'xtol': xtol,'ftol' : 1e-50 , 'disp': False})
  else : 
    print "Nelder-Mead"
    simplex = createSimplex(container.x,container.dx)
  return minimize(container.f,container.x,args=(container.datareg,container.rot), method='Nelder-Mead', options={'xatol': 0.25/ datareg.Kcte,  'disp': False, 'initial_simplex' : simplex})    
  


#python pyrecon.py -r ~/WIP_BE_V3_CT.nii.gz -i test.nii.gz -o res.nii.gz -s 8
class container(object):
  pass

def easy_parallize(f, sequence):
  pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())    
  res = pool.map(f, sequence)
  pool.close()
  pool.join()    
  return res
  
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
    self.criterium.setBinNumber(self.refarray.shape)
    
    #normaization of the parameters to optimize
    Ktranslation = 2/min(np.min(s),np.min(s)) 
    Ktheta       = max(np.max(self.refarray.shape),np.max(self.refarray.shape))*3.1415/180
    self.K = np.array([Ktranslation,Ktranslation,Ktranslation,Ktheta,Ktheta,Ktheta]) 
    self.Kcte = 10000


#volumePixel : size of voxels of the two images (this is a vector of size 6)
#largestSize : size of voxel at the lowest resolution (this is a float since at the lowest resolution, voxels are cubic)
#this function returns the size of voxels for both images for each resolution
#Algorithm : the size of the voxel is decresed with a factor 2 (if this is not inferior to the size of the original voxel).
#Example : 
#if the size of voxels are 1,1,4 mm and  2,2,1, then volumePixel = 1,1,4,2,2,1. With largestSize=6
# we will have s[1] = 6 6 6 6 6 6  (a la resolution la plus basse, on aura des images de taille 6x6x6)
# we will have s[2] = 3 3 4 3 3 3 
# we will have s[3] = 1.5 1.5 4 2 2 1.5 
# we will have s[4] = 1 1 4 2 2 1 
#Remarque 2 : at the end, if we are close to the original resolution, we can not consider a resolution  (cf fonction isclose)
#: Example 
#if the sizes are 1,1,14 mm and 0.9,0.9,0.9. With t largestSize = 4
#on aura s[1] = 4 4 14 4 4 4 
#on aura s[2] = 2 2 14 2 2 2 
#on aura s[3] = 1 1 14 0.9 0.9 0.9 (and not 1 1 12 1 1 1 )

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
    if np.isclose(sequence[i,:],volumePixel,rtol=0.2, atol=0.001).all() == True:     
      sequence[i,:] = volumePixel
      take = i+1
      break
                
  sequenceFinal = sequence[0:take,:] 
  return sequenceFinal


#class to constrain the criterium to be part of L1, L2, IM, or IMNormalized lists
class DefaultListAction(argparse.Action):
  CHOICES = ['L1','L2','IM','IMNormalized']
  def __call__(self, parser, namespace, values, option_string=None):
    if values not in self.CHOICES:
      message = ("invalid choice: {0!r} (choose from {1})".format(values,', '.join([repr(action) for action in self.CHOICES])))
      raise argparse.ArgumentError(self, message)
    setattr(namespace, self.dest, values)
          
if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('-i', '--input', help='Input Image to register on the reference image', type=str, required=True)
  parser.add_argument('-r', '--ref', help='Reference Image', type=str, required=True)
  parser.add_argument('--inmask', help='Mask of the input image', type=str)
  parser.add_argument('--refmask', help='Mask of the reference image', type=str)  
  parser.add_argument('-o', '--output', help='Output Image', type=str, required = True)
  parser.add_argument('--padding', help='Padding value used when no mask is provided', type=float, default=-np.inf)
  parser.add_argument('-s', '--scale', help='Scale for multiresolution registration (default: 4 2 1)', nargs = '*', type=float)
  parser.add_argument('--rx', help='Range of rotation x (default: -30 30)', nargs = '*', type=float)
  parser.add_argument('--ry', help='Range of rotation y (default: -30 30)', nargs = '*', type=float)
  parser.add_argument('--rz', help='Range of rotation z (default: -30 30)', nargs = '*', type=float)
  parser.add_argument('--init_using_barycenter', help='Initialization using image barycenters (default: 1 (choose 0 for no init))', type=int, default=1)
  parser.add_argument('--criterium', help='criteritum to minimize', type=str, default="IMNormalized", action=DefaultListAction)
  
  args = parser.parse_args()
  
  criterium = eval(args.criterium)() #instantiate an obect of type L1, L2, IM or IMNormalized
  
  #Loading images    
  inputimage = nibabel.load(args.input)
  refimage   = nibabel.load(args.ref)

  #manage possible 4D shape when reading 3D image using nibabel
  inputimage = nibabel.Nifti1Image(np.squeeze(inputimage.get_data()), inputimage.affine) 
  refimage = nibabel.Nifti1Image(np.squeeze(refimage.get_data()), refimage.affine) 

#TEST to convert image to slices  
#  slices = image2slices(inputimage)
#  inputimage = slices[20]
#  toto = slices2image(slices,refimage)
#  nibabel.save(toto,'toto.nii.gz')  
  
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
    
 
  if args.scale is not None :
    scales = np.array(args.scale)
  else:
    scales = np.array([4,2,1])
    
  
  if scales.shape[0] == 1: 
    volumePixel = np.zeros(6)
    volumePixel[0:3]=np.asarray(inputimage.header.get_zooms())
    volumePixel[3:6]=np.asarray(refimage.header.get_zooms())      
    if scales[0] == 0:
      scales[0] = np.min(volumePixel)*8
    scales = sequenceofundersampling(volumePixel,scales[0])
  else:    
    n = scales.shape[0]
    scales = np.repeat(scales,6)
    scales = np.reshape(scales,(n,6))
    volumePixel = np.zeros((1,6))
    volumePixel[0,0:3]=np.asarray(inputimage.header.get_zooms())
    volumePixel[0,3:6]=np.asarray(refimage.header.get_zooms())  
    scales = np.concatenate((scales,volumePixel),axis=0)
  print scales
  
    
  
  
  if args.rx is not None :
    rx = args.rx
  else:
    rx = [-30,30]
    
  if args.ry is not None :
    ry = args.ry
  else:
    ry = [-30,30]

  if args.rz is not None :
    rz = args.rz
  else:
    rz = [-30,30]

  #3D/3D, 2D/3D, 2D/n2D
  
  #x = [tx, ty, tz, rx, ry, rz]
  #translations are expressed in mm and rotations in degree
  currentx = np.zeros((6))
  
  
  
  if args.init_using_barycenter == 1:
    #Center of rotation : center of reference image expressed in world coordinate
    refcom     = np.asarray(center_of_mass(np.multiply(refimage.get_data(),refmask.get_data())))
    refcom     = np.concatenate((refcom,np.array([1])))
    refcomw    = np.dot(refimage.affine,refcom) #ref center of mass expressed in world coordinate
                       
    #Initialization of translation using center of mass    
    inputcom   = np.asarray(center_of_mass(np.multiply(inputimage.get_data(),inputmask.get_data())))    
    inputcom   = np.concatenate((inputcom,np.array([1])))    
    inputcomw  = np.dot(inputimage.affine,inputcom) #input center of mass expressed in world coordinate        
                       
    currentx[0:3] = (inputcomw-refcomw)[0:3]
    print('initialization for translation : ')
    print(currentx[0:3])
    crefimw = refcomw
    crefimw[3] = 1 
    
  else:
    #Center of rotation : center of reference image expressed in world coordinate
    crefim = (np.asarray(refimage.get_data().shape[0:3])) / 2.0
    crefim = np.concatenate((crefim,np.array([1]))) 
    crefimw= np.dot(refimage.affine,crefim)
    crefimw[3] = 1 
  
###############MULTI-STARTT for resolution 0 ##########"""
  nbre = 4
  eulerX = np.linspace(rx[0],rx[1],num = nbre, endpoint = True)
  eulerY = np.linspace(ry[0],ry[1],num = nbre, endpoint = True)
  eulerZ = np.linspace(rz[0],rz[1],num = nbre, endpoint = True)  
  taken=[]
  for i in range(nbre):
    for j in range(nbre):
      for k in range(nbre):
        x = np.copy(currentx)
        x[3] = eulerX[i]
        x[4] = eulerY[j]
        x[5] = eulerZ[k]        
        taken.append(x)
        
  if nbre>1:
    sizeSimplexX = (rx[1]-rx[0]) * 1.0 / (nbre-1)
    sizeSimplexY = (ry[1]-ry[0]) * 1.0 / (nbre-1)
    sizeSimplexZ = (rz[1]-rz[0]) * 1.0 / (nbre-1)
    sizeSimplex = max([sizeSimplexX,sizeSimplexY,sizeSimplexZ]) / 2 
  else:
    sizeSimplex = 1
  
##OPTIMISATION DES TRANSLATIONS A L ECHELLE 0

  s = scales[0]
  datareg = dataReg(refimage,inputimage,refmask,inputmask,crefimw,s,criterium,True)
  listofcontainers = []
  for i in range(len(taken)):
    c = container()
    c.f = f
    c.x = np.copy(taken[i])
    c.x = c.x * datareg.K + datareg.Kcte #for optimization so that parameters are comparable --> the parameters are made de-normalized in the f-function              
    c.datareg = datareg
    c.method = 'Powell'
    c.rot = np.copy(c.x[3:6])
    c.x = np.copy(c.x[0:3])
    listofcontainers.append(c)    
  res = easy_parallize(do_minimization, listofcontainers)

  listofcontainers = []
  for i in range(len(taken)):
    c = container()
    c.f = f
    c.x = np.copy(taken[i])
    c.x = c.x * datareg.K + datareg.Kcte #for optimization so that parameters are comparable --> the parameters are made de-normalized in the f-function          
    c.dx = np.ones((3))*6 # 1/2 is the resolution    
    c.datareg = datareg
    c.method = 'Nelder-Mead'
    c.rot = np.copy(c.x[3:6])
    c.x = np.copy(c.x[0:3])
    listofcontainers.append(c)    
  res2 = easy_parallize(do_minimization, listofcontainers)



  takenFinal = []
  for i in range(len(taken)):    
    tak1 = np.copy(taken[i])    
    tak2 = np.copy(taken[i]* datareg.K + datareg.Kcte)
    if res[i].fun<res2[i].fun:
      print 'Powelle prefered'
      tak2[0:3] = res[i].x
    else:
      print 'Nelder-Mead prefered'
      tak2[0:3] = res2[i].x            
    tak2 = (tak2- datareg.Kcte) / datareg.K #to obtain the real paramete 
    takenFinal.append(tak1)
    takenFinal.append(tak2)
  taken = takenFinal
    


###AUTRES ITERATIONS..................;
  for iteration in range(scales.shape[0]):
    s = scales[iteration]        
    if iteration == scales.shape[0]-1:
      datareg = dataReg(refimage,inputimage,refmask,inputmask,crefimw,s,criterium,False)
    else:
      datareg = dataReg(refimage,inputimage,refmask,inputmask,crefimw,s,criterium,True)
        
    #datareg = dataReg(refimage,inputimage,refmask,inputmask,crefimw,np.asarray([s,s,inputimage.header.get_zooms()[2]])) # resampling is anisotropic for slices
                      
    
    listofcontainers = []
    for i in range(len(taken)):
      c = container()
      c.f = f
      c.rot = 0
      c.x = np.copy(taken[i])
      c.x = c.x * datareg.K + datareg.Kcte #for optimization so that parameters are comparable --> the parameters are made de-normalized in the f-function      
      c.datareg = datareg
      c.method = 'Powell'
      c2 = container()
      c2.rot = 0
      c2.f = f
      c2.x = np.copy(c.x)
      c2.datareg = datareg
      c2.method = 'Nelder-Mead'        
      if iteration == 0:
        c2.dx = sizeSimplex * datareg.K   
      else:
        c2.dx = np.ones((6))
      listofcontainers.append(c2)
      listofcontainers.append(c)
      
        

    res = easy_parallize(do_minimization, listofcontainers)
    
    sortedx = sorted(res, key=lambda(r) : r.fun) #Use sorted array to extract possibly multiple best candidates
    best = sortedx[0]    
    print best
    currentx = np.copy(best.x)
    
    taken = []
    fun = []
    taken.append(np.asarray(currentx))
    fun.append(best.fun)
    NUMBER = nbre * nbre *nbre / 4
    number = 1
    for i in range(len(sortedx)):      
      take = 1
      sort = np.copy(np.asarray(sortedx[i].x))
      for j in range(len(taken)):      
        difference = np.abs(taken[j] - sort) 
        if np.max(difference)<1:        
          take = 0        
      if take == 1:        
        taken.append(sort)
        fun.append(sortedx[i].fun)
        number = number+1
      if number>NUMBER:
        break

    #print "AT RESOLUTION " 
    #print s
    #print "Nombre de minima locaux" 
    #print len(taken)
  
    #for i in range(len(taken)):
    #  print taken[i]
   


    print "AT RESOLUTION " 
    print s
    print "Nombre de minima locaux " 
    print len(taken)
  
    for i in range(len(taken)):
      taken[i] = (taken[i]- datareg.Kcte) / datareg.K #to obtain the real parameter
      print str(taken[i]) + " with " + str(fun[i])
   
    best = sortedx[0]    
    currentx = np.copy(best.x)
    currentx = (currentx- datareg.Kcte) / datareg.K #to obtain the real parameter    
    Mref2in = computeMref2in(currentx,datareg)
    warpedarray = affine_transform(np.reshape(datareg.inputimage.get_data(),datareg.inputimage.get_data().shape[0:3]), Mref2in[0:3,0:3], offset=Mref2in[0:3,3], output_shape=datareg.refimage.get_data().shape[0:3],  order=3, mode='constant', cval=0, prefilter=False)     
    nibabel.save(nibabel.Nifti1Image(warpedarray, datareg.refimage.affine),'current_at_scale_'+str(s)+'.nii.gz')
#
#  #Final interpolation using the reference image spacing
  volumePixel = np.zeros(6)
  volumePixel[0:3]=np.asarray(inputimage.header.get_zooms())
  volumePixel[3:6]=np.asarray(refimage.header.get_zooms())
  datareg = dataReg(refimage,inputimage,refmask,inputmask,crefimw,volumePixel,criterium,False)    
  Mref2in = computeMref2in(currentx,datareg)
  warpedarray = affine_transform(np.reshape(inputimage.get_data(),inputimage.get_data().shape[0:3]), Mref2in[0:3,0:3], offset=Mref2in[0:3,3], output_shape=refimage.get_data().shape[0:3],  order=3, mode='constant', cval=0, prefilter=False)     
  nibabel.save(nibabel.Nifti1Image(warpedarray, refimage.affine),args.output)

  #TODO ? blur the data before downsampling
  #TODO ? add linear transform for parameters
  #TODO : define a fast and simple strategy for multiresolution registration using random inits.
  #TODO : initialization for slice to volume maybe different from 3D/3D case
  #TODO : manage slice image resampling
