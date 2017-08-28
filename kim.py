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
from pyrecon import compute_affine_matrix,homogeneousMatrix, createSimplex
import sys
from scipy.interpolate import griddata
#From  : https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429
from scipy import ndimage
EPS = np.finfo(float).eps


              
def ComputeValueOfA2DimageOnALine(input_image,mask,pointDepart,pointArrive, nbrePoints):
  """
  Compute the value of a 2D image alongs a line
  input : 
  input_image : 2D image
  mask        : 2D image
  pointDepart : 2D point (start of the line)
  pointArrive : 2D point (end of the line)
  nbrePoint   : number of the points of the line
  return : 
  val         : value of input_image along the line
  index       : index where the val is of interest  
  """
  
  val = np.zeros(nbrePoints)
  maskl = np.zeros(nbrePoints)
  pointset = np.zeros((2,nbrePoints))
  pointset[0,:] = np.linspace(pointDepart[0] , pointArrive[0] , nbrePoints)
  pointset[1,:] = np.linspace(pointDepart[1] , pointArrive[1] , nbrePoints)
  map_coordinates(input_image.get_data(),[pointset[0,:],pointset[1,:]],output=val ,mode='constant',cval = np.nan     , order=1,prefilter=False)
  map_coordinates(mask.get_data()       ,[pointset[0,:],pointset[1,:]],output=maskl,mode='constant',cval = np.nan, order=0,prefilter=False)
  index = np.multiply(~np.isnan(val),maskl>0)  
  return val,index


      
def ComputeL2criterium(M1,M2,I1,I2,mask1,mask2):
  """
  Compute the L2 criterium  along a line defined by the intersection of the 2-D image I1 transformed by M1, and I2 by M2.  
  input : 
  M1 and M2 :transformation in the world coordinate in homogeneous coordinate
  I1 and I2 :2D images that are considered to be in the 0xy plane
  mask1 and mask2 :2D masks
  return :
  res : somme au carra
  len(v1) : number of points on the line
  """
  Point1,Point2,nbrePoints,OK = CommonCoordinateBetweenTwoImages(M1,M2,I1.get_data().shape,I2.get_data().shape)  

  if OK < 1:
    return 0.,0
  else:
    val1,index1 = ComputeValueOfA2DimageOnALine(I1,mask1,Point1[:,0],Point1[:,1],nbrePoints)
    val2,index2 = ComputeValueOfA2DimageOnALine(I2,mask2,Point2[:,0],Point2[:,1],nbrePoints)
    index = np.multiply(index1,index2)    
    v1 = val1[index]
    v2 = val2[index]
    if v1.shape[0]==0:
      return 0.,0
    
    diff = v2-v1        
    res = np.sum(diff**2)
    
  return res,len(v1)
              
def CommonCoordinateBetweenTwoImages(M1,M2,tailleImage1,tailleImage2):
  """
  compute two common segments from two planes
  input : 
  M1 : 3-D transformation that defines the first plane (transformation of 0xy by M1)
  M2 : 3-D transformation that defines the second plane (transformation of 0xy by M1)
  tailleImage1 : size of the first plane in the 0xy plane
  tailleImage2 : size of the first plane in the 0xy plane
  
  return : 
  Point1 and Point2 : A segment is defined by two points   
  Point1[:,0] is the first point of the segment for the first image in the 0xy plane
  Point1[:,1] is the second point of the segment for the first image in the 0xy plane
  Point2[:,0] is the first point of the segment for the second image in the 0xy plane
  Point2[:,1] is the second point of the segment for the second image in the 0xy planee
  nbrePoints : defines he length of the segment (the number of points that can b computed)
  the last value is 1 if two points have been found, 0 else.
  """
  
  point,tangent,OK = IntersectionLineOfTwoPlanes(M1,M2)
  if OK<1:
    return 0,0,0,0
  lambda1,OK = SegmentFromTheLine(M1,point,tangent,tailleImage1)
  if OK < 1:
    return 0,0,0,0
    
  lambda2,OK = SegmentFromTheLine(M2,point,tangent,tailleImage2)
  if OK < 1:
    return 0,0,0,0
  
  #computation of the intersection
  lambda1 = np.sort(lambda1) # + petit au + grand
  lambda2 = np.sort(lambda2) # + petit au + grand
  intersection = np.zeros((2,))
  intersection[0] = max(lambda1[0],lambda2[0]) 
  intersection[1] = min(lambda1[1],lambda2[1]) 
  if intersection[0] == intersection[1]:
    return 0,0,0,0

  segment3D = np.zeros((4,2))    
  segment3D[3,:]=1 #homogenesous coordinates
  segment3D[0:3,0] = intersection[0] * tangent + point
  segment3D[0:3,1] = intersection[1] * tangent + point
  Point1  = np.dot(np.linalg.inv(M1),segment3D) #in the 0xy plane
  Point2 = np.dot(np.linalg.inv(M2),segment3D) #in the 0xy plane

  

  distance1 = np.linalg.norm(Point1[0:2,1] - Point1[0:2,0])  
  distance2 = np.linalg.norm(Point2[0:2,1] - Point2[0:2,0])  
  if max(distance1,distance2)<1: #La ligne en commun comporte un pixel !!
    return 0,0,0,0
  
  nbrePoints = int(np.round(max(distance1,distance2))+1)
  return Point1,Point2,nbrePoints,1

#  checking that points are coresponding
#  pointSet1 = np.zeros((4,nbrePoints))
#  pointSet1[3,:] = np.ones((nbrePoints))
#  pointSet1[2,:] = np.zeros((nbrePoints))
#  pointSet1[0,:] = np.linspace(Point1[0,1] , Point1[0,1] , nbrePoints)
#  pointSet1[1,:] = np.linspace(Point1[1,1] , Point1[1,1] , nbrePoints)
#  
#  pointSet2 = np.zeros((4,nbrePoints))
#  pointSet2[3,:] = np.ones((nbrePoints))
#  pointSet1[2,:] = np.zeros((nbrePoints))
#  pointSet2[0,:] = np.linspace(Point2[0,1] , Point2[0,1] , nbrePoints)
#  pointSet2[1,:] = np.linspace(Point2[1,1] , Point2[1,1] , nbrePoints)
#  for i in range(nbrePoints):    
#    toto1 = pointSet1[:,i]
#    toto2 = pointSet2[:,i]
#    difference = np.dot(M1,toto1) - np.dot(M2,toto2)
#    if sum(abs(difference))>1e-6:
#      print 'bug'
#      

  
  
def IntersectionLineOfTwoPlanes(M1,M2):
  """
  compute the intersection of two planes defined by the transformation of the Oxy plane by M1 and M2 (4x4 matrices defining
  an affine transformation)
  input : 
  M1 : 3-D transformation that defines the first plane (transformation of 0xy by M1)
  M2 : 3-D transformation that defines the second plane (transformation of 0xy by M1)
  result : 
  the equation of the 3-D line is given as : point + lambda tangente (lambda quelconque). (point belongs to the line)
  the last value is 1 if everithing is OK, 0 if there is no interesction
  """
  
  #we start by computing the normal vector of 0xy transformed by M1 and M2
  
  
  n1 = np.cross(M1[0:3,0], M1[0:3,1])
  norm = np.linalg.norm(n1)
  if norm<1e-6:
    return 0,0,0
  n1 = n1 / norm
    
  
  n2 = np.cross(M2[0:3,0], M2[0:3,1])  
  norm = np.linalg.norm(n2)
  if norm<1e-6:
    return 0,0,0
  n2 = n2 / norm
  
  
  
  #compute the intersection
  alpha = np.dot(n1,n2)
  beta  = np.dot(n1,M1[0:3,3])
  gamma = np.dot(n2,M2[0:3,3])               
  if abs(1-alpha*alpha)<1e-6:
    return 0,0,0
  g = 1/(1-alpha*alpha)*(beta-alpha*gamma)
  h = 1/(1-alpha*alpha)*(gamma-alpha*beta)
  point = g*n1+h*n2
  tangent = np.cross(n1,n2)
  return point,tangent,1
  #the line has equation lambda tangent + point
  
def SegmentFromTheLine(M,point,tangent,sizeImage):
  """
  
  input : 
    M1      : 3-D transformation that defines a plane (transformation of 0xy by M)
    point   : point that is on the line
    tangent : vecteur directeur de la droite
    sizeImage : taille de l image dans le plan 0xy
  result : 
    the result is given in the forme of two points defining the segment in the word coordinate 
    so that the the image of the segment by M is in 0xy plane inside the support of the image (sizeImage)
    since the equation of the line is point + lambda tangente (lambda quelconque), we give two values of lambda
  """
  
  inverse = np.linalg.inv(M[0:3,0:3])
  tangent2D = np.dot(inverse,tangent)
  normal = np.zeros((2,))
  normal = [-tangent2D[1],tangent2D[0]]
  point2D = np.dot(inverse, point - M[0:3,3])
  a = normal[0];
  b = normal[1];
  c = -np.sum(normal*point2D[0:2])

  #the equation of the line is ax+by+c=0 in the 0xy plane
  
  #Computation of the two intersections of the line with the support of the image
  intersection = np.zeros((4,2))
  intersection[3,:]=1 #homogenesous coordinates
  nbreIntersection = 0
  if abs(b)>1e-10:
    #intersection with x=0
    y = -c/b;
    if y >= 0 and  y <= sizeImage[1]-1:
      intersection[0:2,nbreIntersection] = [0,y]
      nbreIntersection = nbreIntersection+1

    #intersection with x=sizeImage[0]-1
    y = -(a*(sizeImage[0]-1)+c)/b;
    if y >= 0 and  y <= sizeImage[1]-1:
      intersection[0:2,nbreIntersection] = [sizeImage[0]-1,y]
      nbreIntersection = nbreIntersection+1

  if abs(a)>1e-10:
    #intersection with y=0
    x = -c/a;

    if x >= 0 and  x <= sizeImage[0]-1:
      intersection[0:2,nbreIntersection] = [x,0]
      nbreIntersection = nbreIntersection+1

    #intersection with y=sizeImage[1]-1
    x = -(b*(sizeImage[1]-1)+c)/a;
    if x >= 0 and  x <= sizeImage[0]-1:
      intersection[0:2,nbreIntersection]  = [x,sizeImage[1]-1]
      nbreIntersection = nbreIntersection+1
  if nbreIntersection <2:
    return 0,0
  
  
  #computation of lambda associated with the two intersections transformed by M.
  transformedPoints = np.dot(M,intersection) 
  proptoTangent = transformedPoints[0:3,:]
  proptoTangent[:,0] = transformedPoints[0:3,0] - point
  proptoTangent[:,1] = transformedPoints[0:3,1] - point
  
  #proptoTangent is proportional to tangent
  #computation of the coefficient of proportionality
  tmp = np.dot(tangent,tangent.transpose())
  tmp = np.dot(1./tmp,tangent.transpose())
  coeff = np.dot(tmp,proptoTangent)
  return  coeff,1
                       
  

  

#Object that is used to put all the stuff needed for registration...
#all values of these classes must not be changed because all are linked together : use set_parameter to change parameter
# if needed, other set functions can be defined
class dataReg:
  def __init__(self,sliceimage,slicemask,centerofRotation,orientation):
    self.sliceimage   = sliceimage
    self.slicemask    = slicemask
    self.Mref2w       = self.sliceimage.affine #reference to world
    self.orientation = orientation # to know if the slices must be considered for registration
    self.centerrot  = centerofRotation
    self.Mw2c       = np.eye(4)
    self.Mc2w       = np.eye(4)  
    self.Mw2c[0:3,3]= -self.centerrot[0:3]
    self.Mc2w[0:3,3]=  self.centerrot[0:3]
    self.parameter = np.zeros((6)) # parameter of interest
    self.Mestimated = compute_affine_matrix(translation=self.parameter[0:3], angles=self.parameter[3:6]) # deformation in the word coordinate : the one of interest
    
    self.total = np.dot(self.Mc2w, np.dot( self.Mestimated,np.dot(self.Mw2c,self.Mref2w) ) )
    
  def set_paramater(self,x): #changing Mestimated leads to change self.total    
    self.parameter = x.copy() 
    self.Mestimated = compute_affine_matrix(translation=x[0:3], angles=x[3:6]) #estimated transform
    self.total = np.dot(self.Mc2w, np.dot( self.Mestimated,np.dot(self.Mw2c,self.Mref2w) ) )
  
  parameter = property(set_paramater) #so that written myObject.Mestimated = calls the set_Mestimated function
  
  def giveMref2in(self):
    return self.total
  
  def computeMref2in(self,x):
    #Compute transform from refimage to inputimage      
    M = compute_affine_matrix(translation=x[0:3], angles=x[3:6]) #estimated transform
    return np.dot(self.Mc2w, np.dot( M,np.dot(self.Mw2c,self.Mref2w) ) )

    
def loadimages(fileImage,fileMask,padding):
  im = nibabel.load(fileImage)
  im = nibabel.Nifti1Image(np.squeeze(im.get_data()), im.affine) 
  if fileMask is not None :
    imMask = nibabel.load(fileMask)
    imMask = nibabel.Nifti1Image(np.squeeze(imMask.get_data()), imMask.affine) 
  else:
    print('Creating mask images using the following padding value:',padding)
    imdata = np.zeros(im.get_data().shape)
    imdata[im.get_data() > args.padding] = 1
    imdata = np.reshape(imdata,imdata.shape[0:3])
    imMask = nibabel.Nifti1Image(imdata, im.affine) 
  return im,imMask



def computeCenterOfRotation(sliceimage,slicemask):
  refcom     = np.asarray(center_of_mass(np.multiply(sliceimage.get_data(),slicemask.get_data())))
  if np.any(np.isnan(refcom)) == True: # barycenter cannot be computed : the image is null or the mask is 0 everywhere 
    return 0,0    
  refcom     = np.concatenate((refcom,np.array([0,1])))
  refcomw    = np.dot(sliceimage.affine,refcom)
  return refcomw,1

#In the following, we define three functions to create slices from an image.
#At the end, we have a 2-D image, but we have a 3-D  transform (slicetransform).
#we suppose that z=0 for the 2D image so that a line of the transformation cannot be computed.
#In order to make the slice transform invertible, we put M[2,2] equal to 1 for image2slicesInZ
#M[0,2] to 1 for image2slicesInX, M[1,2] to 1 for image2slicesInY (we have a rotation in each case)
def image2slicesInX(image,mask,slices,typeSlice):  
  for x in range(image.get_data().shape[0]):
    sliceimage = image.get_data()[x,:,:]
    slicemask  = mask.get_data()[x,:,:]
    M = np.zeros((4,4))
    M[0,3] = x
    M[0,2] = 1 # to obtain a rotation
    M[1,0] = 1
    M[2,1] = 1
    M[3,3] = 1 
    sliceaffine = np.dot(image.affine,M)
    sliceimage = nibabel.Nifti1Image(sliceimage,sliceaffine)
    slicemask = nibabel.Nifti1Image(slicemask,sliceaffine)    
    centerofRotation,OK = computeCenterOfRotation(sliceimage,slicemask)
    if OK>0:
      datareg = dataReg(sliceimage,slicemask,centerofRotation,typeSlice)
      slices.append(datareg)
    
  return slices

def image2slicesInY(image,mask,slices,typeSlice):  
  for y in range(image.get_data().shape[1]):
    sliceimage = image.get_data()[:,y,:]
    slicemask  = mask.get_data()[:,y,:]
    M = np.zeros((4,4))
    M[1,3] = y
    M[1,2] = 1 # to obtain a rotation
    M[0,0] = 1
    M[2,1] = 1
    M[3,3] = 1 
    sliceaffine = np.dot(image.affine,M)
    sliceimage = nibabel.Nifti1Image(sliceimage,sliceaffine)
    slicemask = nibabel.Nifti1Image(slicemask,sliceaffine)
    centerofRotation,OK = computeCenterOfRotation(sliceimage,slicemask)
    if OK>0:
      datareg = dataReg(sliceimage,slicemask,centerofRotation,typeSlice)
      slices.append(datareg)
  return slices

def image2slicesInZ(image,mask,slices,typeSlice):  
  for z in range(image.get_data().shape[2]):
    sliceimage = image.get_data()[:,:,z]
    slicemask  = mask.get_data()[:,:,z]
    M = np.eye(4)
    M[2,3] = z
    sliceaffine = np.dot(image.affine,M)
    sliceimage = nibabel.Nifti1Image(sliceimage,sliceaffine)
    slicemask = nibabel.Nifti1Image(slicemask,sliceaffine)    
    centerofRotation,OK = computeCenterOfRotation(sliceimage,slicemask)
    if OK>0:
      datareg = dataReg(sliceimage,slicemask,centerofRotation,typeSlice)    
      slices.append(datareg)
  return slices

def createSlices(image,mask,slices,typeSlice):  
  zoom = image.header.get_zooms()
  if zoom[0]>zoom[1] and zoom[0]>zoom[2]:
    slices =  image2slicesInX(image,mask,slices,typeSlice)
  elif zoom[1]>zoom[0] and zoom[1]>zoom[2]:    
    slices =  image2slicesInY(image,mask,slices,typeSlice)
  elif zoom[2]>zoom[0] and zoom[2]>zoom[1]:
    slices =  image2slicesInZ(image,mask,slices,typeSlice)
  else : 
    print "problem of size in the images"
    #slices =  image2slicesInX(image,mask,slices)
    

  return slices
      

def criteriumTotal(slices):
  #slices : containter containing all required information
  #compute the total square error and the common number of pixels 
  valueTotal = 0
  nbreTotal = 0
  for i in range(len(slices)):
    value,n = f2(slices[i].parameter,slices,i)
    valueTotal = valueTotal + value
    nbreTotal = n + nbreTotal
  return valueTotal/2,nbreTotal/2
        
  
#criterium for the n-th slice
def f(x,slices,n,valueTotalFaux,nbreTotalFaux): 
  # x : input parameters
  # slices : container containing all required information for minimization
  # n : number of the slice to optimize
  # valueTotalFaux and nbreTotalFaux : criterium withtout considering the n-th slice
  valueFinal,nbreTotal = f2(x,slices,n)
  return (valueTotalFaux+valueFinal)/(nbreTotal+nbreTotalFaux)



def f2(x,slices,n):
  valueFinal = 0
  nbreTotal = 0
  Mn = slices[n].computeMref2in(x)
  for i in range(len(slices)):
    if slices[i].orientation != slices[n].orientation:
      Mi = slices[i].giveMref2in()
      r,s=ComputeL2criterium(Mi,Mn,slices[i].sliceimage, slices[n].sliceimage,slices[i].slicemask,slices[n].slicemask)
      valueFinal = valueFinal+r
      nbreTotal = nbreTotal + s
  return valueFinal,nbreTotal


#for a small algorithm of reconstruction
def applyDirectTransform(input_image, transform):
  
  #create index for the reference space
  i = np.arange(0,input_image.get_data().shape [0])
  j = np.arange(0,input_image.get_data().shape [1])
 
    
  iv,jv = np.meshgrid(i,j,indexing='ij')
  
  iv = np.reshape(iv,(-1))
  jv = np.reshape(jv,(-1))
  
  
  #compute the coordinates in the input image
  pointset = np.zeros((4,iv.shape[0]),dtype=np.int32)
  pointset[0,:] = iv
  pointset[1,:] = jv
  pointset[2,:] = np.zeros((iv.shape[0]))
  pointset[3,:] = np.ones((iv.shape[0]))
  
  imagedata = input_image.get_data()[pointset[0,:],pointset[1,:]]
  
  #compute the transformation
  pointset = np.zeros((4,iv.shape[0]))
  pointset[0,:] = iv
  pointset[1,:] = jv
  pointset[2,:] = np.zeros((iv.shape[0]))
  pointset[3,:] = np.ones((iv.shape[0]))
  
  pointset = np.dot(transform,pointset) 
  
  #compute the interpolation
  return pointset,imagedata
    

def reconstruction(slices,taillePixel):
  
  #calcul d une image dans l espace du monde avec une grille non discrete
  #pointset est les coordonnees
  #imagedata les donnees images
  OK = 0
  for i in range(len(slices)):
    nbreTotal,n = f2(slices[i].parameter,slices,i) #on regarde si la slice s interescte avec les autres
    if n>0: #OK
      pointset2,imagedata2 = applyDirectTransform(slices[i].sliceimage, slices[i].total)
      pointset3,imagedata3 = applyDirectTransform(slices[i].slicemask, slices[i].total)
      garde = np.where(imagedata3>0)
      imagedata2 = imagedata2[garde[0]]
      pointset2 = pointset2[:,garde[0]]
      
      
      if OK == 0:
        pointset = pointset2
        imagedata = imagedata2
        OK = 1
      else:
        pointset = np.append(pointset,pointset2,1)
        imagedata = np.append(imagedata,imagedata2)
    else:
      print "removing for reconstruction"
      print i
    
  #calcul de l image sur une grille discrete
  #creation d une image sur une grille discrete : calcul de la boite englobante
  minX = np.min(pointset[0,:])
  maxX = np.max(pointset[0,:])
  minY = np.min(pointset[1,:])
  maxY = np.max(pointset[1,:])
  minZ = np.min(pointset[2,:])
  maxZ = np.max(pointset[2,:])
  
  
  #calcul de la resolution, de la taille de l image, de affine
  resolution = np.zeros((4))
  resolution[0] = taillePixel  #min(inputimage1.affine[0,0],inputimage2.affine[0,0])
  resolution[1] = taillePixel #min(inputimage1.affine[1,1],inputimage2.affine[1,1])
  resolution[2] = taillePixel #min(inputimage1.affine[2,2],inputimage2.affine[2,2])
  resolution[3] = 1
  
  affine = np.diag(resolution)
  affine[0][3] = minX
  affine[1][3] = minY
  affine[2][3] = minZ
          
  nbreX = np.ceil((maxX-minX)/resolution[0]) + 1
  nbreY = np.ceil((maxY-minY)/resolution[1]) + 1
  nbreZ = np.ceil((maxZ-minZ)/resolution[2]) + 1
  nbreX = np.int32(nbreX)
  nbreY = np.int32(nbreY)
  nbreZ = np.int32(nbreZ)
  print nbreX
  print nbreY
  print nbreZ
  #calcul des coordonnees dans l espace monde
  xi = np.linspace(minX,maxX,nbreX)
  yi = np.linspace(minY,maxY,nbreY)
  zi = np.linspace(minZ,maxZ,nbreZ)
  iv,jv,kv = np.meshgrid(xi,yi,zi,indexing='ij')
  iv = np.reshape(iv,(-1))
  jv = np.reshape(jv,(-1))
  kv = np.reshape(kv,(-1))

  pointsetInter = np.zeros((iv.shape[0],3))
  pointsetInter[:,0] = iv
  pointsetInter[:,1] = jv
  pointsetInter[:,2] = kv
  
  pointset = pointset.transpose()
  print "on commence"
  print pointset.shape
  image = griddata(pointset[:,0:3], imagedata, pointsetInter, method='linear')
  print "on finit"
  image = image.reshape((nbreX,nbreY,nbreZ))
  imrecon =  nibabel.Nifti1Image(image,affine)
  return imrecon 




if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  
  parser.add_argument('-i', '--input', help='Input Image', type=str,action='append')
  parser.add_argument('--inmask', help='Mask of the images', type=str,action='append')  
  #parser.add_argument('-o', '--output', help='Output Image', type=str, required = True)
  parser.add_argument('--padding', help='Padding value used when no mask is provided', type=float, default=-np.inf)
  args = parser.parse_args()

  print args.input
  args.input = ['/home/miv/faisan/data/mar0027_exam01_T2_haste_axial_crop.nii.gz', '/home/miv/faisan/data/mar0027_exam01_T2_haste_coronal_crop.nii.gz', '/home/miv/faisan/data/mar0027_exam01_T2_haste_sagittal_crop.nii.gz']
  args.inmask = ['/home/miv/faisan/data/mar0027_exam01_T2_haste_axial_mask_crop.nii.gz', '/home/miv/faisan/data/mar0027_exam01_T2_haste_coronal_mask_crop.nii.gz', '/home/miv/faisan/data/mar0027_exam01_T2_haste_sagittal_mask_crop.nii.gz']
  argspadding = -np.inf
  #Loading image
  slices=[]
  resolutionFinal = np.inf
  for i in range(len(args.input)):
    if args.inmask is not None :
      inputimage,mask   =  loadimages(args.input[i],args.inmask[i],args.padding)      
    else:
      inputimage,mask   =  loadimages(args.input[i],None,args.padding)    
    resolutionMin = np.min(inputimage.header.get_zooms())
    if resolutionMin<resolutionFinal:
      resolutionFinal = resolutionMin
    slices = createSlices(inputimage,mask,slices,i)
  
#  valueTotal,nbreTotal = criteriumTotal(slices)
###  #registration
#  reference = 25
#  for n in range(3):
#    print n
#    for i in range(len(slices)):      
#      print i
#      if i != reference:
#        print("before optimization slice ", n , " : ", valueTotal/nbreTotal)
#        value,nbre = f2(slices[i].parameter,slices,i) 
#        valueTotal = valueTotal - value # criterium without considering the i-th slice
#        nbreTotal = nbreTotal - nbre
#        param = minimize(f,slices[i].parameter,args=(slices,i,valueTotal,nbreTotal), method='Powell', options={ 'disp': True})
#        slices[i].set_paramater(param.x)
#        value,nbre = f2(slices[i].parameter,slices,i) 
#        valueTotal = valueTotal + value 
#        nbreTotal = nbreTotal + nbre
#        print("after optimization slice ", n , " : ", valueTotal/nbreTotal, "with : " ,valueTotal, " and " ,nbreTotal)
#        valueTotal2,nbreTotal2 = criteriumTotal(slices)
#        print("criterium verification ", n , " : ", valueTotal2/nbreTotal2, "with : " ,valueTotal2," and " ,nbreTotal2)
#        
#        
#  
#  #simple reconstruction
  imrecon = reconstruction(slices,resolutionFinal)
  nibabel.save(imrecon,"tmp.nii.gz")

  
  
