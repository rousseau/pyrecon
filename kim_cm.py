#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 13:59:55 2021

@author: mercier
"""


import nibabel as nib 
import numpy as np
import matplotlib.pyplot as plt 
from scipy.ndimage import map_coordinates
from sliceObject import SliceObject
from display import plotsegment
#from display import show_slice,plotsegment



##Chargement des images
img1 = nib.load('../donnee/transfer_2241503_files_0c0883a5/20090325_T2_AXIAL_01.nii.gz')
img2 = nib.load('../donnee/transfer_2241503_files_0c0883a5/20090325_T2_SAG_01.nii.gz')
img3 = nib.load('../donnee/transfer_2241503_files_0c0883a5/20090325_T2_COR_01.nii.gz')
masque1 = nib.load('../donnee/transfer_2241503_files_0c0883a5/20090325_T2_AXIAL_01_mask.nii.gz')
masque2 = nib.load('../donnee/transfer_2241503_files_0c0883a5/20090325_T2_SAG_01_mask.nii.gz')
masque3 = nib.load('../donnee/transfer_2241503_files_0c0883a5/20090325_T2_COR_01_mask.nii.gz')


    

def intersectionLineBtw2Planes(M1,M2) : 
    """
    Compute the intersection line between two planes

    input : 
    M1 : 4x4 matrix
        3-D transformation that defines the first plane
         
    M2 : 4x4 matrix
        3-D transformtation that defines the second place
        

    Returns :
    coeff : 3x1 vector
        vector tangent to the line of intersection
    pt : 3x1 vector
        point on the line of intersection
    ok : integer
        1 if there is an intersection, else 0
    """  
    
    #normal vector to the 0xy plan
    n1 = np.cross(M1[0:3,0],M1[0:3,1]) 
    if np.linalg.norm(n1)<1e-6: #no division by 0, case M11 // M12 (normally that souldn't be the case)
        return 0,0,0
    n1norm = n1/np.linalg.norm(n1)
    n1 = n1norm
    t1 = M1[0:3,3]
   
    n2 = np.cross(M2[0:3,0],M2[0:3,1]) 
    if np.linalg.norm(n2)<1e-6: #no division by 0, case M21 // M22 (normally that souldn't be the case)
        return 0,0,0
    n2norm = n2/np.linalg.norm(n2)
    n2 = n2norm
    t2 = M2[0:3,3]
    
    
    alpha = n1 @ n2 #if the vector are colinear alpha will be equal to one (since n1 and n2 are normalized), can happend if we consider two parralel slice
    beta =  n1 @ t1
    gamma = n2 @ t2
    
    if abs((1 - alpha*alpha))<1e-6: #if the vector are colinear, there is no intersection
        return 0,0,0
    a = 1/(1 - alpha*alpha)
    g = a*(beta - alpha*gamma)
    h = a*(gamma - alpha*beta)

   
    #line equation
    coeff = np.cross(n1,n2)
    pt = g*n1 + h*n2
   
    return coeff, pt, 1
 
  
def intersectionSegment(sliceimage,coeff,pt):
    """
    Compute the segment of intersection between the line and the 2D slice
    
    input :
    sliceimage : Slice
        contains all the necessary information on the slice, including the 3D matrix transformation and the data from the slice)
    coeff : 3x1 vector
        vector tangent to the line of intersection
    pt :  3x1 vector
        point on the line 
    ok : integer
        1 if there is an intersection, else 0

    Output :
    lambdaPropo : 2 values of lambda which defines the intersection points on the line


    """
    
    #line equation into the image plan
    M = sliceimage.get_transfo()
    
    Minv = np.linalg.inv(M)
    rinv = np.linalg.inv(M[0:3,0:3])
    n = rinv @ coeff
    pt = np.concatenate((pt,np.array([1])))
    ptimg =Minv @ pt
   
    a = -n[1]
    b = n[0]
    c = -(a*ptimg[0] + b*ptimg[1])
    
    
    #Intersection with the plan
    intersection = np.zeros((4,2)) #2 points of intersection of coordinates i,j,k 
    intersection[3,:] = np.ones((1,2))
    width = sliceimage.get_slice().shape[0]-1
    height = sliceimage.get_slice().shape[1]-1
 

    indice=0
    #The intersection on a corner are considered only once
    
    if (abs(a)>1e-10): #if a==0, the division by zeros in not possible, in this case we have only two intersection possible : 
           
        i=(-c/a); j=0;
        
        if  i >= 0 and i < width: #if y=0 x=-c/a  #the point (0,0) is considered here
            intersection[0,indice] =  i
            intersection[1,indice] =  j
            indice=indice+1
        
        i=((-c-b*height)/a); j=height
       
        if (i>0) and (i <= width) : #if y=height x=-(c-b* height)/a #the point  (width,height) is considered here
            intersection[0,indice] = i
            intersection[1,indice] = j
            indice=indice+1
        
         
    if (abs(b)>1e-10): #if b==0, the divistion by zeros in not possible, in this case we have only two intersection possible :
           
        i=0; j=(-c/b);
        
        if j>0 and  j <= height: #if x=0 y=-c/b #the point (0,heigth) is considered here
            intersection[0,indice] = i 
            intersection[1,indice] = j
            indice=indice+1
       
        i=width; j=(-c-a*width)/b
        
        if j>=0  and j<height: #if x=width y=(-c-a*width)/b  #the point (width,0) is considered here
            intersection[0,indice] = i
            intersection[1,indice] = j
            indice=indice+1

    
    if indice < 2 or indice > 2:
        return 0,0
        
    #Compute the intersection point coordinates in the 3D space
    interw = np.zeros((4,2)) #2 points of intersection, with 3 coordinates x,y,z
    interw[3,:] = np.ones((1,2))
    interw = M @ intersection 
    
    interw[0:3,0] = interw[0:3,0] - pt[0:3]
    interw[0:3,1] = interw[0:3,1] - pt[0:3]
    
    squareNorm = coeff @ coeff.transpose()
    lambdaPropo = (((1/squareNorm) * coeff.transpose()) @ interw[0:3,:]) 

    
    return lambdaPropo,1


    
    
def minLambda(lambdaPropo1,lambdaPropo2):
    
    """
    Compute the common segment between two images
    
    Inputs : 
        
    lambdaPropo1 : 2D vector
        2 values of lambda which represents the two intersection between the line and the slice1
    lambdaPropo2 : 2D vector
        2 values of lambda which represents the two intersection between the line and the slice2
    
    Outputs : 
        
    lambdaMin : 2D vector
        2 values of lamda which represents the common segment between the 2 slices
        
    """
    
    lambdaMin = np.zeros(2)
    lambdaMin[0] = min(min(lambdaPropo1),min(lambdaPropo2))
    lambdaMin[1] = max(max(lambdaPropo1),max(lambdaPropo2))
    
    #lambdaMin[0] = max(min(lambdaPropo1),min(lambdaPropo2))
    #lambdaMin[1] = min(max(lambdaPropo1),max(lambdaPropo2))
    
    
    return lambdaMin #Return 2 values of lambda that represent the common segment between the 2 slices

def commonSegment(Slice1,Slice2):
    """
    Compute the coordinates of the two extremity points of the segment in the 2 image plans

    Inputs : 
    
    Slice1 : slice
        contains all the necessary information on the slice 1, including the transformation M into the 3D space and the information on the header
    Slice2: slice
        Contains all the necessary information on the slice 2, including the transformation M into the 3D space and the information on the header   
        
        
    Outputs : 
    
    pointImg1 : 3x2 matrix
        the extremites of the segment in the slice1 plan
    pointImg2 : 3x2 matrix
        the extremites of the segment in the slice1 plan
    nbpoint : integer
        number of points between the two extremities
    ok : interger
        1 if the common segment was computed well, 0 else    
        

    """
    
    M1 = Slice1.get_transfo()
    M2 = Slice2.get_transfo()
    
    coeff,pt,ok = intersectionLineBtw2Planes(M1,M2)
    
    
    if ok<1: #if there is no intersection lines (the 2 planes are parralel) it is useless to compute the intersectionSegment
        return 0,0,0,0
    
    
    lambdaPropo1,ok = intersectionSegment(Slice1,coeff,pt) #if there is no intersection segment (the line of intersection is outisde of the image or on a corner), it useless to compute a common segment
    
   
    if ok<1:
        return 0,0,0,0
    
    lambdaPropo2,ok = intersectionSegment(Slice2,coeff,pt)
    
    
    if ok<1:
        return 0,0,0,0
    
    
    lambdaMin = minLambda(lambdaPropo1,lambdaPropo2)
        
    if lambdaMin[0]==lambdaMin[1]: #the segment is nul, there is no intersection
        return 0,0,0,0
        
    M1 = Slice1.get_transfo()

    M2 = Slice2.get_transfo()

    point3D = np.zeros((3,2))
    
    
    point3D[0:3,0] = lambdaMin[0] * coeff + pt #Point corresponding to the value of lambda
    point3D[0:3,1] = lambdaMin[1] * coeff + pt
    point3D = np.concatenate((point3D,np.array([[1,1]])))
    
        
    pointImg1 = np.zeros((4,2))
    pointImg1[3,:] = np.ones((1,2))
    pointImg2 = np.zeros((4,2))
    pointImg2[3,:] = np.ones((1,2))
    
    pointImg1 = np.linalg.inv(M1) @ point3D

    pointImg2 = np.linalg.inv(M2) @ point3D 
    
    distance1 = np.linalg.norm(pointImg1[0:2,0] - pointImg1[0:2,1]) #distance between two points on the two images
    distance2 = np.linalg.norm(pointImg2[0:2,0] - pointImg2[0:2,1]) 
        
    res = min(Slice1.get_slice().header.get_zooms()) #the smaller resolution of a voxel
        
    if res<0: #probmem with the resolution of the image
        return 0,0,0,0
        
    if max(distance1,distance2)<1: #no pixel in commun
        return 0,0,0,0
        
    nbpoint = int(np.round(max(distance1,distance2)+1)/res) #choose the max distance and divide it by the smaller resolution 
        

    return pointImg1[0:3],pointImg2[0:3],nbpoint,1


def sliceProfil(Slice,pointImg,nbpoint):
    """
    Interpol values on the segment to obtain the profil intensity

    Inputs : 
        
    Slice : slice
        type slice, contains all the necessary information about the slice, including data and mask 
    pointImg : 3x2 matrix
        the extremites of the segment in the slice plan (x1,x2;y1,y2)
    nbpoint : integer
        number of points between the two extremities

    Ouputs : 
    
    interpol : nbpointx1 vector
        values of intensity
    index : boolean nbpointx1 vector
        index of interest
    """
    if nbpoint == 0:
        return 0,0
    interpol= np.zeros(nbpoint)
    interpolMask = np.zeros(nbpoint)
    pointInterpol = np.zeros((2,nbpoint))
    pointInterpol[0,:] = np.linspace(pointImg[0,0],pointImg[0,1],nbpoint)
    pointInterpol[1,:] = np.linspace(pointImg[1,0],pointImg[1,1],nbpoint)
      
    mask = Slice.get_mask()
    map_coordinates(Slice.get_slice().get_fdata(), pointInterpol , output=interpol, order=1, mode='constant', cval=np.nan, prefilter=False)
    map_coordinates(mask, pointInterpol, output=interpolMask, order=0, mode='constant',cval=np.nan,prefilter=False)
    
    index =~np.isnan(interpol)*interpolMask>0
    #val_mask = interpol * interpolMask
    #index=val_mask>0
      
    return interpol,index
  
def commonProfil(val1,index1,val2,index2,nbpoint):
    """
    
    Compute the intensity of points of interest in the first slice or the second slice
    Inputs :
    
    val1 : nbpointx1 vector
        values of intensity in the first slice
    index1 : nbpointx1 vector
        values of interest in the first slice
    val2 : nbpointx1 vector
        values of intensity in the seconde slice
    index2 : nbpointx1 vector
        values of interest in the second slice

    Output :
    
    val1[index] : vector of the size of index
        values of interset in val1
    
    val2[index] : vector of the size of index
        values of interest in val2

    """
    if nbpoint==0:
        return 0,0
    valindex=np.linspace(0,nbpoint-1,nbpoint,dtype=int)
    index = index1+index2 
    index = valindex[index==True]

    return val1[index],val2[index],index
    

def loadSlice(img,mask,listSlice):
    
    for z in range(img.shape[2]): #Lecture des images
    
        slice_img = img.get_fdata()[:,:,z]
        mz = np.eye(4)
        mz[2,3]= z
        sliceaffine = img.affine @ mz
        nifti = nib.Nifti1Image(slice_img,sliceaffine)
        slice_masque = mask.get_fdata()[:,:,z]
        c_im1 = SliceObject(nifti,slice_masque)
        listSlice.append(c_im1)
        
    return listSlice
    
    #map_coordinates(input, coordinates)
    
# def erreur(commonVal1,commonVal2):
    
#     erreur = 0
    
#     for i in range(commonVal1.shape[0]):
#         e = abs(commonVal1[i] - commonVal2[i])
#         erreur = erreur + e 
        
#     return erreur, commonVal1.shape[0], erreur/commonVal1.shape[0]

# def optimization(erreur,N,Slice):
#     minimize(erreur/N,Slice.get_parameters,)

# #Fonction qui interpole les valeurs d intensite sur le segment d intersection et renvoit le profil d intensite
# def SliceProfil(Coupe,PtsegmentCoupe,nbpoint):
    
    
#     #nbpointx = np.abs(int(PtsegmentCoupe[0,0]) - int(PtsegmentCoupe[0,1])) #nb de points du segment
#     x = np.linspace(int(PtsegmentCoupe[0,0]), int(PtsegmentCoupe[0,1]), num=nbpoint, endpoint=True, retstep=False, dtype=int, axis=0)  #donne les coordonnees des points de x du segments
#     y = np.linspace(int(PtsegmentCoupe[1,0]), int(PtsegmentCoupe[1,1]), num=nbpoint, endpoint=True, retstep=False, dtype=int, axis=0)  #donne les coordonnees des points de y du segments
#     masque = Coupe.get_masque()
#     #Coupe_sortie = Coupe.get_data().copy()
#     val = np.zeros((nbpoint,2))
    
#     i=0
#     for i in range(x.shape[0]):
#         if masque.get_fdata()[x[i],y[i]]>0:
#            val[i,:] = [x[i],y[i]]
#             #Coupe_sortie[x[i],y[i]] = 0
    
#     #nifti_sortie = nib.Nifti1Image(Coupe_sortie,Coupe1.get_affine())
    
#     print(np.transpose(val))
#     print(x)
#     print(y)
#     sortie = np.zeros(nbpoint)
#     #masque_sortie = np.zeros(nbpoint)
    
#     map_coordinates(Coupe.get_data(),np.transpose(val),output=sortie, order=1, mode='constant', cval=0.0, prefilter=True)
    
#     index = sortie>0
    
#     return sortie,index


# # def CalculErreur(Profil1,Profil2,nbpoint):
        
# #     erreur = 0
        
# #     for i in range(nbpoint-1):
# #         erreur = erreur + (Profil1[i]-Profil2[i])*(Profil1[i]-Profil2[i])
        
# #     return erreur 
    





# ###Test : 
    
Coupe_img1 = []
Coupe_img2 = []
Coupe_img3 = []
Masque_img = []


Coupe_img1 = loadSlice(img1, masque1, Coupe_img1)    
Coupe_img2 = loadSlice(img2, masque2, Coupe_img2)    
Coupe_img3 = loadSlice(img3, masque3, Coupe_img3)    

Coupe1 = Coupe_img1[7]
#Masque1 = Masque_img[4]
Coupe2 = Coupe_img2[6]

#coeff,pt,ok = intersectionLineBtw2Planes(Coupe1.get_transfo(), Coupe2.get_transfo()) #calcul de la droite d intersection

# if ok>0:
#     lambdaPropo1,ok = intersectionSegment(Coupe1, coeff, pt)

#if ok>0:
#lambdaPropo2,ok = intersectionSegment(Coupe2,coeff, pt)    

# if ok>0:
pointImg1,pointImg2,nbpoint,ok = commonSegment(Coupe1,Coupe2)

if ok>0:
    
    
    val1,index1=sliceProfil(Coupe1, pointImg1, nbpoint)
    val2,index2=sliceProfil(Coupe2, pointImg2, nbpoint)
    commonVal1,commonVal2,index = commonProfil(val1, index1, val2, index2,nbpoint)
 
    #Before applying the mask
    title1 = 'Intersection segment for image1, without mask'
    title2 = 'Intersection segment for image2, without mask'
    plotsegment(Coupe1,pointImg1,ok,nbpoint,title1,mask=np.nan,index=np.nan)
    plotsegment(Coupe2,pointImg2,ok,nbpoint,title2,mask=np.nan,index=np.nan)
   
    #After applying the mask
    title1 = 'Intersection segment for image1, with mask'
    title2 = 'Intersection segment for image2, with mask'
    plotsegment(Coupe1,pointImg1,ok,nbpoint,title1,mask=Coupe1.get_mask(),index=index)
    plotsegment(Coupe2,pointImg2,ok,nbpoint,title2,mask=Coupe2.get_mask(),index=index)

    plt.figure()
    plt.plot(commonVal1)
    plt.plot(commonVal2)

# print(coeff,pt)
# seg1,inter1,lambdaPropo1 = sliceSegmentIntersection(Coupe1, coeff, pt) #calcul du segment d intersection pour chaque image
# seg2,inter2,lambdaPropo2 = sliceSegmentIntersection(Coupe2, coeff, pt)
# ptimg1 = plotsegment(Coupe1, coeff, pt, lambdaPropo1)

# # print(seg1,seg2)
# # PtsegmentCoupe1, PtsegmentCoupe2, nbpoint1, nbpoint2 = IntersectionEntre2Coupes(Coupe1, seg1, Coupe2, seg2) #Calcul de la partie commune d intersection pour les deux images

# # Centre = CentreRotation(Masque1)

# plt.figure() #Affiche les images qui nous interessent
# show_slice([Coupe1.get_data(),Coupe2.get_data()])

# x1 = inter1[0,:]
# y1 = inter1[1,:]
# # x1bis = PtsegmentCoupe1[0,:]
# # y1bis = PtsegmentCoupe1[1,:]

# plt.figure()
# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(Coupe1.get_data().T,cmap="gray",origin="lower") #Affiche les images avec les segments d intersection (celui qui prend toute l image en bleu et le commun en vert), pour la coupe1
# axes[0].plot(x1,y1,marker='o')
# # axes[0].plot(x1bis,y1bis,marker='o',color='g')
# #axes[0].scatter(Centre[0],Centre[1],color='r')
# #axes[0].plot(Centre,marker='o')

# x2 = inter2[0,:]
# y2 = inter2[1,:]
# # x2bis = PtsegmentCoupe2[0,:]
# # y2bis = PtsegmentCoupe2[1,:]

# axes[1].imshow(Coupe2.get_data().T,cmap="gray",origin="lower") #Affiche les intersections pour la coupe2
# axes[1].plot(x2,y2,marker='o')
# axes[1].plot(x2bis,y2bis,marker='o',color='g')

# # val1,nbpoint1= PointsCoupe(Coupe1, PtsegmentCoupe1)
# # val2,nbpoint2= PointsCoupe(Coupe2, PtsegmentCoupe2)

# nbpoint=max(nbpoint1,nbpoint2) #Calcul le segment qui a le plus grand nombre de points
# profil1,index1 = ProfilCoupe(Coupe1,PtsegmentCoupe1,nbpoint) #Interpolation des valeurs du segment sur celui qui a le plus grand nombre de points
# profil2,index2 = ProfilCoupe(Coupe2,PtsegmentCoupe2,nbpoint)

# if index1.shape[0]>index2.shape[0]:
#     index=index1
# else:
#     index=index2
    
    
# Coupe2avcMasque = Coupe2.get_masque().get_fdata()*Coupe2.get_data(); 
# Coupe1avcMasque = Coupe1.get_masque().get_fdata()*Coupe1.get_data(); 
# plt.figure()
# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(Coupe2avcMasque,cmap="gray",origin="lower") #Affiche la coupe1 avec le masque 
# axes[1].imshow(Coupe1avcMasque,cmap="gray",origin="lower")


# # e = CalculErreur(profil1,profil2,nbpoint)

# # x2,y2,nifti_sortie2,val2,nbpoint2,interpol2 = ProfilCoupe(Coupe2, PtsegmentCoupe2)

# # plt.figure()
# # fig, axes = plt.subplots(1, 2)
# # axes[0].imshow(nifti_sortie1.get_data().T,cmap="gray",origin="lower")

# # axes[1].imshow(Coupe1.get_data().T,cmap="gray",origin="lower")
# # axes[1].imshow(Coupe1.get_data().T,cmap="gray",origin="lower")
# # axes[1].plot(x1bis,y1bis,marker='o',color='g')

# # plt.figure()
# # fig, axes = plt.subplots(1, 2)
# # axes[0].imshow(nifti_sortie2.get_data().T,cmap="gray",origin="lower")

# # axes[1].imshow(Coupe2.get_data().T,cmap="gray",origin="lower")
# # axes[1].imshow(Coupe2.get_data().T,cmap="gray",origin="lower")
# # axes[1].plot(x2bis,y2bis,marker='o',color='g')

# plt.figure()
# plt.plot(profil1[index])
# plt.plot(profil2[index])

# # plt.figure()
# # fig, axes = plt.subplots(1, 2)
# # axes[0].imshow(Coupe1avcMasque.T,cmap="gray",origin="lower")
# # axes[0].plot(val1[:,0],val1[:,1],marker='o',color='r')
# # axes[1].imshow(Coupe2avcMasque.T,cmap="gray",origin="lower")
# # axes[1].plot(val2[:,0],val2[:,1],marker='o',color='r')


# # axes[1].plot(xx,interpol1(xx))
# # axes[1].plot(xx,interpol2(xx))
