#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 13:59:55 2021

@author: mercier
"""


import nibabel as nib 
import numpy as np
import math
import matplotlib.pyplot as plt 
from scipy.ndimage import map_coordinates


##Chargement des images
img1 = nib.load('../donnee/transfer_2241503_files_0c0883a5/20090325_T2_AXIAL_01_crop.nii.gz')
img2 = nib.load('../donnee/transfer_2241503_files_0c0883a5/20090325_T2_SAG_01_crop.nii.gz')
img3 = nib.load('../donnee/transfer_2241503_files_0c0883a5/20090325_T2_COR_01_crop.nii.gz')
masque1 = nib.load('../donnee/transfer_2241503_files_0c0883a5/20090325_T2_AXIAL_01_mask.nii.gz')
masque2 = nib.load('../donnee/transfer_2241503_files_0c0883a5/20090325_T2_SAG_01_mask.nii.gz')
masque3 = nib.load('../donnee/transfer_2241503_files_0c0883a5/20090325_T2_COR_01_mask.nii.gz')

class Coupe:
    def __init__(self,slice,nbcoupe,masque)    :
        
        self.slice = slice
        self._data = self.slice.get_fdata()
        self._parametre = np.zeros(6)
        self._trigide = np.eye(4)
        CRot = CentreRotation(masque)
        self._centre = np.eye(4)
        self._centreinv = np.eye(4)
        self._centre[0:4,3] = CRot
        self._centreinv[0:4,3] = -CRot
        mz = np.eye(4)
        mz[2,3]=z
        self._affine = self.slice.affine @ mz
        self._transfo = self._centreinv @ (self._trigide @ (self._centre @ self._affine))
        

    #fonction get et set
    def get_data(self):
        return self._data
    
    def get_parameters(self):
        return self._parameter
    
    def set_parameters(self,x):
        self.parameter = x
        
        
    def get_transfo(self):
        return self._transfo
    
    def get_affine(self):
        return self._affine
    
#Calcul l intersection entre deux coupes et renvoit les points en communs
def DroiteIntersection2Coupes(Coupe1,Coupe2) : 
   
    #Calcul la normale au plan Oxy
   M1 = Coupe1.get_transfo()
   M2 = Coupe2.get_transfo()
   
   n1 = np.cross(M1[0:3,0],M1[0:3,1])
   print("M1 = ",M1)
   n1norm = n1/np.linalg.norm(n1)
   n1 = n1norm
   print("n1norm =", n1)
   t1 = M1[0:3,3]
   
   n2 = np.cross(M2[0:3,0],M2[0:3,1])
   print("n1 = ",n1)
   n2norm = n2/np.linalg.norm(n2)
   n2 = n2norm
   print("n2norm =", n2)
   t2 = M2[0:3,3]
   
   alpha = n1 @ n2
   beta =  n1 @ t1
   gamma = n2 @ t2
   
   a = 1/(1 - alpha*alpha)
   g = a*(beta - alpha*gamma)
   h = a*(gamma - alpha*beta)
   print("a = ",alpha)
   print("g = ",gamma)
   print("h = ",beta)
   
   
   #equation de la droite
   coeff = np.cross(n1,n2)
   pt = g*n1 + h*n2
   
   return coeff, pt
 
    
def SegmentIntersectionCoupe(Coupe,coeff,pt):
    
    #equation de la droite dans le plan de l image
    M = Coupe.get_transfo()
    
    rotinv = np.linalg.inv(M[0:3,0:3])
    trans = M[0:3,3]
    ab = rotinv @ coeff
    print("ab = ", ab)
    ptplans = rotinv @ (pt - trans) 
    print("ptplans =", ptplans)
     
    #calcul des coordonnees cartesiennes de la droite
    a = -ab[1]
    b = ab[0]
    c = -(a*ptplans[0] + b*ptplans[1])
    print("C =",c)
    
    #Calcul de l intersection avec le plan
    intersection = np.zeros((3,2))
    long = Coupe.slice.shape[0]-1
    larg = Coupe.slice.shape[1]-1
 
    i=0
    if (-c/a) > 0 and (-c/a) < long: 
        intersection[0,i] =  -c/a
        intersection[1,i] =  0
        i=i+1
    
    if (-c/b)>0 and (-c/b)< larg:
        intersection[0,i] = 0 
        intersection[1,i] = -c/b
        i=i+1
   
    if ((-c-a*long)/b)>0  and ((-c-a*long)/b) < larg:
        intersection[0,i] = long
        intersection[1,i] = (-c-a*long)/b;
        i=i+1
  
    if (((-c-b*larg)/a)>0) and (((-c-b*larg)/a)<long) :
        intersection[0,i] = (-c-b*larg)/a;
        intersection[1,i] =larg;
        
    
    #On recalcule les points dans l espace monde
    intermonde = np.zeros((3,2))
    rot = M[0:3,0:3]
    intermonde[0:3,0] = rot @ intersection[0:3,0] + trans
    intermonde[0:3,1] = rot @ intersection[0:3,1] + trans

    
    return intermonde[0:3,:],intersection[0:2,:]

def show_slice(slices): #definition de la fonction slice qui prend en paramètre une image
        #""Function di display row of image slices""
         fig, axes = plt.subplots(1, len(slices))
         for i, slice in enumerate(slices):
             axes[i].imshow(slice.T,cmap="gray",origin="lower") #affiche l'image en niveau de gris (noir pour la valeur minimale et blanche pour la valeur maximale)

        
def CentreRotation(masque):
    
    X,Y = masque.shape
    img = masque.get_fdata()
    Centre = np.zeros(2)
    Somme_x = 0
    Somme_y = 0
    nbpoint = 0
    
    for i in range(X):
        for j in range(Y):
                if img[i,j] == 1:
                    Somme_x = Somme_x + i
                    Somme_y = Somme_y + j
                    nbpoint = nbpoint + 1
    Centre[0] = int(Somme_x/nbpoint)
    Centre[1] = int(Somme_y/nbpoint) 
     
    CentreMonde = np.concatenate((Centre,np.array([0,1])))
    CentreMonde = masque.affine @ CentreMonde 
    return CentreMonde
    

def IntersectionEntre2Coupes(Coupe1,seg1,Coupe2,seg2):
    
    #sortiinter1 = inter1[inter1[0,:].argsort(kind='mergesort')]
    #sortseg1 = np.concatenate((seg1,np.array([[1,1]])))
    #sortinter2 = inter2[inter2[0,:].argsort(kind='mergesort')]
    #sortseg2 = np.concatenate((seg2,np.array([[1,1]])))
    sortseg1 = np.zeros((3,2))
    sortseg2 = np.zeros((3,2))
    
    if (seg1[0,0]<seg1[0,1]):
        sortseg1[:,0]=seg1[:,0]
        sortseg1[:,1]=seg1[:,1]
    else:
        sortseg1[:,0]=seg1[:,1]
        sortseg1[:,1]=seg1[:,0]
        
    if (seg2[0,0]<seg2[0,1]):
        sortseg2[:,0]=seg2[:,0]
        sortseg2[:,1]=seg2[:,1]
    else:
        sortseg2[:,0]=seg2[:,1]
        sortseg2[:,1]=seg2[:,0]
    
    print("sortseg1 = ", sortseg1)
    print("sortseg2 = ", sortseg2)
    
    Ptsegment = np.zeros((3,2))
    #Ptsegment[3,:] = np.ones((1,2))
    
    if(sortseg1[0,0]>sortseg2[0,0]):
        Ptsegment[:,0] = sortseg1[:,0]
    else:   
        Ptsegment[:,0] = sortseg2[:,0]
    if(sortseg1[0,1]<sortseg2[0,1]):
        Ptsegment[:,1] = sortseg1[:,1]
    else:   
        Ptsegment[:,1] = sortseg2[:,1]  
    
    #Calcul les coordoonees des points pour les deux images
    M1 = Coupe1.get_transfo()
    rotM1 = np.linalg.inv(M1[0:3,0:3])
    transM1 = M1[0:3,3]
    PtsegmentCoupe1 = np.zeros((3,2))
    PtsegmentCoupe1[0:3,0] = rotM1 @ (Ptsegment[0:3,0] - transM1)
    PtsegmentCoupe1[0:3,1] = rotM1 @ (Ptsegment[0:3,1] - transM1)
     
    M2 = Coupe2.get_transfo()
    rotM2 = np.linalg.inv(M2[0:3,0:3])
    transM2 = M2[0:3,3]
    PtsegmentCoupe2 = np.zeros((3,2))
    PtsegmentCoupe2[0:3,0] = rotM2 @ (Ptsegment[0:3,0] - transM2)
    PtsegmentCoupe2[0:3,1] = rotM2 @ (Ptsegment[0:3,1] - transM2)
    
    print("Ptsegment = ", Ptsegment)
    return PtsegmentCoupe1, PtsegmentCoupe2, 

def PointsCoupe(Coupe,PtsegmentCoupe):
    
    #1) On creer une ligne entre les deux points
    nbpointx = np.abs(int(PtsegmentCoupe[0,0]) - int(PtsegmentCoupe[0,1]))
    x = np.linspace(int(PtsegmentCoupe[0,0]), int(PtsegmentCoupe[0,1]), num=nbpointx, endpoint=False, retstep=False, dtype=int, axis=0)
    y = np.linspace(int(PtsegmentCoupe[1,0]), int(PtsegmentCoupe[1,1]), num=nbpointx, endpoint=True, retstep=False, dtype=int, axis=0) 
    
    # Coupe_sortie = Coupe.get_data().copy()
    val = np.zeros((nbpointx,2))
    
    i=0
    for i in range(x.shape[0]):
        #Coupe_sortie[x[i],y[i]] = 0
        val[i,:] = [x[i],y[i]]

    # nifti_sortie = nib.Nifti1Image(Coupe_sortie,Coupe1.get_affine())

    return val,nbpointx


def  ProfilCoupe(nbpoint,val,Coupe)   :
    
    sortie = np.zeros(nbpoint)
    interpol = map_coordinates(Coupe.get_data(),np.transpose(val),output=sortie, order=3, mode='constant', cval=0.0, prefilter=True)

    return interpol

###Test : 
    
Coupe_img1 = []
Coupe_img2 = []
Coupe_img3 = []
Masque_img = []


for z in range(img1.shape[2]):
    slice_im1 = img1.get_fdata()[:,:,z]
    slice_im1 = nib.Nifti1Image(slice_im1,img1.affine)
    slice_masque = masque1.get_fdata()[:,:,z]
    slice_masque = nib.Nifti1Image(slice_masque,masque1.affine)
    c_im1 = Coupe(slice_im1,z,slice_masque)
    Coupe_img1.append(c_im1)
    Masque_img.append(slice_masque)
    
for z in range(img2.shape[2]):
    slice_im2 = img2.get_fdata()[:,:,z]
    slice_im2 = nib.Nifti1Image(slice_im2,img2.affine)
    slice_masque = masque2.get_fdata()[:,:,z]
    slice_masque = nib.Nifti1Image(slice_masque,masque2.affine)
    c_im2 = Coupe(slice_im2,z,slice_masque)
    Coupe_img2.append(c_im2)
    
    
for z in range(img3.shape[2]):
    slice_im3 = img3.get_fdata()[:,:,z]
    slice_im3 = nib.Nifti1Image(slice_im3,img3.affine)
    slice_masque = masque3.get_fdata()[:,:,z]
    slice_masque = nib.Nifti1Image(slice_masque,masque3.affine)
    c_im3 = Coupe(slice_im3,z,slice_masque)
    Coupe_img3.append(c_im3)

Coupe1 = Coupe_img3[5]
Masque1 = Masque_img[4]
Coupe2 = Coupe_img2[1]


coeff,pt = DroiteIntersection2Coupes(Coupe1, Coupe2)
print(coeff,pt)
seg1,inter1 = SegmentIntersectionCoupe(Coupe1, coeff, pt)
seg2,inter2 = SegmentIntersectionCoupe(Coupe2, coeff, pt)
print(seg1,seg2)
PtsegmentCoupe1, PtsegmentCoupe2 = IntersectionEntre2Coupes(Coupe1, seg1, Coupe2, seg2)

Centre = CentreRotation(Masque1)

plt.figure()
show_slice([Coupe1.get_data(),Coupe2.get_data()])

x1 = inter1[0,:]
y1 = inter1[1,:]
x1bis = PtsegmentCoupe1[0,:]
y1bis = PtsegmentCoupe1[1,:]

plt.figure()
fig, axes = plt.subplots(1, 2)
axes[0].imshow(Coupe1.get_data().T,cmap="gray",origin="lower")
axes[0].plot(x1,y1,marker='o')
axes[0].plot(x1bis,y1bis,marker='o',color='g')
#axes[0].scatter(Centre[0],Centre[1],color='r')
#axes[0].plot(Centre,marker='o')

x2 = inter2[0,:]
y2 = inter2[1,:]
x2bis = PtsegmentCoupe2[0,:]
y2bis = PtsegmentCoupe2[1,:]

axes[1].imshow(Coupe2.get_data().T,cmap="gray",origin="lower")
axes[1].plot(x2,y2,marker='o')
axes[1].plot(x2bis,y2bis,marker='o',color='g')

val1,nbpoint1= PointsCoupe(Coupe1, PtsegmentCoupe1)
val2,nbpoint2= PointsCoupe(Coupe1, PtsegmentCoupe1)

nbpoint=max(nbpoint1,nbpoint2)
profil1 = ProfilCoupe(nbpoint,val1,Coupe1)
profil2 = ProfilCoupe(nbpoint,val2,Coupe2)



# x2,y2,nifti_sortie2,val2,nbpoint2,interpol2 = ProfilCoupe(Coupe2, PtsegmentCoupe2)

# plt.figure()
# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(nifti_sortie1.get_data().T,cmap="gray",origin="lower")

# axes[1].imshow(Coupe1.get_data().T,cmap="gray",origin="lower")
# axes[1].imshow(Coupe1.get_data().T,cmap="gray",origin="lower")
# axes[1].plot(x1bis,y1bis,marker='o',color='g')

# plt.figure()
# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(nifti_sortie2.get_data().T,cmap="gray",origin="lower")

# axes[1].imshow(Coupe2.get_data().T,cmap="gray",origin="lower")
# axes[1].imshow(Coupe2.get_data().T,cmap="gray",origin="lower")
# axes[1].plot(x2bis,y2bis,marker='o',color='g')

# plt.figure()
# plt.plot(profil1)
# plt.plot(profil2)


# axes[1].plot(xx,interpol1(xx))
# axes[1].plot(xx,interpol2(xx))