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

class Coupe: #Contient les données pour le recalage associe a chaque coupe
    def __init__(self,slice,nbcoupe,masque)    :
        
        self.__slice = slice
        self.__data = self.__slice.get_fdata()
        self.__parametre = np.zeros(6)
        self.__trigide = np.eye(4)
        CRot = CentreRotation(masque)
        self.__centre = np.eye(4)
        self.__centreinv = np.eye(4)
        self.__centre[0:4,3] = CRot
        self.__centreinv[0:4,3] = -CRot
        mz = np.eye(4)
        mz[2,3]=nbcoupe #translation en z pour que la coupe corresponde a l origine
        self.__affine = self.__slice.affine @ mz
        self.__transfo = self.__centreinv @ (self.__trigide @ (self.__centre @ self.__affine))
        self.__masque = masque
        

    #fonction get et set
    def get_data(self):
        return self.__data
    
    def get_parameters(self):
        return self.__parameter
    
    def set_parameters(self,x):
        self.__parameter = x
              
    def get_transfo(self):
        return self.__transfo
    
    def get_affine(self):
        return self.__affine
    
    def get_slice(self):
        return self.__slice
    
    def get_masque(self):
        return self.__masque
    
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
 
#Calcul le segment d intersection de la droite avec l image    
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
    intersection = np.zeros((3,2)) #2 points d intersection de coordonnees i,j,k 
    long = Coupe.get_slice().shape[0]-1
    larg = Coupe.get_slice().shape[1]-1
 
    i=0
    if (-c/a) > 0 and (-c/a) < long: #si y=0 x=-c/a (intersection en bas)
        intersection[0,i] =  -c/a
        intersection[1,i] =  0
        i=i+1
    
    if (-c/b)>0 and (-c/b)< larg: #si x=0 y=-c/b (intersection a gauche)
        intersection[0,i] = 0 
        intersection[1,i] = -c/b
        i=i+1
   
    if ((-c-a*long)/b)>0  and ((-c-a*long)/b) < larg: #si x=long y=-c-a*long (intersection a droite)
        intersection[0,i] = long
        intersection[1,i] = (-c-a*long)/b;
        i=i+1
  
    if (((-c-b*larg)/a)>0) and (((-c-b*larg)/a)<long) : #si y=larg x=-c-b*larg (intersection en haut)
        intersection[0,i] = (-c-b*larg)/a;
        intersection[1,i] =larg;
        
    
    #On recalcule les points dans l espace monde
    intermonde = np.zeros((3,2)) #2 points d intersection de coordonnes x,y,z
    rot = M[0:3,0:3] 
    intermonde[0:3,0] = rot @ intersection[0:3,0] + trans
    intermonde[0:3,1] = rot @ intersection[0:3,1] + trans

    
    return intermonde[0:3,:],intersection[0:2,:]

def show_slice(slices): #definition de la fonction show_slice qui prend en paramètre une image
        #""Function di display row of image slices""
         fig, axes = plt.subplots(1, len(slices))
         for i, slice in enumerate(slices):
             axes[i].imshow(slice.T,cmap="gray",origin="lower") #affiche l'image en niveau de gris (noir pour la valeur minimale et blanche pour la valeur maximale)

#Calcul le centre de gravite du masque et donc de l image        
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
    
    sortseg1 = np.zeros((3,2)) #Trie le segment pour avoir la plus petite valeur de x à gauche et la plus grande a droite
    sortseg2 = np.zeros((3,2))
    
    if (seg1[0,0]<seg1[0,1]): #Trie le segment de l image1
        sortseg1[:,0]=seg1[:,0]
        sortseg1[:,1]=seg1[:,1]
    else:
        sortseg1[:,0]=seg1[:,1]
        sortseg1[:,1]=seg1[:,0]
        
    if (seg2[0,0]<seg2[0,1]): #Trie le segment de l image 2
        sortseg2[:,0]=seg2[:,0]
        sortseg2[:,1]=seg2[:,1]
    else:
        sortseg2[:,0]=seg2[:,1]
        sortseg2[:,1]=seg2[:,0]
    
    print("sortseg1 = ", sortseg1)
    print("sortseg2 = ", sortseg2)
    
    Ptsegment = np.zeros((3,2))
    #Ptsegment[3,:] = np.ones((1,2))
    
    if(sortseg1[0,0]>sortseg2[0,0]): #Choisis la plus grande valeurs de x parmis la plus petite valeur de x pour les deux segments
        Ptsegment[:,0] = sortseg1[:,0]
    else:   
        Ptsegment[:,0] = sortseg2[:,0]
    if(sortseg1[0,1]<sortseg2[0,1]): #Choisis la plus petite valeur de x parmis la plus grande valeur de x pour les deux segments
        Ptsegment[:,1] = sortseg1[:,1]
    else:   
        Ptsegment[:,1] = sortseg2[:,1]  
    
    #Calcul les coordoonees des points pour les deux images
    M1 = Coupe1.get_transfo()
    rotM1 = np.linalg.inv(M1[0:3,0:3])
    transM1 = M1[0:3,3]
    PtsegmentCoupe1 = np.zeros((3,2)) #Point d intersection du segment dans le repere Coupe1
    PtsegmentCoupe1[0:3,0] = rotM1 @ (Ptsegment[0:3,0] - transM1)
    PtsegmentCoupe1[0:3,1] = rotM1 @ (Ptsegment[0:3,1] - transM1)
     
    M2 = Coupe2.get_transfo()
    rotM2 = np.linalg.inv(M2[0:3,0:3])
    transM2 = M2[0:3,3]
    PtsegmentCoupe2 = np.zeros((3,2)) #Point d intersection du segment dans le repere Coupe2
    PtsegmentCoupe2[0:3,0] = rotM2 @ (Ptsegment[0:3,0] - transM2)
    PtsegmentCoupe2[0:3,1] = rotM2 @ (Ptsegment[0:3,1] - transM2)
    
    nbpoint1 = np.abs(int(PtsegmentCoupe1[0,0]) - int(PtsegmentCoupe1[0,1]))
    nbpoint2 = np.abs(int(PtsegmentCoupe2[0,0]) - int(PtsegmentCoupe2[0,1]))
    
    print("Ptsegment = ", Ptsegment)
    return PtsegmentCoupe1, PtsegmentCoupe2, nbpoint1, nbpoint2 #Retourne les points d intersection du segment pour les deux coupes

#Fonction qui interpole les valeurs d intensite sur le segment d intersection et renvoit le profil d intensite
def ProfilCoupe(Coupe,PtsegmentCoupe,nbpoint):
    
    
    #nbpointx = np.abs(int(PtsegmentCoupe[0,0]) - int(PtsegmentCoupe[0,1])) #nb de points du segment
    x = np.linspace(int(PtsegmentCoupe[0,0]), int(PtsegmentCoupe[0,1]), num=nbpoint, endpoint=True, retstep=False, dtype=int, axis=0)  #donne les coordonnees des points de x du segments
    y = np.linspace(int(PtsegmentCoupe[1,0]), int(PtsegmentCoupe[1,1]), num=nbpoint, endpoint=True, retstep=False, dtype=int, axis=0)  #donne les coordonnees des points de y du segments
    masque = Coupe.get_masque()
    #Coupe_sortie = Coupe.get_data().copy()
    val = np.zeros((nbpoint,2))
    
    i=0
    for i in range(x.shape[0]):
        if masque.get_fdata()[x[i],y[i]]>0:
           val[i,:] = [x[i],y[i]]
            #Coupe_sortie[x[i],y[i]] = 0
    
    #nifti_sortie = nib.Nifti1Image(Coupe_sortie,Coupe1.get_affine())
    
    print(np.transpose(val))
    print(x)
    print(y)
    sortie = np.zeros(nbpoint)
    #masque_sortie = np.zeros(nbpoint)
    
    map_coordinates(Coupe.get_data(),np.transpose(val),output=sortie, order=1, mode='constant', cval=0.0, prefilter=True)
    
    index = sortie>0
    
    return sortie,index

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
