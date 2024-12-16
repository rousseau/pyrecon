#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:32:32 2017

@author: faisan
"""




from numba import jit
from math import cos,sin,pi
import numpy as np
import nibabel as nib 
from scipy.ndimage.interpolation import affine_transform
from numpy.linalg import inv

# size est la taille d'un pixel en mm.
# M,T est la transformation qui fait passer de l'espace pixel à l'espace millimétrique.
# on suppose que le premier pixel a pour coordoonees size/2 dans l'espace millimérique (validation de cette hypothèse
# dans la fonction principale main de ce fichier)
def createTransfoInmmWord(size):
    M = np.diag(size)
    T = size/2
    return matriceCoordHomogene(M,T) 


# eulers : angles d'euler en degré
# M : matrice de rotation (cf http://www.fil.ion.ucl.ac.uk/spm/doc/books/hbf2/pdfs/Ch2.pdf)
# avec la convention prise ici, si on veut itérer sur toutes les matrices de rotation possible, le premier et le troisième angle peuvent aller de 0 à 360, 
# et le second de 0 à 180 (on aurait pu faire autemenent). 
#on a : createRotationMatrix([theta1,theta2,theta3]) = createRotationMatrix([180+theta1,180-theta2,180+theta3]) 
def createRotationMatrix(eulers):    
    angles = eulers.copy()
    angles = angles / 180 * pi    
    c4 = cos(angles[0])
    s4 = sin(angles[0])
    c5 = cos(angles[1])
    s5 = sin(angles[1])
    c6 = cos(angles[2])
    s6 = sin(angles[2])    
    M = np.array([[c5*c6,c5*s6,s5],[-s4*s5*c6-c4*s6,-s4*s5*s6+c4*c6,s4*c5],[-c4*s5*c6+s4*s6,-c4*s5*s6-s4*c6,c4*c5]])    
    return M


#rend la matrice de transformation (coordonnées homogène) à partir de la rotation M et de la translation T
def matriceCoordHomogene(M,T):
    Mres = np.zeros((4,4))
    Mres[0:3,0:3]=M
    Mres[3,3]=1
    Mres[0:3,3]=T
    return Mres



#im1 : image dont on veut une version réduite.
#desireVolumePixel :dimension des pixels de l image resultat
# cette fonction retourne l'image résultats.
# commentaires : cette fonction suppose que l'image doit être réduite et non zoomée (le zoom peut être égal à 1 pour certains axes)
def underSample(im1,desiredVolumePixel):
    M1 = createTransfoInmmWord(desiredVolumePixel)           
    M2 = createTransfoInmmWord(np.asarray(im1.header.get_zooms()))
    M = inv(M2).dot(M1)
    
    T = M[0:3,3]
    M = M[0:3,0:3]
    
    zoom = np.diag(M)    
    res = affine_transform(im1.get_data(), M, offset=T, output_shape=None,  order=3, mode='constant', cval=0.0, prefilter=False)
        
    MaxCoord = np.ceil((im1.get_data().shape-T)/zoom)
    MaxCoord = MaxCoord.astype(int)
    res2 = res[0:MaxCoord[0],0:MaxCoord[1],0:MaxCoord[2]]     
    return res2
    

#volumePixel : taille des pixels pour les 2 images à recaler;
#latrgestSize : volume de départ : c'est un réel -> le volume de départ est un cube
#cette fonction retourne les différentes tailles de pixel où l'on va passer.
#Algo : on multiplie la taille par 2 (à condition qu'elle ne devienne pas supérieur à la taille de l'image d'origine).
#Exemple : 
#si on a une image de taille 1,1,4 mm et une autre de taille 2,2,1 on aura volumePixel = 1,1,4,2,2,1 et largestSize=6
# on aura s[1] = 6 6 6 6 6 6  (a la resolution la plus basse, on aura des images de taille 6x6x6)
# on aura s[2] = 3 3 4 3 3 3  (la premiere image aura une résolution 3x3x4 et l'autre 3x3x3)
# on aura s[3] = 1.5 1.5 4 2 2 1.5  (la premiere image aura une résolution 3x3x4 et l'autre 3x3x3)
# on aura s[4] = 1 1 4 2 2 1  (la résolution d origine)
#Remarque 1 : si on a une image tres mal resolue dans une dimension, si largestSize<une des tailles, alors dès le départ, l'image ne sera pas isotrope.
#Remarque 2 : a la fin, si on est proche de la résolution d origine sans l atteindre, on se permet de sauter une résolution (cf fonction isclose)
#Exemple pour les deux remarques:
#si on a une image de taille 1,1,14 mm et une autre de taille 0.9,0.9,0.9 et largestSize = 4
#on aura s[1] = 4 4 14 4 4 4 (cf remarque 1)
#on aura s[2] = 2 2 14 2 2 2 (cf remarque 1)
#on aura s[3] = 1 1 14 0.9 0.9 0.9 (cf remarque 2 et non 1 1 14 1 1 1 qui aurait nécessité d'avoir la résolution d'origine en s[4])

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
      

  
if __name__ == '__main__':  
    #exemple d utilisation des fonctions ci-dessus : on a deux images que l on veut recaler, et on calcule leurs barycentres à différentes résolutions, et on 
    #regarde si les résultats sont cohérents si on passe d'une résolution à l'autre (en transformant les images)
    imgref  = nib.load("/home/miv/faisan/patient16_exam02_T1GADO.nii.gz")
    imgtr = nib.load("/home/miv/faisan/patient16_exam02_T2.nii.gz")

    volumePixel1 =  np.asarray(imgtr.header.get_zooms())
    volumePixel2 =  np.asarray(imgref.header.get_zooms())
    volumePixel = np.zeros(6)
    volumePixel[0:3]=volumePixel1
    volumePixel[3:6]=volumePixel2
    s = sequenceofundersampling(volumePixel,8)



    for resolution in range(s.shape[0]):
        size_tr = s[resolution,0:3]
        size_ref = s[resolution,3:6]
 
        print s[resolution,:]
        imgtr_down = underSample(imgtr,size_tr)    
        imgref_down = underSample(imgref,size_ref)
    
   
        #calcul des matrices pour se retrouver dans un espace millimétrique    
        Mtr = createTransfoInmmWord(size_tr)               
        Mref = createTransfoInmmWord(size_ref)           
    
        #calcul des barycentres
        centreTr = computeBarycentre(imgtr_down)        
        centreRef = computeBarycentre(imgref_down)
        centreTr = np.append(centreTr,1)                                 
        centreRef = np.append(centreRef,1)            

        #calcul de la translation à appliquer pour que : M.dot(centreRef) - centreTr)
        T = Mtr.dot(centreTr)-Mref.dot(centreRef)      
        x0  = np.zeros(6)
        x0[3:6] = T[0:3]    
        M = createRotationMatrix(x0[0:3]) #identitité    
        T = x0[3:6]    
        M = matriceCoordHomogene(M,T)  #ici    M.dot(centreRef) != centreTr (il faut prendre en compte Mtr et Mref)
    
        if resolution == 0:
            Mtr0  = Mtr.copy()
            Mref0 = Mref.copy()        
            centreTr0 = centreTr.copy()
            centreRef0 = centreRef.copy()
            M0 = M.copy()    
            
        M = inv(Mtr).dot(M).dot(Mref) # M.dot(centreRef) = centreTr)
        print ("ca doit etre nul" + str(M.dot(centreRef) - centreTr))
        T = M[0:3,3]/M[3,3]    
        Mf = M[0:3,0:3]/M[3][3]     
        res = affine_transform(imgtr_down, Mf, offset=T, output_shape=imgref_down.shape,  order=3, mode='constant', cval=0, prefilter=False)
    
        centredeux = computeBarycentre(res)
    
    
        print ('barycentre tr calculé dans image tr'  + str(centreTr[0:3]))    
        if resolution>0:
            barycentretr = inv(Mtr).dot(Mtr0).dot(centreTr0)                
            print("proche de 0 (difference si on estime la position grâce à l'estimation dans l'image de résolution basse tr)"  + str(barycentretr[0:3]-centreTr[0:3]) )
            barycentretr2 = inv(Mtr).dot(M0).dot(Mref).dot(centreRef)
            print("proche de 0 (difference si on estime la position grâce à l'estimation dans l'image de résolution basse ref)"  + str(barycentretr2[0:3]-centreTr[0:3]) )
        
     
        print ('barycentre ref calcule dans image ref' + str(centreRef[0:3]))
        print("proche de 0 (différence entre le barycente de l'image de ref et le barycentre de l'image transformé tr)"  + str(centreRef[0:3]-centredeux[0:3]) )    
        if resolution>0:
            barycentreref = inv(Mref).dot(Mref0).dot(centreRef0) 
            print("proche de 0 (différence entre le barycentre de l'image de ref et son estimation à partir de l'image de résolution basse ref"  + str(centreRef[0:3]-barycentreref[0:3]) )    
        
        
        
        
        
        
        
        
    
    
    
