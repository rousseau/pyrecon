#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:32:32 2017

@author: faisan
"""


from criterium import criteriumToMinimize,translateToRealParameter,translateToFalseParameter,createSimplex
from transformation import createRotationMatrix,matriceCoordHomogene,sequenceofundersampling,underSample,createTransfoInmmWord
import numpy as np
import nibabel as nib 
from scipy.ndimage.interpolation import affine_transform,map_coordinates
from scipy.optimize import minimize
from numpy.linalg import inv


#M est le nombre de points à prendre par angle
#imageARecaler comporte les images à recaler ainsi que des infos pour cacluler le critere
#centreTr et centreRef : barycentre des deux images à recaler
#minimal est la valeur minimale des angles 
#maximal est la valeur maximale des angles 
#cyclic = true si on veut les intervalles maximales (dans ce cas, on ne regarde pas minimal et maximal)
#cyclic = false : on veut chercher la solution dans des intervalles réduits (on regarde minimal et maximal)
#Remarque : si cyclic == true, M doit etre pair, (de manière à passer par le point 180)...
#pour chaque angle, on va optimiser la translation, on rend donc x, ty et tz.
def iterateOverCoarseGrid(M,imageARecaler,centreTr,centreRef,cyclic,minimal=0,maximal=360):    
    if cyclic == True:
        if M%2 == 1:
            print("M doit etre pair")
            exit(0)            
        euler1 = np.linspace(0,360,num = M, endpoint = False)
        euler2 = np.linspace(0,180,num = M/2, endpoint = False)
        euler3 = np.linspace(0,360,num = M, endpoint = False)
    else:
        euler1 = np.linspace(minimal,maximal,num=M,endpoint=True)
        euler2 = np.linspace(minimal,maximal,num=M,endpoint=True)
        euler3 = np.linspace(minimal,maximal,num=M,endpoint=True)
        
    
    tx = np.zeros((euler1.shape[0],euler2.shape[0],euler3.shape[0]))
    ty = np.zeros((euler1.shape[0],euler2.shape[0],euler3.shape[0]))
    tz = np.zeros((euler1.shape[0],euler2.shape[0],euler3.shape[0]))
    eulerAngle = np.zeros(3)
    
    for an1 in range(euler1.shape[0]): 
        eulerAngle[0] = euler1[an1]          
        for an2 in range(euler2.shape[0]):
            eulerAngle[1] = euler2[an2]
            for an3 in range(euler3.shape[0]):
                
                eulerAngle[2] = euler3[an3]
                Mrot = createRotationMatrix(eulerAngle)
                M    = matriceCoordHomogene(Mrot,[0,0,0])
                T    = imageARecaler.Mtr.dot(centreTr)- M.dot(imageARecaler.Mref).dot(centreRef) #on initialise T pour que les barycentres coincident                
                imageARecaler.Mrot = Mrot                
                
                x0 = translateToFalseParameter(T[0:3],imageARecaler)
                
                sizeSimplex = np.asarray(imageARecaler.ImRef.shape) * np.asarray(imageARecaler.sizeRef) / 2 #(translation max en mm)                              
                sizeSimplex = translateToFalseParameter(sizeSimplex,imageARecaler) - imageARecaler.offset                               
                simplex = createSimplex(x0,sizeSimplex,imageARecaler) #taille de 50mm
                
                res = minimize(criteriumToMinimize, x0, imageARecaler, method='Nelder-Mead', options={'initial_simplex' : simplex, 'xtol': 0.0005, 'disp': False})    
                #res = minimize(criteriumToMinimize, x0, imageARecaler, method='Powell',options={'xtol':0.001})
                
                res.x = translateToRealParameter(res.x,imageARecaler)
                tx[an1][an2][an3] = res.x[0]
                ty[an1][an2][an3] = res.x[1]
                tz[an1][an2][an3] = res.x[2]
    return tx,ty,tz,euler1,euler2,euler3
  
    

#les valeurs de tx, ty et tz ont été calculées pour différents angles définis par cyclique, minimal, maximal et M.
#on double ici la résolution (il faut faire attention car quand cyclique == true, le voisinage est différent)
#on rend également les valeurs des angles d euler pour pouvoir associer facilement ce qui va ensemble (cf fonctions computeOnFinerGrid)
def doubleTaille(tx,ty,tz,cyclique,minimal,maximal,M):
    
    #on agrandit le tableau pour gerer l aspect periodique
    if cyclique is True:     
        euler1 = np.linspace(0,360,num = 2*M, endpoint = False)
        euler2 = np.linspace(0,180,num = M, endpoint = False)
        euler3 = np.linspace(0,360,num = 2*M, endpoint = False)
        
        txx = np.zeros((tx.shape[0]+1,tx.shape[1]+1,tx.shape[2]+1))
        tyy = np.zeros((tx.shape[0]+1,tx.shape[1]+1,tx.shape[2]+1))
        tzz = np.zeros((tx.shape[0]+1,tx.shape[1]+1,tx.shape[2]+1))
        for an1 in range(txx.shape[0]):
            for an2 in range(txx.shape[1]):
                for an3 in range(txx.shape[2]):
                    txx[an1][an2][an3]=tx[an1%tx.shape[0]][an2%tx.shape[1]][an3%tx.shape[2]]
                    tyy[an1][an2][an3]=ty[an1%tx.shape[0]][an2%ty.shape[1]][an3%ty.shape[2]]
                    tzz[an1][an2][an3]=tz[an1%tx.shape[0]][an2%tz.shape[1]][an3%tz.shape[2]]
    
        i = np.arange(0,2*tx.shape[0])
        j = np.arange(0,2*tx.shape[1])
        k = np.arange(0,2*tx.shape[2])
    else:
        euler1 = np.linspace(minimal,maximal,num=2*M-1,endpoint=True)
        euler2 = np.linspace(minimal,maximal,num=2*M-1,endpoint=True)
        euler3 = np.linspace(minimal,maximal,num=2*M-1,endpoint=True)
        
        txx = tx
        tyy = ty
        tzz = tz
        i = np.arange(0,2*tx.shape[0]-1)
        j = np.arange(0,2*tx.shape[1]-1)
        k = np.arange(0,2*tx.shape[2]-1)
        
    iv,jv,kv = np.meshgrid(i,j,k,indexing='ij')
    iv = np.reshape(iv,(-1))
    jv = np.reshape(jv,(-1))
    kv = np.reshape(kv,(-1))
    
    #compute the coordinates in the input image
    pointset = np.zeros((3,iv.shape[0]))
    pointset[0,:] = iv/2.
    pointset[1,:] = jv/2.
    pointset[2,:] = kv/2.
    
    txx = map_coordinates(txx,[pointset[0,:],pointset[1,:],pointset[2,:]],order=1)
    tyy = map_coordinates(tyy,[pointset[0,:],pointset[1,:],pointset[2,:]],order=1)
    tzz = map_coordinates(tzz,[pointset[0,:],pointset[1,:],pointset[2,:]],order=1)
    if cyclique is True:
        txx = np.reshape(txx,(2*tx.shape[0],2*tx.shape[1],2*tx.shape[2]))
        tyy = np.reshape(tyy,(2*tx.shape[0],2*tx.shape[1],2*tx.shape[2]))
        tzz = np.reshape(tzz,(2*tx.shape[0],2*tx.shape[1],2*tx.shape[2]))
    else:
        txx = np.reshape(txx,(2*tx.shape[0]-1,2*tx.shape[1]-1,2*tx.shape[2]-1))
        tyy = np.reshape(tyy,(2*tx.shape[0]-1,2*tx.shape[1]-1,2*tx.shape[2]-1))
        tzz = np.reshape(tzz,(2*tx.shape[0]-1,2*tx.shape[1]-1,2*tx.shape[2]-1))
    return txx,tyy,tzz,euler1,euler2,euler3
    
     

         
   
#on calcule le critère sur une grille fine (tx et ty et tz) avec les angles euler1, euler2, et euler3
#imageArecaler donne les infos sur les images et sur la manière de calculer le critère.
def computeOnFinerGrid(tx,ty,tz,euler1,euler2,euler3,imageARecaler):
    criterium = np.zeros((euler1.shape[0],euler2.shape[0],euler3.shape[0]))
    eulerAngle = np.zeros(3)
    
    for an1 in range(euler1.shape[0]):           
        eulerAngle[0] = euler1[an1]
        for an2 in range(euler2.shape[0]):
            eulerAngle[1] = euler2[an2]
            for an3 in range(euler3.shape[0]):
                eulerAngle[2] = euler3[an3]
                Mrot = createRotationMatrix(eulerAngle)                
                imageARecaler.Mrot = Mrot
                x0 = [tx[an1][an2][an3],ty[an1][an2][an3],tz[an1][an2][an3]]
                x0 = np.asarray(x0)
                x0 = translateToFalseParameter(x0,imageARecaler)
                criterium[an1][an2][an3] = criteriumToMinimize(x0,imageARecaler)                     
    return criterium


#calcule des maxima locaux sur la grille du critère rendue par computeOnFinerGrid. La valeur de cyclique est importante car elle permet de définir le voisinage
# listeMaximum represente les coordonnées et listeCriterium les valeurs.
def searchMaxima(criterium,cyclique):
    listeMaximum=[]    
    listeCriterium = []
    if cyclique is True:
        for an1 in range(criterium.shape[0]):            
            for an2 in range(criterium.shape[1]):
                for an3 in range(criterium.shape[2]):
                    OK = 1
                    for i in [-1,0,1]: 
                        indice1 = (an1+i)%criterium.shape[0]
                        for j in [-1,0,1]:
                            indice2 = (an2+j)%criterium.shape[1]
                            for k in [-1,0,1]:
                                indice3 = (an3+k)%criterium.shape[2]
                                if  criterium[an1][an2][an3] >= criterium[indice1][indice2][indice3]:
                                    if indice1 != an1 or indice2 != an2 or indice3 != an3:
                                        OK=0  
                                    
                    if OK==1:
                        listeMaximum.append([an1,an2,an3])
                        listeCriterium.append(criterium[an1][an2][an3])        
                    
    else:
        for an1 in range(criterium.shape[0]):
            for an2 in range(criterium.shape[1]):
                for an3 in range(criterium.shape[2]):
                    OK=1
                    for i in [-1,0,1]: 
                        indice1 = an1+i                        
                        for j in [-1,0,1]:
                            indice2 = an2+j                            
                            for k in [-1,0,1]:
                                indice3 = an3+k
                                if indice1 >= 0 and indice2 >= 0 and indice3 >= 0 and indice1 < criterium.shape[0] and indice2 < criterium.shape[1] and indice3 < criterium.shape[2]:
                                    if  criterium[an1][an2][an3] >= criterium[indice1][indice2][indice3] :
                                        if indice1 != an1 or indice2 != an2 or indice3 != an3:
                                            OK = 0
                    if OK == 1:
                        listeMaximum.append([an1,an2,an3])
                        listeCriterium.append(criterium[an1][an2][an3])
    return listeMaximum,listeCriterium


def anepasutiliser(tx,nb):
    tx2 = np.zeros((2*nb,nb,2*nb))
    
    for an1 in range(nb):
        if an1 == nb-1:
            vd1 = 0
        else:
            vd1 = an1+1
        for an2 in range(nb/2):
            if an2 == nb/2-1:
                vd2 = 0
            else:
                vd2 = an2+1                                
            for an3 in range(nb):
                if an3 == nb-1:
                    vd3 = 0
                else:
                    vd3 = an3+1                                                                
                #points qui sont aux mêmes endroits
                tx2[2*an1][2*an2][2*an3] = tx[an1][an2][an3]    
                #points qui sont sur les memes aretes
                tx2[2*an1+1][2*an2][2*an3] = ( tx[an1][an2][an3] + tx[vd1][an2][an3] ) /2
                tx2[2*an1][2*an2+1][2*an3] = ( tx[an1][an2][an3] + tx[an1][vd2][an3] ) /2
                tx2[2*an1][2*an2][2*an3+1] = ( tx[an1][an2][an3] + tx[an1][an2][vd3] ) /2                  
                #points qui sont sur les memes surfaces
                tx2[2*an1+1][2*an2+1][2*an3] = ( tx[an1][an2][an3] + tx[vd1][an2][an3]  + tx[an1][vd2][an3] + tx[vd1][vd2][an3]) /4
                tx2[2*an1+1][2*an2][2*an3+1] = ( tx[an1][an2][an3] + tx[vd1][an2][an3]  + tx[an1][an2][vd3] + tx[vd1][an2][vd3]) /4
                tx2[2*an1][2*an2+1][2*an3+1] = ( tx[an1][an2][an3] + tx[an1][vd2][an3]  + tx[an1][an2][vd3] + tx[an1][vd2][vd3]) /4
                #points qui sont au milieu                   
                tx2[2*an1+1][2*an2+1][2*an3+1] = (tx[an1][an2][an3]+tx[vd1][an2][an3]+tx[an1][vd2][an3]+tx[vd1][vd2][an3]+tx[an1][an2][vd3]+tx[vd1][an2][vd3]+tx[an1][vd2][vd3] + tx[vd1][vd2][vd3]  ) /8
    return tx2


if __name__ == '__main__':  
    print "test de searcheMaxima"
    criterium = np.ones((5,5,5))
    criterium[2][2][1] = -10
    criterium[0][0][0] = -5
    criterium[0][4][0] = -15
    criterium[4][4][0] = -14                    
    criterium[0][0][4] = -12
    criterium[4][4][4] = -13             
             
    listeMaximum,listeCriterium = searchMaxima(criterium,True)
    print listeMaximum
    print listeCriterium

    listeMaximum,listeCriterium = searchMaxima(criterium,True)
    print listeMaximum
    print listeCriterium

    print "test de doubleTaille dans le cas cyclic = True"
    tx = np.random.randn(10,5,10)
    ty = np.random.randn(10,5,10)
    tz = np.random.randn(10,5,10)
    M = 10
    txx,tyy,tzz,euler1,euler2,euler3 = doubleTaille(tx,ty,tz,True,0,0,10)
    txx2 = anepasutiliser(tx,10)
    diff = np.abs(txx - txx2)
    print ("ca doit etre egal a 0 :" + str(np.max(diff)))
    