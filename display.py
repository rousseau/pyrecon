#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 10:09:43 2021

@author: mercier
"""
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from skimage.color import gray2rgb


def show_slice(slices):  # definition de la fonction show_slice qui prend en paramètre une image
    # ""Function di display row of image slices""
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        # affiche l'image en niveau de gris (noir pour la valeur minimale et blanche pour la valeur maximale)
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


def plotsegment(slice, pointImg, ok, nbpoint,title=' ',mask=np.nan,index=np.nan):

    if ok < 1:
        return 0
    
    pointInterpol = np.zeros((2, nbpoint))
    pointInterpol[0, :] = np.linspace(pointImg[0, 0], pointImg[0, 1], nbpoint)
    pointInterpol[1, :] = np.linspace(pointImg[1, 0], pointImg[1, 1], nbpoint)
    sliceimage = slice.get_slice()
    
    if  ~np.isnan(mask).all():
        print('test')
        img_with_mask = nib.Nifti1Image((slice.get_slice().get_fdata()) * mask, slice.get_slice().affine)
        sliceimage = img_with_mask
        pointInterpol = pointInterpol[:,index]
        nbpoint = pointInterpol.shape[1]
   
    plt.figure()
    plt.imshow(sliceimage.get_fdata().T, cmap="gray", origin="lower")
   
    for i in range(nbpoint):
        plt.plot(pointInterpol[0, i],pointInterpol[1, i], 'ro',markersize=1,label='%nbpoint')
        print(pointInterpol[:, i])
    
    plt.figtext(0,0,"nbpoint : %d" %(nbpoint))
    plt.title(title)
    
def plotin3DSpace(slice):
    
    ax = plt.gca(projection='3d')
    slice_image = slice.get_slice().get_fdata()
    size = np.shape(slice_image)[0]
    x = np.arange(0,size)
    print(x)
    xx, yy = np.meshgrid(x,x,indexing='ij')
    xx = np.reshape(xx,(-1))
    yy = np.reshape(yy,(-1))
    img_coordinates = np.zeros((4,size*size),dtype=int)
    img_coordinates[0,:] = xx
    img_coordinates[1,:] = yy
    img_coordinates[3,:] = np.ones((1,size*size))
    slicedata = slice_image[img_coordinates[0,:],img_coordinates[1,:]]
    M = slice.get_transfo() 

    slice3D = M @ img_coordinates
    
    #step of reconstruction
    minX = min(slice3D[0,:])
    maxX = max(slice3D[0,:])
    minY = min(slice3D[1,:])
    maxY = max(slice3D[1,:])
    minZ = min(slice3D[2,:])
    maxZ = max(slice3D[3,:])
    
    resolution = min(slice.get_slice().header.get_zooms())
    
    nbpointX = int(np.ceil((maxX-minX)/resolution) + 1)
    nbpointY = int(np.ceil((maxY-minY)/resolution) + 1)
    nbpointZ = int(np.ceil((maxZ-minZ)/resolution) + 1)

    
    i = np.linspace(minX,maxX,nbpointX)   
    j = np.linspace(minY,maxY,nbpointY)
    z = np.linspace(minZ,maxZ,nbpointZ)
    print(z)
    
    grid = np.meshgrid(i,j,z,indexing='ij')
    ii = np.reshape(grid[0],(-1))
    jj = np.reshape(grid[1],(-1))
    zz = np.reshape(grid[2],(-1))
    
    world_coordinates = np.zeros((np.shape(ii)[0],3))
    world_coordinates[:,0] = ii
    world_coordinates[:,1] = jj
    world_coordinates[:,2] = zz
    
    
    print(slice3D.T[:,0:3])
    print(img_coordinates)
    print(np.shape(slicedata))
    slice3D = np.transpose(slice3D)
    print(world_coordinates)
    imginterpol = interpolate.griddata(slice3D[:,0:3],slicedata,world_coordinates,method='linear')

   
    imginterpol = imginterpol.reshape((nbpointX,nbpointY,nbpointZ))
    Z = np.array([zz])
    print(Z)
    print(gray2rgb(imginterpol).shape)
    imginterpol = gray2rgb(imginterpol)
    ax.plot_surface(ii,jj,Z,rstride=5,cstride=5,cmap='gray',facecolor=imginterpol)
    #ax.plot_trisurf(ii,jj,zz,imginterpol,cmap='gray')
    #ax.imshow(imginterpol, cmap="gray", origin="lower",aspect="auto")
    #ax.plot_trisurf(world_coordinates[:,0], world_coordinates[:,1],world_coordinates[:,2],imginterpol)
    #plt.show()
    return slicedata,img_coordinates,world_coordinates
    
    
    
    
    