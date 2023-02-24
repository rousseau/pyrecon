#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 09:23:21 2022

@author: mercier
"""

import nibabel as nib
import matplotlib.pyplot as plt
import ipywidgets as widgets
import numpy as np
from registration import  commonSegment, computeCostBetweenAll2Dimages,costFromMatrix, global_optimisation, commonProfil, sliceProfil, updateCostBetweenAllImageAndOne,cost_fct, normalization, OptimisationThreshold 
from load import loadSlice, loadimages
from data_simulation import createMvt
from display import displayIntensityProfil, plotsegment, indexMse, indexGlobalMse, Histo, nbpointIndex, gaussianApproximation
import warnings
from input_argparser import InputArgparser
from os import listdir,getcwd,chdir
from os.path import isfile, join, splitext
import joblib
import napari
from tools import createVolumesFromAlist
from scipy import stats
warnings.filterwarnings("ignore")

class Viewer3D:
    """
    Viewer 3D is a class that alow to visualize the intersection between two orthogonal slices. You can select which image 
    you want to vizualise and which slice in each images. The intersection can be visualize with and without the mask
    The class take three diffents 3D images as parameters : it can be 'axial', 'sagital' and 'coronal' and their associated
    mask
    """
    def __init__(self,listSlice,data): 
        
    
        key=[p[0] for p in data] #name of element in data
        element=[p[1] for p in data] #element in data, results of the registration algorithm 
        
        
        #attributes initialisation
        
        listtmp = listSlice.copy()
        self.listSlice = listtmp
        self.listSlice = normalization(self.listSlice) #normalisation of the slices
        print('taille list:', len(self.listSlice))
        self.nbSlice=len(self.listSlice)
        
        
        
        self.images,mask = createVolumesFromAlist(self.listSlice.copy()) #create volumes corresponding to differents acquiered orientations
        self.imgsize=[] #list of the images size
       
        for i in range(len(self.images)):
            self.imgsize.append(len(self.images[i]))
        
        self.orientation1 = 0 #default orientation of the first choosen image for the visualisation
        self.orientation2 = 1 #default orientation of the second choosen image for the visualisation
        self.numImg1 = len(self.images[self.orientation1])//2 #default index of the first choosen slice in listSlice
        self.numImg2 = len(self.images[self.orientation2])//2 #default index of the second choosen slice in listSlice
        self.error = 'mse' #default observed error
        
        self.ErrorEvolution =element[key.index('ErrorEvolution')]  #Evolution of the mse on each iteration
        self.DiceEvolution =element[key.index('DiceEvolution')]   #Evolution of the dice on each iteration
        
        self.nbit = len(self.ErrorEvolution) #total number of iteration
   
        self.EvolutionGridError = element[key.index('EvolutionGridError')] #Evolution of the error grid on each iteration
        self.EvolutionGridNbpoint = element[key.index('EvolutionGridNbpoint')] #Evolution of the nbpoint grid on each iteration
        self.EvolutionGridInter= element[key.index('EvolutionGridInter')] #Evolution of the intersection grid on each iteration 
        self.EvolutionGridUnion= element[key.index('EvolutionGridUnion')] #Evolution of the union grid on each iteration
        self.EvolutionParameters = element[key.index('EvolutionParameters')] #Evolution of the parameters on each iteration
        
        EvolutionTransfo = element[key.index('EvolutionTransfo')] #Evolution of the rigid transformation on each iteration
        
        self.Transfo=EvolutionTransfo[self.nbit-1,:,:,:] #rigid transformation on the last iteration

        self.iImg1 = 0
        self.iImg2 = 0
        
        self.Nlast=self.EvolutionGridError[self.nbit-1,:,:] #gridError on the last iteration
        self.Dlast=self.EvolutionGridNbpoint[self.nbit-1,:,:] #gridNbpoint on the last iteration
        self.lastMse=self.Nlast/self.Dlast #Mse on the last iteration
        self.valMax=np.max(self.lastMse[np.where(~np.isnan(self.lastMse))]) #max value of mse on the last iteration ??
        
        image1=self.listSlice[self.numImg1].get_slice().get_fdata() #default image1
        image2=self.listSlice[self.numImg2].get_slice().get_fdata() #default image2
        affine1=self.Transfo[self.numImg1,:,:].copy() #affine of image1
        affine2=self.Transfo[self.numImg2,:,:].copy() #affine of image2
        
        #hack to get very thin slices (by modifying the slice thickness)
        header1 = self.listSlice[self.numImg1].get_slice().header
        header2 = self.listSlice[self.numImg2].get_slice().header        
        affine1[:3, :3] *= (1,1,0.1/header1["pixdim"][3])
        affine2[:3, :3] *= (1,1,0.1/header2["pixdim"][3])
        
        error1=sum(self.EvolutionGridError[self.nbit-1,:,self.numImg1])+sum(self.EvolutionGridError[self.nbit-1,self.numImg1,:])
        nbpoint1=sum(self.EvolutionGridNbpoint[self.nbit-1,:,self.numImg1])+sum(self.EvolutionGridNbpoint[self.nbit-1,self.numImg1,:])
        MSE1=error1/nbpoint1 #MSE of image1
        name1='1 : Mse : %f, Slice : %d' %(MSE1,self.numImg1)
        
        error2=sum(self.EvolutionGridError[self.nbit-1,:,self.numImg2])+sum(self.EvolutionGridError[self.nbit-1,self.numImg2,:])
        nbpoint2=sum(self.EvolutionGridNbpoint[self.nbit-1,:,self.numImg2])+sum(self.EvolutionGridNbpoint[self.nbit-1,self.numImg2,:])
        MSE2=error2/nbpoint2 #MSE of image2
        name2='2 : Mse : %f, Slice : %d' %(MSE2,self.numImg2)
    
        self.previousname1=name1
        self.previousname2=name2
        
        #initialisation of napari viewer with the default values
        self.viewer = napari.view_image(image1,affine=affine1,name=self.previousname2,blending='opaque',rendering='translucent',interpolation='nearest',opacity=1,ndisplay=3,visible=True)
        self.viewer.add_image(image2,affine=affine2,name=self.previousname1,blending='opaque',rendering='translucent',interpolation='nearest',opacity=1,visible=True)
        napari.run()
        
        self.RejectedSlices = element[key.index('RejectedSlices')]
        print(self.RejectedSlices)
        
        print("Cost Before Registration : ", self.ErrorEvolution[0])
        print("Cost After Registration : ", self.ErrorEvolution[self.nbit-1])
        
        
        #display Evolution of the error on each iteration
        plt.plot(self.ErrorEvolution)
        plt.title('Evolution of the global cost over %d iteration' %(self.nbit-1))
        plt.show()
        
        #display the mse grid on each iteration
        ipy_error = widgets.interact_manual(self.ErrorDisplay,
            nit=widgets.IntSlider(
            value=self.nbit-1,
            min=0,
            max=self.nbit-1,
            description='Iteration',
            disabled=False,
            button_style='', 
            tooltip='Description',
            icon='check',
            ))
        
        ipy_error.widget.children[1].description = 'Image Error' #change name of 'run_interact' button
        
        #display parameters for each slices on each iteration
        ipy_parameters = widgets.interact_manual(self.ErrorParametersDisplay,
            nit=widgets.IntSlider(
            value=self.nbit-1,
            min=0,
            max=self.nbit-1,
            description='Iteration',
            disabled=False,
            button_style='', 
            tooltip='Description',
            icon='check' 
        ))
        
        ipy_parameters.widget.children[1].description = 'Parameters' #change name of 'run_interact' button
        
        
        #select the two images you are interested in
        widgets.interact(
        self.chooseImage12,orientation1 = widgets.RadioButtons(
        options=range(len(self.images)),
        value=0,
        description='Image 1:',
        disabled=False,
        ),
        orientation2  = widgets.RadioButtons(
        options=range(len(self.images)),
        value=1,
        description='Image 2:',
        disabled=False,
        ))
        
        #use to start napari if the windows is closed
        widgets.interact(self.go_napari,
        go=widgets.ToggleButton(
        value=False,
        description='Start Napari',
        disabled=False,
        button_style='', 
        tooltip='Description',
        icon='check'
        ))

        
    """
    Method of the class Viewer3D : choose two images of interest
    """
    def chooseImage12(self,orientation1,orientation2):
        
        self.orientation1 = orientation1
        self.orientation2 = orientation2
        
        data_img1=[];data_img2=[]
        for i in range(len(self.images)):
            if self.orientation1==i: 
                data_img1=self.images[i]
            if self.orientation2==i:
                data_img2=self.images[i]
       
                
        nbImage1 = len(data_img1)-1
        nbImage2 = len(data_img2)-1
        
        #choose the slices you are interested in
        widgets.interact(
        self.chooseSlice12,numImg1 = widgets.IntSlider(
        min=0,
        max=nbImage1,
        value=np.ceil(nbImage1)/2,
        description='Slice in 1:',
        disabled=False,
        ),

        numImg2 = widgets.IntSlider(
        min=0,
        max=nbImage2,
        value=np.ceil(nbImage2)/2,
        description='Slice in 2:',
        disabled=False,
        )) 
    
    
    """
    Method of the class Viewer3D, choose 2 slices of interest
    """
    def chooseSlice12(self,numImg1,numImg2):
        
        self.iImg1 = numImg1
        self.iImg2 = numImg2
        
        for i in range(len(self.images)):
            if self.orientation1==i:
                n=i-1; sum=0;
                while n>=0:
                    sum=sum+self.imgsize[n]
                    n=n-1
                self.numImg1 = numImg1 + sum
            if self.orientation2==i:
                n=i-1; sum=0;
                while n>=0:
                    sum=sum+self.imgsize[n]
                    n=n-1
                self.numImg2 = numImg2 + sum
            
       
            if self.orientation1==i:
                n=i-1; sum=0;
                while n>=0:
                    sum=sum+self.imgsize[n]
                    n=n-1
                self.numImg1 = numImg1 + sum 
        display('img1',self.numImg1,'img2',self.numImg2)
        display('error',self.Nlast[max(self.numImg1,self.numImg2),min(self.numImg1,self.numImg2)])
        display('nbpoint',self.Dlast[max(self.numImg1,self.numImg2),min(self.numImg1,self.numImg2)])
        display('MSE',self.Nlast[max(self.numImg1,self.numImg2),min(self.numImg1,self.numImg2)]/self.Dlast[max(self.numImg1,self.numImg2),min(self.numImg1,self.numImg2)])
        
        slice1=self.listSlice[self.numImg1]
        slice2=self.listSlice[self.numImg2]
        self.visu_withnapari(slice1,slice2,self.numImg1,self.numImg2)
            
        #display lines of intersection intersection on images
        widgets.interact(self.DisplayProfil,
        nit=widgets.IntSlider(
        value=self.nbit-1,
        min=0,
        max=self.nbit-1,
        description='Iteration',
        disabled=False,
        button_style='',
        tooltip='Description',
        icon='check'
        )) 
        
        
        #display mse for each slices
        ipy_mse = widgets.interact_manual(self.DisplayAllErrors,error='mse',
        nit=widgets.IntSlider(
        value=self.nbit-1,
        min=0,
        max=self.nbit-1,
        description='Iteration',
        disabled=False,
        button_style='', 
        tooltip='Description',
        icon='check' 
        ))
        
        ipy_mse.widget.children[2].description = 'Mse' #change name of 'run_interact' button
        
        #display dice for each slice
        ipy_dice = widgets.interact_manual(self.DisplayAllErrors,error='dice',
        nit=widgets.IntSlider(
        value=self.nbit-1,
        min=0,
        max=self.nbit-1,
        description='Iteration',
        disabled=False,
        button_style='', 
        tooltip='Description',
        icon='check' 
        ))
        
        ipy_dice.widget.children[2].description = 'Dice' #change name of 'run_interaction' button
            
    
    """
    Method of the class Viewer3D, display the evolution of the error in the choosen iteration
    """
    def ErrorDisplay(self,nit):
            
            print(self.RejectedSlices)
            rejectedSlices = self.RejectedSlices
            
            fig = plt.figure(figsize=(8, 8))
            cx  = plt.subplot()
            N = self.EvolutionGridError[nit,:,:].copy() 
            D = self.EvolutionGridNbpoint[nit,:,:].copy()
            MSE = N/D
            
            for i_slice in range(len(self.listSlice)):
                
                slicei = self.listSlice[i_slice]
                orientation =  slicei.get_orientation()
                index_slice = slicei.get_index_slice()
                info_slicei = (orientation,index_slice)
                if info_slicei in rejectedSlices :
                    for i_slice2 in range(len(self.listSlice)):
                        slice2_orientation = self.listSlice[i_slice2].get_orientation()
                        if (slice2_orientation != orientation) :
                            if (i_slice>i_slice2) :
                                N[i_slice,i_slice2]=100 
                                D[i_slice,i_slice2]=1 
                            else : 
                                N[i_slice2,i_slice]=100
                                D[i_slice2,i_slice]=1
            
            
            
            im = cx.imshow(N/D,vmin=0,vmax=self.valMax)
            cbar = fig.colorbar(im,ax=cx)
            
            for i in range(1,len(self.images)):
                n=i-1; sum=0;
                while n>=0:
                    sum=sum+self.imgsize[n]
                    n=n-1
                line=sum
                cx.hlines(y=line,xmin=0,xmax=self.nbSlice-1,lw=2,color='r')
                cx.vlines(x=line,ymin=0,ymax=self.nbSlice-1,lw=2,color='r')

            plt.show()
            
            
            
            
    """
    Method of the class Viwer3D, display the error in parameters estimation in case of a simulation
    """
    def ErrorParametersDisplay(self,nit):
        
        NBPOINT_GLOB, ERROR_GLOB = indexGlobalMse(self.Nlast,self.Dlast) 
        lastmse=ERROR_GLOB/NBPOINT_GLOB
        a=np.where(np.isnan(lastmse))
        lastmse[a]=0
        nbpoint, error=indexGlobalMse(self.EvolutionGridError[nit,:,:],self.EvolutionGridNbpoint[nit,:,:])
        cmse=error/nbpoint
        b=np.where(np.isnan(cmse))
        fig = plt.figure(figsize=(10, 10))
        im=plt.scatter(range(self.nbSlice),self.EvolutionParameters[self.nbit-1,:,1],marker='.',c=lastmse)
        ax1=plt.subplot(3,2,1)
        ax1.scatter(range(self.nbSlice),self.EvolutionParameters[nit,:,0],marker='.',c=cmse) #[nit,0,:]
        ax1.set_title("Angle along x")
        ax1.set_xlabel("Number of slice")
        cbar = fig.colorbar(im,ax=ax1)
        ax2=plt.subplot(3,2,2)
        ax2.scatter(range(self.nbSlice),self.EvolutionParameters[nit,:,1],marker='.',c=cmse) #[nit,1,:]
        ax2.set_title("Angle along y")
        ax2.set_xlabel("Number of slice")
        cbar=fig.colorbar(im,ax=ax2)
        ax3=plt.subplot(3,2,3)
        ax3.scatter(range(self.nbSlice),self.EvolutionParameters[nit,:,2],marker='.',c=cmse) #[nit,2,:]
        ax3.set_title("Angle along z")
        ax3.set_xlabel("Number of slice")
        cbar = fig.colorbar(im,ax=ax3)
        ax4=plt.subplot(3,2,4)
        ax4.scatter(range(self.nbSlice),self.EvolutionParameters[nit,:,3],marker='.',c=cmse) #[nit,3,:]
        ax4.set_title("Translation along x")
        ax4.set_xlabel("Number of slice")
        cbar = fig.colorbar(im,ax=ax4)
        ax5=plt.subplot(3,2,5)
        ax5.scatter(range(self.nbSlice),self.EvolutionParameters[nit,:,4],marker='.',c=cmse) #[nit,4,:]
        ax5.set_title("Translation along y")
        ax5.set_xlabel("Number of slice")
        cbar = fig.colorbar(im,ax=ax5)
        ax6=plt.subplot(3,2,6)
        ax6.scatter(range(self.nbSlice),self.EvolutionParameters[nit,:,5],marker='.',c=cmse) #[nit,5,:]
        ax6.set_title("Translation along z")
        ax6.set_xlabel("Number of slice")
        cbar=fig.colorbar(im,ax=ax6)
        plt.title('Visualisation of the parameters for each slice at iteration %d' %(nit))
        fig.tight_layout()
        plt.show()
        
    """
    Method of the class Viewer3D, display the evolution of the intersection between two slices
    """       
    def DisplayProfil(self,nit):
            
            listIteration = self.listSlice.copy()

            parameters = self.EvolutionParameters[nit,:,:].copy()
         
            i_slice = 0
            for s in listIteration:
                x = parameters[i_slice,:]
                s.set_parameters(x)
                i_slice=i_slice+1
        
            slice_img1 = listIteration[self.numImg1]
            slice_img2 = listIteration[self.numImg2]
            
            image1=slice_img1.get_slice().get_fdata();image2=slice_img2.get_slice().get_fdata()
            M1=slice_img1.get_transfo();M2=slice_img2.get_transfo();res=min(slice_img1.get_slice().header.get_zooms())
            pointImg1,pointImg2,nbpoint,ok=commonSegment(image1,M1,image2,M2,res)
            nbpoint=np.int32(nbpoint[0,0]);ok=np.int32(ok[0,0])

            if ok>0:
                val1,index1,nbpointSlice1=sliceProfil(slice_img1, pointImg1, nbpoint)
                val2,index2,nbpointSlice2=sliceProfil(slice_img2, pointImg2, nbpoint)
                commonVal1,commonVal2,index=commonProfil(val1, index1, val2, index2, nbpoint)
                displayIntensityProfil(commonVal1,index1,commonVal2,index2,index)

            
                #display the intersection segment on the image, with ot without the mask
                fig=plt.figure(figsize=(10, 10))
                ax1=plt.subplot(1,2,1)
                ax2=plt.subplot(1,2,2)


                title1 = 'Intersection segment for image %s with mask,slice %d' %(self.orientation1,self.iImg1)
                title2 = 'Intersection segment for image %s with mask,slice %d' %(self.orientation2,self.iImg2)
                plotsegment(slice_img1,pointImg1,ok,nbpoint,ax1,title1,mask=slice_img1.get_mask(),index=index,nbpointSlice=nbpointSlice1)
                plotsegment(slice_img2,pointImg2,ok,nbpoint,ax2,title2,mask=slice_img2.get_mask(),index=index,nbpointSlice=nbpointSlice2) 
                fig.tight_layout()
                plt.show()
                
            
            self.visu_withnapari(slice_img1,slice_img2,self.numImg1,self.numImg2)
            
    """
    Method of the class Viewer3D that gives the error and the number of point for each slices
    """
    def DisplayAllErrors(self,error,nit):
            
            #compute the error : case error is mse
            self.error=error
            
            N=len(self.listSlice)
            
            
            
            if self.error=='mse':
                
                Num=self.EvolutionGridError[nit,:,:]
                Denum=self.EvolutionGridNbpoint[nit,:,:]
                NumImg1Row=self.EvolutionGridError[nit,self.numImg1,:];NumImg1Col=self.EvolutionGridError[nit,:,self.numImg1]
                DenumImg1Row=self.EvolutionGridNbpoint[nit,self.numImg1,:];DenumImg1Col=self.EvolutionGridNbpoint[nit,:,self.numImg1]
                NumImg2Row=self.EvolutionGridError[nit,self.numImg2,:];NumImg2Col=self.EvolutionGridError[nit,:,self.numImg2]
                DenumImg2Row=self.EvolutionGridNbpoint[nit,self.numImg2,:];DenumImg2Col=self.EvolutionGridNbpoint[nit,:,self.numImg2]
                ylabel_ax1='MSE'
                ylabel_ax2='Nbpoint'
                ax1_title_1='MSE between slice %d in image %s and its orthogonal slices' %(self.iImg1,self.orientation1)
                ax3_title_1='Histogram of the MSE between slice %d in image %s and its orthogonal slices' %(self.iImg1,self.orientation1)
                ax1_title_2='MSE between slice %d in image %s and its orthogonal slices' %(self.iImg2,self.orientation2)
                ax3_title_2='Histogram of the MSE between slice %d in image %s and its orthogonal slices' %(self.iImg2,self.orientation2)
                ax1_title_GLOB='Global MSE between each slices'
                ax3_title_GLOB='Histogram of the Global MSE'
                lab='MSE'
                Threshold = OptimisationThreshold(Num,Denum)
                x_gauss,y_gauss = gaussianApproximation(Num,Denum,N)
                
            if self.error=='dice':
                
                Num=self.EvolutionGridInter[nit,:,:]
                Denum=self.EvolutionGridUnion[nit,:,:]
                NumImg1Row=self.EvolutionGridInter[nit,self.numImg1,:];NumImg1Col=self.EvolutionGridInter[nit,:,self.numImg1]
                DenumImg1Row=self.EvolutionGridUnion[nit,self.numImg1,:];DenumImg1Col=self.EvolutionGridUnion[nit,:,self.numImg1]
                NumImg2Row=self.EvolutionGridInter[nit,self.numImg2,:];NumImg2Col=self.EvolutionGridInter[nit,:,self.numImg2]
                DenumImg2Row=self.EvolutionGridUnion[nit,:,self.numImg2];DenumImg2Col=self.EvolutionGridUnion[nit,:,self.numImg2]
                ylabel_ax1='Dice'
                ylabel_ax2='Union'
                ax1_title_1='DICE between slice %d in image %s and its orthogonal slices' %(self.iImg1,self.orientation1)
                ax3_title_1='Histogram of the DICE between slice %d in image %s and its orthogonal slices' %(self.iImg1,self.orientation1)
                ax1_title_2='DICE between slice %d in image %s and its orthogonal slices' %(self.iImg2,self.orientation2)
                ax3_title_2='Histogram of the DICE between slice %d in image %s and its orthogonal slices' %(self.iImg2,self.orientation2)
                ax1_title_GLOB='Global DICE between each slices'
                ax3_title_GLOB='Histogram of the Global DICE'
                display("Dice", np.where((Denum==0)*(Num!=0)))
                display("check :", (Num/Denum)[30,7])
                display("Num :", Num[30,7], "Denum :", Denum[30,7])
                lab='DICE'
            
               
            indexError1, indexNbpoint1=indexMse(Num,Denum,self.numImg1) 
            indexError2, indexNbpoint2=indexMse(Num,Denum,self.numImg2)
            maskproportion = nbpointIndex(self.listSlice)

                
            size_indexError1=indexError1.shape[0]
            size_indexError2=indexError2.shape[0]
            indexMse1=np.zeros(size_indexError1)
            indexMse2=np.zeros(size_indexError2)

            for i in range(size_indexError1):
                indexMse1[i]=indexError1[i]/indexNbpoint1[i]
            for i in range(size_indexError2):
                indexMse2[i]=indexError2[i]/indexNbpoint2[i]

            maxIndex=max(self.numImg1,self.numImg2)
            minIndex=min(self.numImg1,self.numImg2)
            error=self.EvolutionGridError[nit,maxIndex,minIndex]
            commonPoint=self.EvolutionGridNbpoint[nit,maxIndex,minIndex]
                

            MSEloc=error/commonPoint
                

            sumError1=sum(NumImg1Row)  + sum(NumImg1Col)
            sumNbpoint1=sum(DenumImg1Row) + sum(DenumImg1Col)
            MSEGlobImg1=sumError1/sumNbpoint1

            sumError2=sum(NumImg2Row)  + sum(NumImg2Col)
            sumNbpoint2=sum(DenumImg2Row) + sum(DenumImg2Col)
            MSEGlobImg2=sumError2/sumNbpoint2
                

            NBPOINT_GLOB, ERROR_GLOB = indexGlobalMse(Num,Denum)
            
            size_error=NBPOINT_GLOB.shape[0]
            MSE_GLOB=np.zeros(size_error)
            
            

            for i in range(size_error):
                MSE_GLOB[i] = ERROR_GLOB[i]/NBPOINT_GLOB[i]
                
            
             
            MSE_GLOB = MSE_GLOB[np.where(~np.isnan(MSE_GLOB)*~np.isinf(MSE_GLOB))]
            mu = np.mean(MSE_GLOB)
            sigma = np.sqrt(sum((MSE_GLOB-mu)**2)/mu)
            h = (3.49*sigma)/(N**(1/3)) 
            display(h)
            display(min(MSE_GLOB))
            display(max(MSE_GLOB))
            bins=np.linspace(min(MSE_GLOB),max(MSE_GLOB),int(h))
            
            #compute the threshold value, used in the second part of the registation process
            if self.error=='mse':
                bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5]
            else : 
                bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] 
           
            if  np.any(~np.isnan(indexMse1)):
                fig=plt.figure(figsize=(30, 8))
                ax1=plt.subplot(1,2,1)
                ax1.set_ylabel(ylabel_ax1)
                plotmse= ax1.scatter(range(self.nbSlice),indexMse1,label=lab)
                ax2 = ax1.twinx()
                plotnbpoint= ax2.scatter(range(self.nbSlice),maskproportion,c='orange',label='mask_size')
                plt.legend(handles=[plotmse,plotnbpoint])
                ax2.set_ylabel(ylabel_ax2)
                ax1.set_title(ax1_title_1)
                ax1.set_xlabel('Slices')
                fig.tight_layout()
                indexHistoMse1=(indexMse1[np.where(~np.isnan(indexMse1)*~np.isinf(indexMse1))])
                ax3=plt.subplot(1,2,2)
                histoMse1, bin_edges=np.histogram(indexHistoMse1,bins)
                ax3.bar(bin_edges[:-1], histoMse1/sum(histoMse1),align='edge',edgecolor='black',width=0.1)
                ax3.set_title(ax3_title_1)
                fig.tight_layout()
                plt.show()

            if  np.any(~np.isnan(indexMse2)):   
                fig=plt.figure(figsize=(30, 8))
                ax1=plt.subplot(1,2,1)
                ax1.set_ylabel(ylabel_ax1)
                plotmse = ax1.scatter(range(self.nbSlice),indexMse2,label=lab)
                ax2 = ax1.twinx()
                plotnbpoint = ax2.scatter(range(self.nbSlice),maskproportion,c='orange',label='size_mask')
                plt.legend(handles =[plotmse,plotnbpoint])
                ax2.set_ylabel(ylabel_ax2)
                plt.title(ax1_title_2)
                fig.tight_layout()
                indexHistoMse2=(indexMse2[np.where(~np.isnan(indexMse2)*~np.isinf(indexMse2))])
                ax3=plt.subplot(1,2,2)
                histoMse2, bin_edges=np.histogram(indexHistoMse2,bins)
                ax3.bar(bin_edges[:-1], histoMse2/sum(histoMse2),align='edge',edgecolor='black',width=0.1)
                ax3.set_title(ax3_title_2)
                fig.tight_layout()
                plt.show()

            if  np.any(~np.isnan(MSE_GLOB)):
                fig=plt.figure(figsize=(30, 8))
                ax1=plt.subplot(1,2,1)
                ax1.set_ylabel(ylabel_ax1)
                plotmse = ax1.scatter(range(self.nbSlice),MSE_GLOB,label=lab)
                ax2 = ax1.twinx()
                pltnbpoint = ax2.scatter(range(self.nbSlice),maskproportion,c='orange',label='size_mask')
                plt.legend(handles=[plotmse,plotnbpoint])
                ax2.set_ylabel(ylabel_ax2)
                plt.title(ax1_title_GLOB)
                fig.tight_layout()
                ax3=plt.subplot(1,2,2)
                HistoMSE_GLOB=MSE_GLOB[~np.isnan(MSE_GLOB)]
                histoGlobal, bin_edges=np.histogram(HistoMSE_GLOB,bins)
                print(bin_edges,histoGlobal,histoGlobal/sum(histoGlobal))
                ax3.bar(bin_edges[:-1], histoGlobal/sum(histoGlobal),align='edge',edgecolor='black',width=0.1)
                if self.error == 'mse':
                    fit_alpha, fit_loc, fit_scale = stats.gamma.fit(HistoMSE_GLOB)
                    print(fit_alpha, fit_loc, fit_scale)
                    x = np.linspace(min(HistoMSE_GLOB),max(HistoMSE_GLOB),100)
                    pdf = stats.gamma.pdf(bin_edges[:-1],fit_alpha,fit_loc,fit_scale)
                    print(pdf)
                    ax3.plot(bin_edges[:-1],pdf/sum(pdf),color='coral')
                    #ax3.axvline(x=Threshold2,color='blue')
                    

                ax3.set_title(ax3_title_GLOB)
                fig.tight_layout()
                plt.show()
                    
    """
    Method of the class Viewer3D, allow to visualize two slices in real word coordinate system with napari
    """
    def visu_withnapari(self,slice1,slice2,numImg1,numImg2):
        
        display(self.viewer.layers[0])
        self.viewer.layers.remove(self.viewer.layers[0])
        display(self.viewer.layers[0])
        self.viewer.layers.remove(self.viewer.layers[0])
        
        #image1=self.listSlice[numImg1].get_slice().get_fdata()
        #image2=self.listSlice[numImg2].get_slice().get_fdata()
        #affine1=self.Transfo[numImg1,:,:].copy()
        #affine2=self.Transfo[numImg2,:,:].copy()
        image1=slice1.get_slice().get_fdata()
        image1[np.where(np.isnan(image1))]=0
        image2=slice2.get_slice().get_fdata()
        image2[np.where(np.isnan(image2))]=0
        affine1 = slice1.get_transfo()
        affine2 = slice2.get_transfo()
        
        header1 = self.listSlice[self.numImg1].get_slice().header
        header2 = self.listSlice[self.numImg2].get_slice().header        
        affine1[:3, :3] *= (1,1,0.1/header1["pixdim"][3])
        affine2[:3, :3] *= (1,1,0.1/header2["pixdim"][3])

        
        error1=sum(self.EvolutionGridError[self.nbit-1,:,self.numImg1])+sum(self.EvolutionGridError[self.nbit-1,self.numImg1,:])
        nbpoint1=sum(self.EvolutionGridNbpoint[self.nbit-1,:,numImg1])+sum(self.EvolutionGridNbpoint[self.nbit-1,self.numImg1,:])
        MSE1=error1/nbpoint1
        name1='1 : Mse : %f, Slice : %d' %(MSE1,self.numImg1)
        print('name1 :',name1)
        
        error2=sum(self.EvolutionGridError[self.nbit-1,:,self.numImg2])+sum(self.EvolutionGridError[self.nbit-1,self.numImg2,:])
        nbpoint2=sum(self.EvolutionGridNbpoint[self.nbit-1,:,numImg2])+sum(self.EvolutionGridNbpoint[self.nbit-1,self.numImg2,:])
        MSE2=error2/nbpoint2
        name2='2 : Mse : %f, Slice : %d' %(MSE2,self.numImg2)
        print('name2 :',name2)
        
        self.viewer.add_image(image1,affine=affine1,name=name1,blending='opaque',rendering='translucent',interpolation='nearest',opacity=1,visible=True)
        self.viewer.add_image(image2,affine=affine2,name=name2,blending='opaque',rendering='translucent',interpolation='nearest',opacity=1,visible=True)
        
        self.previousname1=name1
        self.previousname2=name2
    
    """
    Method of the class Viewer3D. Used to start napari when the window is closed
    """
    def go_napari(self,go):
        
        if go==True:
            
            image1=self.listSlice[self.numImg1].get_slice().get_fdata()
            image2=self.listSlice[self.numImg2].get_slice().get_fdata()

            affine1=self.Transfo[self.numImg1,:,:].copy()
            affine2=self.Transfo[self.numImg2,:,:].copy()
            
            #hack to get very thin slices (by modifying the slice thickness)
            header1 = self.listSlice[self.numImg1].get_slice().header
            header2 = self.listSlice[self.numImg2].get_slice().header        
            affine1[:3, :3] *= (1,1,0.1/header1["pixdim"][3])
            affine2[:3, :3] *= (1,1,0.1/header2["pixdim"][3])
        
            error1=sum(self.EvolutionGridError[self.nbit-1,:,self.numImg1])+sum(self.EvolutionGridError[self.nbit-1,self.numImg1,:])
            nbpoint1=sum(self.EvolutionGridNbpoint[self.nbit-1,:,self.numImg1])+sum(self.EvolutionGridNbpoint[self.nbit-1,self.numImg1,:])
            MSE1=error1/nbpoint1
            name1='1 : Mse : %f, Slice : %d' %(MSE1,self.numImg1)
        
            error2=sum(self.EvolutionGridError[self.nbit-1,:,self.numImg2])+sum(self.EvolutionGridError[self.nbit-1,self.numImg2,:])
            nbpoint2=sum(self.EvolutionGridNbpoint[self.nbit-1,:,self.numImg2])+sum(self.EvolutionGridNbpoint[self.nbit-1,self.numImg2,:])
            MSE2=error2/nbpoint2
            name2='2 : Mse : %f, Slice : %d' %(MSE2,self.numImg2)
    
            self.previousname1=name1
            self.previousname2=name2
        
            self.viewer = napari.view_image(image1,affine=affine1,name=self.previousname2,blending='opaque',rendering='translucent',interpolation='nearest',opacity=1,ndisplay=3,visible=True)
            self.viewer.add_image(image2,affine=affine2,name=self.previousname1,blending='opaque',rendering='translucent',interpolation='nearest',opacity=1,visible=True)
            napari.run()

def choose_joblib(joblib_name): 
    
    res=joblib.load(open(joblib_name,'rb'))
    key=[p[0] for p in res]
    element=[p[1] for p in res]
    listSlice=element[key.index('listSlice')]     
    Viewer3D(listSlice,res)
    return joblib_name