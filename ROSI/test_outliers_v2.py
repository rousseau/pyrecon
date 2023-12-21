#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:54:23 2023

@author: mercier
"""

import joblib
#import outliers_detection_intersection as out
#import data_simulation as ds
import numpy as np
import nibabel as nib
#import load
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random
import pandas as pd
from rosi.registration.outliers_detection.multi_start import  removeBadSlice
from rosi.registration.intersection import compute_cost_matrix
from rosi.registration.load import convert2Slices
from rosi.registration.outliers_detection.feature import data_to_classifier,update_features
from rosi.registration.outliers_detection.outliers import sliceFeature
#import registration as re
import matplotlib.pyplot as plt
import os.path

from rosi.simulation.validation import same_order, tre_for_each_slices
from rosi.registration.transformation import ParametersFromRigidMatrix



nb_data=3

nb_sample_train = 0#nb_data//2
nb_sample_test = 1#nb_sample_train + ((nb_data-nb_sample_train)//2)
nb_sample_validate = 2#nb_sample_train + nb_data - (nb_sample_train + nb_sample_test)
features= ['mse','mask_proportion','inter','dice'] #
nb_features = len(features)

X_train=np.zeros((0,nb_features))
Y_train=np.zeros((0))
error_train=np.zeros((0))
X_test=np.zeros((0,nb_features))
Y_test=np.zeros((0))
error_test=np.zeros((0))
X_validate=np.zeros((0,nb_features))
Y_validate=np.zeros((0))
Y_gholipour = np.zeros((0))
Y_kim = np.zeros((0))

nb_outlier_moy=[]
nb_taille_list=[]


nb_train_data=0
nb_test_data=0
nb_validate_data=0
image_name_list = ['trespetit','petit','moyen','grand']
directory_name_list=  ['tres_petit','Petit','Moyen','Grand']

estimated = []
#image_name_list = ['data']
index=np.linspace(1,nb_data,nb_data,dtype=int)
random.shuffle(index)
print(index)
id=0
for indice_name in range(0,len(image_name_list)):
    
    image_name = image_name_list[indice_name]
    print(image_name)
    directory_name = directory_name_list[indice_name]
    print(directory_name)

    for indice in range(0,nb_data):

        im=index[indice] #we take one random value every time
        #print(i)
        #im=2
        #image_name='grand'
        #directory_name='Data'
        name = image_name + '%d' %(im)
        dir_name = directory_name + '%d' %(im)

        #error,number_point,intersection,union=re.computeCostBetweenAll2Dimages(listSlice)
            
        directory = '../../simu/'+dir_name+'/' 
 
        # name='nomvt'
        transfo_axial =  directory + 'transfoAx_'+name+'.npy' #%(i)
        transfo_sagittal  = directory + 'transfoSag_'+name+'.npy' #%(i)
        transfo_coronal = directory + 'transfoCor_'+name+'.npy' #%(i)
        transfo = np.array([transfo_axial,transfo_coronal,transfo_sagittal])
        print(transfo)
            
        axial_without_movment = directory + 'LrAxNifti_'+name+'.nii.gz'
        img_axial_without_movment = nib.load(axial_without_movment)
        axial_mask_without_movment = directory + 'brain_mask/LrAxNifti_'+name+'.nii.gz' #%(i)
        #img_axial_mask_without_movment=nib.Nifti1Image(axial_mask_without_movment, img_axial_without_movment.affine)
        img_axial_mask_without_movment = nib.load(axial_mask_without_movment)
        coronal_without_movment = directory + 'LrCorNifti_'+name+'.nii.gz'
        img_coronal_without_movment = nib.load(coronal_without_movment)
        coronal_mask_without_movment = directory + 'brain_mask/LrCorNifti_'+name+'.nii.gz' #%(i)
        # #img_coronal_mask_without_movment=nib.Nifti1Image(coronal_mask_without_movment, img_coronal_without_movment.affine)
        img_coronal_mask_without_movment = nib.load(coronal_mask_without_movment)
        sagittal_without_movment = directory + 'LrSagNifti_'+name+'.nii.gz'
        img_sagittal_without_movment = nib.load(sagittal_without_movment)
        sagittal_mask_without_movment =  directory+ 'brain_mask/LrSagNifti_'+name+'.nii.gz' #%(i)
        img_sagittal_mask_without_movment=nib.load(sagittal_mask_without_movment)
        # #img_sagittal_mask_without_movment=nib.Nifti1Image(sagittal_mask_without_movment, img_sagittal_without_movment.affine)#img_sagittal_mask_without_movment = nib.load(sagittal_mask_without_movment)
            
        listSlice = []
        output = convert2Slices(img_axial_without_movment,img_axial_mask_without_movment,[],0,0)
        listSlice.extend(output)
        output = convert2Slices(img_coronal_without_movment,img_coronal_mask_without_movment,[],1,1)
        listSlice.extend(output)
        output = convert2Slices(img_sagittal_without_movment,img_sagittal_mask_without_movment,[],2,2)
        listSlice.extend(output)


        nbSlice=len(listSlice)

        #parameters
        parameters_axial = directory + 'transfoAx_'+name+'.npy'
        parametes_coronal = directory + 'transfoCor_'+name+'.npy'
        parameters_sagittal = directory+'transfoSag_'+name+'.npy'
        parameters = np.array([parameters_axial,parametes_coronal,parameters_sagittal])
        
        listFeatures = [sliceFeature(slicei.get_stackIndex(),slicei.get_indexSlice()) for slicei in listSlice]
        error,number_point,intersection,union = compute_cost_matrix(listSlice)
        
        update_features(listSlice,listFeatures,error,number_point,intersection,union)
        it=0
        while it < len(listFeatures):
             if listFeatures[it].get_mask_proportion()<0.1:
                del listFeatures[it]
                del listSlice[it]
             else:    
                it+=1
        
        nbSlice = len(listSlice)
        
        nb_simul_not_corrected = 5#np.int(np.ceil(nbSlice/3)) #random.randint(2,np.ceil(nbSlice/3))
        not_corrected = random.sample(range(0,nbSlice),nb_simul_not_corrected) 
        
        for i_slice in range(0,nbSlice):
            if i_slice not in not_corrected:
                i_stacks = listSlice[i_slice].get_stackIndex()
                i_index = listSlice[i_slice].get_indexSlice()
                transfo = np.load(parameters[i_stacks])
                t_slice = transfo[i_index,:,:]        
                x,y,z = listSlice[i_slice].get_slice().shape
                #center_image = np.ones(4); center_image[0] = x//2; center_image[1] = y//2; center_image[2] = 0; center_image[3]= 1
                #center_world = listSlice[i_slice].get_slice().affine @ center_image
                corner_to_center = np.eye(4); corner_to_center[0:3,3]=listSlice[i_slice].get_centerOfRotation()#-center_world[0:3]
                center_to_corner = np.eye(4); center_to_corner[0:3,3]=-listSlice[i_slice].get_centerOfRotation()#center_world[0:3]   
                M_k =  np.linalg.inv(center_to_corner) @ t_slice @ np.linalg.inv(corner_to_center)
                p_slice=ParametersFromRigidMatrix(M_k)
                print(p_slice)
                listSlice[i_slice].set_parameters(p_slice)
        from rosi.simulation.validation import same_order
        listFeatures = [sliceFeature(slicei.get_stackIndex(),slicei.get_indexSlice()) for slicei in listSlice]
        tre_for_each_slices(listSlice,listSlice,listFeatures,parameters,[])
        #print([e.get_error() for e in listFeatures])
        error,number_point,intersection,union = compute_cost_matrix(listSlice)
        update_features(listSlice,listFeatures,error,number_point,intersection,union)

        nb_features=len(features)
        nb_point=len(listFeatures)
        X=np.zeros((nbSlice,nb_features))
        Y=np.zeros(nbSlice)
        Y[not_corrected]=1


        for i_feature in range(0,len(features)):
            fe = features[i_feature]
            vaules_features = np.array([getattr(currentError,'_'+fe) for currentError in listFeatures])
            vaules_features=np.squeeze(vaules_features)
            print(vaules_features.shape)
            vaules_features[np.isnan(vaules_features)]=0
            vaules_features[np.isinf(vaules_features)]=0
            X[:,i_feature]=vaules_features
        
        set_r = removeBadSlice(listSlice,Y)
        tre_for_each_slices(listSlice,listSlice,listFeatures,parameters,set_r)
        tr = [e.get_error() for e in listFeatures]
        if im%2==0:
            X_test = np.concatenate((X_test,X))
            Y_test = np.concatenate((Y_test,Y))
            Y_estimated_kim = X[:,0] > (1.25*np.mean(X[:,0]))
            Y_kim = np.concatenate((Y_kim,Y_estimated_kim))
            q1 = np.quantile(X[:,0],0.25)
            q3 = np.quantile(X[:,0],0.75)
            Y_estimated_gholipour = X[:,0] > (2.5*q3 - 1.5*q1)
            Y_gholipour = np.concatenate((Y_gholipour,Y_estimated_gholipour))
        else : 
            X_train = np.concatenate((X_train,X))
            Y_train = np.concatenate((Y_train,Y))
            

np.save('../../simu/outliers/X_train.npy',X_train)
np.save('../../simu/outliers/Y_train.npy',Y_train)
np.save('../../simu/outliers/X_test.npy',X_test)
np.save('../../simu/outliers/Y_test.npy',Y_test)
np.save('../../simu/outliers/Y_kim.npy',Y_kim)
np.save('../../simu/outliers/Y_gholipour.npy',Y_gholipour)


X_train = np.load('../../simu/outliers/X_train.npy')
Y_train = np.load('../../simu/outliers/Y_train.npy')
X_test = np.load('../../simu/outliers/X_test.npy')
Y_test = np.load('../../simu/outliers/Y_test.npy')
Y_kim = np.load('../../simu/outliers/Y_kim.npy')
Y_gholipour = np.load('../../simu/outliers/Y_gholipour.npy')

clf = RandomForestClassifier()
clf.fit(X_train, Y_train)
estimated_y = clf.predict(X_test)

import pickle
filename = 'my_model_test.pickle'
pickle.dump(clf,open(filename,'wb'))

import pickle
filename = 'my_model_test.pickle'
clf = pickle.load(open(filename,'rb'))

clf = RandomForestClassifier(oob_score=True)
nb_data,nb_classes = X_train.shape
X_train = (X_train[:,2])#,X_train[:,2].T,X_train[:,3].T))
X_train = X_train.reshape((-1,1)) #X_train.reshape((3,nb_data)).T
nb_data,nb_classes = X_test.shape
X_test = (X_test[:,2])#,X_test[:,2].T,X_test[:,3].T))
X_test = X_test.reshape((-1,1))#X_test.reshape((3,nb_data)).T
w=18*np.zeros(len(X_train))
clf.fit(X_train, Y_train,sample_weight=w)
estimated_y = clf.predict(X_test)

plt.rcParams.update({'figure.figsize': [6.4,4.8]})
plt.rcParams.update({'xtick.labelsize' : 'xx-large'})
plt.rcParams.update({'ytick.labelsize' :'xx-large'})
plt.rcParams.update({'axes.labelsize' : 'xx-large'})
plt.rcParams.update({'font.size' : 15})
cmd = ConfusionMatrixDisplay.from_predictions(Y_test, estimated_y,sample_weight=w,normalize='true',display_labels=['good','outliers'],xticks_rotation="vertical",text_kw={'fontsize' : 'xx-large'})
cmd.plot()
plt.title('DIFF',fontsize=30)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
savefile = 'confusion_matrix_classification_mse_inter_dice.jpeg' 
plt.savefig(savefile,bbox_inches="tight")


features= ['mse','mask_proportion','inter','dice']
importances = clf.feature_importances_
forest_importances = pd.Series(importances, index=features)
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

fig, ax = plt.subplots(figsize=(10,12))
forest_importances.plot.bar() #yerr=std, ax=ax
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
#fig.tight_layout()
plt.savefig("fetures_importance_mse_mask_inter_dice.png")

# #Ebner
ebner_y = X_test[:,0]>0.8
#cm = confusion_matrix(Y_test, estimated_y,normalize='true')
#plt.figure()
#plt.rcParams({'font.size':20})
cmd = ConfusionMatrixDisplay.from_predictions(Y_test,ebner_y,normalize='true',display_labels=['good','outliers'],xticks_rotation="vertical",text_kw={'fontsize' : 'xx-large'})
cmd.plot()
plt.title('NiftyMIC',fontsize=30)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
savefile = 'confustion_matrix_classification_ebner.jpeg' 
plt.savefig(savefile,bbox_inches="tight")

# #kim
#cm = confusion_matrix(Y_test, Y_kim ,normalize='true'
plt.figure()
cmk = ConfusionMatrixDisplay.from_predictions(Y_test,Y_kim,normalize='true',display_labels=['good','outliers'],xticks_rotation="vertical",text_kw={'fontsize' : 'xx-large'})
cmk.plot()
plt.title('SLIMMER',fontsize=20)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
savefile = 'confustion_matrix_classification_kim.jpeg' 
plt.savefig(savefile,bbox_inches="tight")

# #Gholipour
plt.figure()
cm = confusion_matrix(Y_test, Y_gholipour,normalize='true')
cmg = ConfusionMatrixDisplay.from_predictions(Y_test,Y_gholipour,normalize="true",display_labels=['good','outliers'],xticks_rotation="vertical",text_kw={'fontsize' : 'xx-large'})
fig = plt.figure()
cmg.plot()
plt.title('GHOLIPOUR',fontsize=30)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
savefile = 'confusion_matrix_classification_gholipour.jpeg' 
plt.savefig(savefile,bbox_inches="tight")
