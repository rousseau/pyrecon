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
from rosi.registration.intersection import compute_cost_matrix
from rosi.registration.load import convert2Slices
from rosi.registration.outliers_detection.feature import data_to_classifier,update_features
from rosi.registration.outliers_detection.outliers import sliceFeature
#import registration as re
import matplotlib.pyplot as plt
import os.path

from rosi.simulation.validation import same_order, tre_for_each_slices



nb_data=1

nb_sample_train = 0#nb_data//2
nb_sample_test = 1#nb_sample_train + ((nb_data-nb_sample_train)//2)
nb_sample_validate = 2#nb_sample_train + nb_data - (nb_sample_train + nb_sample_test)
features= ['ncc','mse', 'std_intensity','mask_proportion','inter','dice'] #
nb_features = len(features)

X_train=np.zeros((0,nb_features))
Y_train=np.zeros((0))
Z_train=np.zeros((0))
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
image_name_list = ['grand']#['petit','trespetit','moyen','grand']
directory_name_list=   ['Data']#['Petit','tres_petit','Moyen','Grand']

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

        name = image_name + '%d' %(im)
        dir_name = directory_name + '%d' %(im)
        joblib_name = '../../res/outliers/'+name+'.joblib.gz'
        print(joblib_name)
        if os.path.exists(joblib_name) :
            res = joblib.load(open(joblib_name,'rb'))
            key=[p[0] for p in res]
            element=[p[1] for p in res]
            listSlice=element[key.index('listSlice')]
            lerror=element[key.index('ListError')]
            nbSlice = len(listSlice)
            nb_taille_list.append(nbSlice)
            
            #print(element[key.index('RejectedSlices')])
            rejectedSlices = element[key.index('RejectedSlices')]
            
            #union_dice = element[key.index('EvolutionGridUnion')][-1,:,:]
            #intersection_dice =element[key.index('EvolutionGridInter')][-1,:,:] 
            error = element[key.index('EvolutionGridError')][-1,:,:]
            number_point = element[key.index('EvolutionGridNbpoint')][-1,:,:]
            intersection = element[key.index('EvolutionGridInter')][-1,:,:]
            union = element[key.index('EvolutionGridUnion')][-1,:,:]

            #error,number_point,intersection,union=re.computeCostBetweenAll2Dimages(listSlice)
            
            directory = '../../simu/outliers/'+dir_name+'/' 
 
            # name='nomvt'
            transfo_axial =  directory + 'transfoAx_'+name+'.npy' #%(i)
            transfo_sagittal  = directory + 'transfoSag_'+name+'.npy' #%(i)
            transfo_coronal = directory + 'transfoCor_'+name+'.npy' #%(i)
            transfo = np.array([transfo_axial,transfo_coronal,transfo_sagittal])
            print(transfo)
            
            axial_without_movment = directory + 'LrAxNifti_nomvt.nii.gz'
            img_axial_without_movment = nib.load(axial_without_movment)
            axial_mask_without_movment = directory + 'LrAxNifti_'+name+'_mask.nii.gz' #%(i)
            #img_axial_mask_without_movment=nib.Nifti1Image(axial_mask_without_movment, img_axial_without_movment.affine)
            img_axial_mask_without_movment = nib.load(axial_mask_without_movment)
            coronal_without_movment = directory + 'LrCorNifti_nomvt.nii.gz'
            img_coronal_without_movment = nib.load(coronal_without_movment)
            coronal_mask_without_movment = directory + 'LrCorNifti_'+name+'_mask.nii.gz' #%(i)
            # #img_coronal_mask_without_movment=nib.Nifti1Image(coronal_mask_without_movment, img_coronal_without_movment.affine)
            img_coronal_mask_without_movment = nib.load(coronal_mask_without_movment)
            sagittal_without_movment = directory + 'LrSagNifti_nomvt.nii.gz'
            img_sagittal_without_movment = nib.load(sagittal_without_movment)
            sagittal_mask_without_movment =  directory+ 'LrSagNifti_'+name+'_mask.nii.gz' #%(i)
            img_sagittal_mask_without_movment=nib.load(sagittal_mask_without_movment)
            # #img_sagittal_mask_without_movment=nib.Nifti1Image(sagittal_mask_without_movment, img_sagittal_without_movment.affine)#img_sagittal_mask_without_movment = nib.load(sagittal_mask_without_movment)
            
            listnomvt = []
            output = convert2Slices(img_axial_without_movment,img_axial_mask_without_movment,[],0,0)
            listnomvt.extend(output)
            output = convert2Slices(img_coronal_without_movment,img_coronal_mask_without_movment,[],1,1)
            listnomvt.extend(output)
            output = convert2Slices(img_sagittal_without_movment,img_sagittal_mask_without_movment,[],2,2)
            listnomvt.extend(output)
            
            erreur_coupe = [e.get_error() for e in lerror]
            listErrorSlice = [sliceFeature(slicei.get_stackIndex(),slicei.get_indexSlice()) for slicei in listSlice]
            image,theorique,lfeatures,transfolist=same_order(listSlice,listnomvt,listErrorSlice,transfo)
            listOfSlice=np.concatenate(image)
            error,number_point,intersection,union=compute_cost_matrix(listOfSlice)
            matrix = np.array([error,number_point,intersection,union])
            listFeatures=np.concatenate(lfeatures)
            listFeatures=listFeatures.tolist()
            listOfSlice=listOfSlice.tolist()
            
            update_features(listOfSlice,listFeatures,error,number_point,intersection,union)
            it=0
            while it < len(listFeatures):
                if listFeatures[it].get_mask_proportion()<0.1:
                    del listFeatures[it]
                    del listOfSlice[it]
                else:
                    it+=1
            
            print([len(image[i]) for i in range(0,3)],[len(theorique[i]) for i in range(0,3)],[len(lfeatures[i]) for i in range(0,3)])
            tre_for_each_slices(listnomvt,listOfSlice,listFeatures,transfo,[])
            X,Y=data_to_classifier(listFeatures,listOfSlice,transfolist,features,3)
            X_train = np.concatenate((X_train,X))
            Y_train = np.concatenate((Y_train,Y))
            Z = [e.get_error() for e in listErrorSlice]
            print('error',Z)
            
            id=id+1
            error_slices = [s.get_error() for s in listErrorSlice]
            print('nb_outlier',np.sum(Y))
            nb_outlier_moy.append(np.sum(Y))
            
            if  im%3==0 or im%3==1:
            #if dir_name != "Grand1"  and dir_name != "Grand2" and dir_name != "Grand3" and dir_name != "Grand4":
                nb_train_data = nb_train_data+1
                X_train = np.concatenate((X_train,X))
                Y_train = np.concatenate((Y_train,Y))
                Z_train = np.concatenate((Z_train,Z))
                error_train = np.concatenate((error_train,error_slices))

                plt.figure()
                title = 'fig_mask_%d' %(id)
                plt.scatter(X[:,2],X[:,0],c=Y)
                plt.savefig(title)

                
            else:
                nb_test_data = nb_test_data+1
                X_test = np.concatenate((X_test,X))
                Y_test = np.concatenate((Y_test,Y))
                Y_estimated_kim = X[:,1] > (1.25*np.mean(X[:,1]))
                Y_kim = np.concatenate((Y_kim,Y_estimated_kim))
                q1 = np.quantile(X_test[:,1],0.25)
                q3 = np.quantile(X_test[:,1],0.75)
                Y_estimated_gholipour = X[:,1] > (2.5*q3 - 1.5*q1)
                Y_gholipour = np.concatenate((Y_gholipour,Y_estimated_gholipour))
                error_test = np.concatenate((error_test,error_slices))
            
            # else:
            #     nb_validate_data = nb_validate_data+1
            #     X_validate = np.concatenate((X_validate,X))
            #     Y_validate = np.concatenate((Y_validate,Y))
        
clf = RandomForestClassifier()
clf.fit(X_train.reshape(-1,1), Y_train)
estimated_y = clf.predict(X_test.reshape(-1,1))


import pickle
filename = 'my_model_test.pickle'
pickle.dump(clf,open(filename,'wb'))

plt.rcParams.update({'figure.figsize': [6.4,4.8]})
plt.rcParams.update({'xtick.labelsize' : 'xx-large'})
plt.rcParams.update({'ytick.labelsize' :'xx-large'})
plt.rcParams.update({'axes.labelsize' : 'xx-large'})
plt.rcParams.update({'font.size' : 15})
#plt.rcParams.update({'axes.xmargin' : -0.5})
#plt.rcParams.update({'axes.ymargin' : -0.5})
#fig,axes = plt.subplots(figsize=(10,10))
#cm = confusion_matrix(Y_test, estimated_y,normalize='true')
cmd = ConfusionMatrixDisplay.from_predictions(Y_test, estimated_y,normalize='true',display_labels=['good','outliers'],xticks_rotation="vertical",text_kw={'fontsize' : 'xx-large'})
#fig = cmd.ax_.get_figure() 
#fig.set_figwidth(20)
#fig.set_figheight(20)  
#fig = plt.figure()
cmd.plot()
plt.title('DIFF',fontsize=30)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
savefile = 'confusion_matrix_classification_std.jpeg' 
plt.savefig(savefile,bbox_inches="tight")


importances = clf.feature_importances_
forest_importances = pd.Series(importances, index=features)
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

fig, ax = plt.subplots()
forest_importances.plot.bar() #yerr=std, ax=ax
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
#fig.tight_layout()
plt.savefig("fetures_importance.png")

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
