from math import ceil
import pickle

from numpy import concatenate, median, quantile, reshape, squeeze, std, where, zeros, array, sum, isinf, isnan, nan, ones, linspace,mean
from rosi.registration.intersection import cost_between_2slices,ncc_between2slice
from rosi.registration.outliers_detection.outliers import sliceFeature
from rosi.registration.sliceObject import SliceObject
from rosi.registration.tools import separate_slices_in_stacks, somme
from rosi.simulation.validation import theorical_misregistered
import numpy as np

import scipy
import csv

def mask_proportion(mask : array) -> float:
    """
    Compute the mask proprtion on the slice
    """
    x,y,z = mask.shape
    in_mask=0
    if x !=0 and y !=0 :
        for i in range(0,x):
            for j in range(0,y):
                if mask[i,j]>0:
                    in_mask=in_mask+1         
        res = in_mask/(x*y)
    else :
        res=0
    return res



def cost_slice(grid_numerator : array,grid_denumerator : array,k : int) -> float:
    """
    return the local cost of a slice
    """ 

    num = somme(grid_numerator[:,k])+somme(grid_numerator[k,:])
    denum = somme(grid_denumerator[:,k])+somme(grid_denumerator[k,:])
    if denum==0:
        return 0
    
    return num/denum

def distance_to_center(stack : 'list[SliceObject]',k : int,k_center : int) -> int: 
    """
    Return the distance between slice k and the slice in the center of the stack (k_center)
    """
    
    res=k-k_center
    if  res> 0:
        res = abs(res)/(len(stack)-k_center)
    else :
        res = abs(res)/k_center
    return res

def mask_difference(union_matrix : array ,intersection_matrix : array ,k : int) -> float:
    """
    Return the mask difference ie
    U(k,k') = sum_v (m_k(v) or m_k'(v'))
    M(k,k') = sum_v (m_k(v) and m_k'(v'))
    D_k =   sum_k' (U(k,k') - M(k'k)) / ((sum_k' 1 U(k,k') + sum_k' M(k,k')))
    """ 
    union = concatenate((union_matrix[:,k],union_matrix[k,:]))
    index_union = where(union>0)
    union = union[index_union]
    inter = concatenate((intersection_matrix[:,k],intersection_matrix[k,:]))
    inter = inter[index_union]
   
    return median(union-inter)

def std_intensity(data : array,mask : array) -> float:
    """
    Return std intensity of a mask slice
    """
    
    image=data*mask
    std_res=std(image)
    
    return std_res

def slice_center(mask3D : array) -> array:
    """
    Compute the slice at the center of the mask
    """
    
    index = where(mask3D>0)
    center = sum(index,axis=1)/(somme(mask3D))

    return center

def std_volume(stack : 'list[SliceObject]', masks :array) -> float:
    """
    Compute the standard deviation on a 3D lr image
    (use in normalisation)
    """
    
    data = [reshape(slicei.get_slice().get_fdata(),-1) for slicei in stack]
    data = concatenate(data)
    mask_data = [reshape(mask,-1) for mask in masks]
    mask_data = concatenate(mask_data)
    brain_data = data*mask_data
    res = std(brain_data)
    
    return res


def update_features(listOfSlice : 'list[SliceObject]',
                   listOfFeatures : 'list[sliceFeature]',
                   square_error_matrix : array,
                   nbpoint_matrix : array,
                   intersection_matrix : array,
                   union_matrix : array):
    
    """
    Update the features associated to each slices
    #pourrait être bien de pouvoir choisir quelles features mettre à jour
    """
    
    stacks,masks=separate_slices_in_stacks(listOfSlice)
    variance = [compute_noise_variance(img) for img in stacks]
    index = [len(masks[n]) for n in range(0,len(masks))]
    pmasktot=[max(sum(masks[n][0:index[n]],axis=(1,2))) for n in range(0,len(masks))]
    mask_volume=[concatenate(masks[n][0:index[n]],axis=2) for n in range(0,len(masks))]
    center_volume = [slice_center(mask_volume[n]) for n in range(0,len(stacks))]
    std_total = [std_volume(stacks[n],masks[n]) for n in range(0,len(masks))]
    
    slices_index = range(0,len(listOfSlice))
    slices_index = array(slices_index)
    index_variance = array([img[0].get_indexVolume() for img in stacks])
 
    for k in slices_index:
        
        current_feature = listOfFeatures[k]
        slicek=listOfSlice[k]
        fk=slicek.get_indexVolume()
        vk = where(index_variance==fk)[0][0]
        var_fk=variance[vk]
     
        #Update MSE
        #Correspond to eqaution : 
        #sum _{i=1,i!=fk} (1/ (var_fk + var_i)) *( ( sum_{k',f(k)=i} S^2(k,k') ) /  ( sum_{k',f(k)=i} N^2(k,k') ) )
        stack = stacks[vk]
        mse=[]
        nmse=[]
        for i_stack in range(0,len(stacks)):
            if i_stack != vk:
                
                vi = where(index_variance==i_stack)[0][0]
                var_i = variance[vi]
                #interesting_values  = [(slicei.get_stackIndex()==fk or slicei.get_stackIndex()==i_stack) for slicei in listOfSlice]
                interesting_values  = [(slicei.get_indexVolume()==i_stack) for slicei in listOfSlice]
                interesting_values = array(interesting_values)
                zeros_values = slices_index[where(interesting_values==False)[0]]
                mse_tmp=compute_outliers_mse(k,listOfSlice,zeros_values,square_error_matrix,nbpoint_matrix)
                nmse.append((1/((var_fk)+(var_i)))*mse_tmp)
                mse.append(mse_tmp)
               
                #*

        if len(mse)>1: 
            mse = concatenate(mse)
            nmse = concatenate(nmse)
     
        current_feature.set_mse(median(mse))
        current_feature.set_nmse(median(nmse))
        
        current_feature.set_nbpoint(np.sum(nbpoint_matrix[:,k])+np.sum(nbpoint_matrix[k,:]))
        
        #Update DICE
        union = concatenate((union_matrix[k,:],union_matrix[:,k]))
        union=array(union)
        index_union = where(union>0)
        union = union[index_union]
        inter = concatenate((intersection_matrix[k,:],intersection_matrix[:,k]))
        inter = inter[index_union] #we take only values where union_matrix is not zeros (if union matrix is zeros then slices to not intersect)
        
        dice_vect=inter/union
        dice=median(dice_vect)
        #cost_slice(intersection_matrix,union_matrix,k)
        current_feature.set_dice(dice)
        
        #Update DIFF
        inter=mask_difference(union_matrix,intersection_matrix,k)
        current_feature.set_inter(inter)
        
        #Update mask_proportion
        fk=slicek.get_indexVolume()
        mtot=pmasktot[vk]
        mask=slicek.get_mask()
        mprop=somme(mask)
        current_feature.set_mask_point(mprop)
        current_feature.set_mask_proportion(mprop/mtot)
        
        #Update the distance to the mask
        #stack=stacks[fk]
        #i_slice_in_stack = [stack[indexSlice].get_indexSlice()==slicei.get_indexSlice() for indexSlice in range(0,len(stack))].index(True)
        #center_dist = distance_to_center(stack,i_slice_in_stack,center_volume[fk])
        #current_feature.set_center_distance(center_dist)
        
        #Update the std intensity of the mask
        data=slicek.get_slice().get_fdata()
        std_in_image=std_intensity(data,mask)
        std_norm = std_total[vk]
        current_feature.set_std_intensity(std_in_image/std_norm)

        ncc = compute_ncc(k,listOfSlice)
        current_feature.set_ncc(ncc)

def update_features_v2(listOfSlice : 'list[SliceObject]',
                   listOfFeatures : 'list[sliceFeature]',
                   square_error_matrix : array,
                   nbpoint_matrix : array,
                   intersection_matrix : array,
                   union_matrix : array):
    
    """
    Update the features associated to each slices
    #pourrait être bien de pouvoir choisir quelles features mettre à jour
    """
    
    stacks,masks=separate_slices_in_stacks(listOfSlice)
    variance = [compute_noise_variance(img) for img in stacks]
    index = [len(masks[n]) for n in range(0,len(masks))]
    pmasktot=[max(sum(masks[n][0:index[n]],axis=(1,2))) for n in range(0,len(masks))]
    mask_volume=[concatenate(masks[n][0:index[n]],axis=2) for n in range(0,len(masks))]
    center_volume = [slice_center(mask_volume[n]) for n in range(0,len(stacks))]
    std_total = [std_volume(stacks[n],masks[n]) for n in range(0,len(masks))]
    
    slices_index = range(0,len(listOfSlice))
    slices_index = array(slices_index)
    for k in slices_index:
        
        current_feature = listOfFeatures[k]
        slicek=listOfSlice[k]
        fk=slicek.get_stackIndex()
        var_fk=variance[fk]
     
        #Update MSE
        #Correspond to eqaution : 
        #sum _{i=1,i!=fk} (1/ (var_fk + var_i)) *( ( sum_{k',f(k)=i} S^2(k,k') ) /  ( sum_{k',f(k)=i} N^2(k,k') ) )
        stack = stacks[fk]
        n_stack=0
        mse=0
        for i_stack in range(0,len(stacks)):
            if i_stack != fk:

                var_i = variance[i_stack]
                interesting_values  = [(slicei.get_stackIndex()==fk or slicei.get_stackIndex()==i_stack) for slicei in listOfSlice]
                
                interesting_values = array(interesting_values)
                zeros_values = slices_index[where(interesting_values==False)[0]]
                #print('slice',k,zeros_values)
                mse_tmp=compute_outliers_mse(k,listOfSlice,zeros_values,square_error_matrix,nbpoint_matrix)
                #print('mse_tmp',mse_tmp)
                if mse_tmp != 0 :
                    n_stack+=1
                #mse = mse + (1/(var_fk+var_i))*mse_tmp 
                mse=mse+mse_tmp

        mse=mse/(n_stack+1e-16)
        current_feature.set_mse(mse)
        
        
        #Update DICE
        dice=cost_slice(intersection_matrix,union_matrix,k)
        current_feature.set_dice(dice)
        
        #Update DIFF
        inter=mask_difference(union_matrix,intersection_matrix,k)
        current_feature.set_inter(inter)
        
        #Update mask_proportion
        fk=slicek.get_stackIndex()
        mtot=pmasktot[fk]
        mask=slicek.get_mask()
        mprop=somme(mask)
        current_feature.set_mask_point(mprop)
        current_feature.set_mask_proportion(mprop/mtot)
        
        #Update the distance to the mask
        #stack=stacks[fk]
        #i_slice_in_stack = [stack[indexSlice].get_indexSlice()==slicei.get_indexSlice() for indexSlice in range(0,len(stack))].index(True)
        #center_dist = distance_to_center(stack,i_slice_in_stack,center_volume[fk])
        #current_feature.set_center_distance(center_dist)
        
        #Update the std intensity of the mask
        data=slicek.get_slice().get_fdata()
        std_in_image=std_intensity(data,mask)
        std_norm = std_total[fk]
        current_feature.set_std_intensity(std_in_image/std_norm)


        ncc = compute_ncc(k,listOfSlice)
        current_feature.set_ncc(ncc)
        
        

def compute_noise_variance(stack : 'list[SliceObject]') -> float:
    """
    Compute the noise variance for each volume of the stack. (Used to normalized the MSE for outliers detection)
    some help on computed the noise : 
    Reference: J. Immerkær, “Fast Noise Variance Estimation”, Computer Vision and Image Understanding, Vol. 64, No. 2, pp. 300-302, Sep. 1996
    """
    data=[slice_k.get_slice().get_fdata().squeeze() for slice_k in stack]
    mask=[slice_k.get_mask().squeeze() for slice_k in stack]
    values = np.concatenate(data)
    values_mask = np.concatenate(mask)
    laplacien=array([[0,1,0],[1,-4,1],[0,1,0]]) #laplacien filter
    laplacien_convolution = [scipy.ndimage.convolve(data_slice,laplacien) for (data_slice,mask_slice) in zip(data,mask)] #convolve with laplacien
    data_mask = [mask_slice.reshape(-1) for mask_slice in mask]
    data_mask = concatenate(data_mask).reshape(-1)
    noise = concatenate(laplacien_convolution).reshape(-1) 
    noise=noise[data_mask>0]
    
    #Estimate variance using mad estimator
    med = median(noise)
    mad = median(abs(noise - med))
    k=1.4826
    sigma=(k*mad)
    variance=sigma**2

   
    return variance/20 #(1^2*4 + (-4)^2)=20


def compute_ncc(k : int,listSlice : 'list[SliceObject]'):
    """
    Compute the normalized cross corelation (NCC)
    """
    
    ncc_moy=[]
    slicei=listSlice[k]
    for slice2 in listSlice:
        if slice2 != slicei:
            ncc = ncc_between2slice(slicei,slice2)
            if ncc != -1 :
                ncc_moy.append(ncc)
    ncc_moy=array(ncc_moy)
    #ncc_moy>0
    n = len(ncc_moy)
    #somme(ncc_moy>0)
    ncc_moy= somme(ncc_moy)
    #somme(ncc_moy[ncc_moy>0])
    
    return ncc_moy/(n+1e-16)

def compute_outliers_mse(k : int,
                         listSlice : 'list[SliceObject]',
                         mse_indexes : array,
                         square_error_matrix : array,
                         nbpoint_matrix : array) -> float:
    """
    Compute a temporary mse that is use to computed to weigthed mse for feature detection
    """
    
    mse_moy=[]
    nbpoint=[]
    slice_k=listSlice[k]
    for kprime in range(0,len(listSlice)):
        if not kprime in mse_indexes:
            slice_kprime=listSlice[kprime]
            if slice_kprime != slice_k:
                mse_moy.append(square_error_matrix[max(k,kprime),min(k,kprime)])
                nbpoint.append(nbpoint_matrix[max(k,kprime),min(k,kprime)])
    mse_moy=array(mse_moy)
    nbpoint=array(nbpoint)
    index = array(nbpoint>0)
    mse_moy=array(mse_moy[index==1])
    points_moy=array(nbpoint[index==1])
    if somme(nbpoint[nbpoint>-1])==0:
        return array([0])
    
    return mse_moy/points_moy
        
def data_to_classifier(listFeatures : 'list[sliceFeature]',
                       listOfSlice : 'list[SliceObject]',
                       transfolist : array,
                       features : 'array[str]',
                       max_error : float):
    
    """
    Create the X and Y input for the classifiers
    X are the features
    Y are labels (1 for outliers else 0)
    """
    
    Y=theorical_misregistered(listOfSlice,listFeatures,transfolist)
    #Y=[e.get_error()>max_error for e in listFeatures]
    #Y = array(Y)
    
    nb_features=len(features)
    nb_point=len(listFeatures)
    X=zeros((nb_point,nb_features))
    
    for i_feature in range(0,len(features)):
        fe = features[i_feature]
        vaules_features = array([getattr(currentError,'_'+fe) for currentError in listFeatures])
        vaules_features=squeeze(vaules_features)
        vaules_features[isnan(vaules_features)]=0
        vaules_features[isinf(vaules_features)]=0
        X[:,i_feature]=vaules_features

    
    return X,Y #X contains the features for all points and Y a classification of bad and good slices

def data_to_classifier_real(listFeatures : 'list[sliceFeature]',
                            features : 'array[str]'):
    
    """
    Create the X and Y input for the classifiers
    X are the features
    Y are labels (1 for outliers else 0)
    """
    
    #Y=[e.get_error()>max_error for e in listFeatures]
    #Y = array(Y)
    
    nb_features=len(features)
    nb_point=len(listFeatures)
    X=zeros((nb_point,nb_features))
    
    for i_feature in range(0,len(features)):
        fe = features[i_feature]
        vaules_features = array([getattr(currentError,'_'+fe) for currentError in listFeatures])
        vaules_features=squeeze(vaules_features)
        vaules_features[isnan(vaules_features)]=0
        vaules_features[isinf(vaules_features)]=0
        X[:,i_feature]=vaules_features

    
    return X
 
def data_to_classifier_v2(listFeatures : 'list[sliceFeature]',
                       features : 'array[str]',
                       max_error : float):
    
    """
    Create the X and Y input for the classifiers
    X are the features
    Y are labels (1 for outliers else 0)
    """
    
    slice_tre=array([current_feature.get_error() for current_feature in listFeatures])
    Y=slice_tre>max_error
    
    nb_features=len(features)
    nb_point=len(listFeatures)
    X=zeros((nb_point,nb_features+2))
    
    for i_feature in range(0,len(features)):
        fe = features[i_feature]
        vaules_features = array([getattr(currentError,'_'+fe) for currentError in listFeatures])
        vaules_features=squeeze(vaules_features)
        vaules_features[isnan(vaules_features)]=0
        vaules_features[isinf(vaules_features)]=0
        X[:,i_feature]=vaules_features

        if fe=='mse':
            i_mse = i_feature
    
    if 'mse' in features :
        X[:nb_point-1,i_feature+1]= X[1:,i_mse]
        X[1:,i_feature+2]= X[:nb_point-1,i_mse]
    
    return X,Y #X contains the features for all points and Y a classification of bad and good slices


def detect_misregistered_slice(listOfSlices : 'list[SliceObject]',
                               cost_matrix : array,
                               loaded_model : pickle,
                               threshold : int) -> (array,array):
     """
     Labels each slices to well-registered or badly-registered
     """
     square_error_matrix = cost_matrix[0,:,:]
     nbpoint_matrix = cost_matrix[1,:,:]
     intersection_matrix = cost_matrix[2,:,:]
     union_matrx= cost_matrix[3,:,:]
     
     
     #Create a list of features and update the features 
     #note à moi meme : si possible udapte directement les features
     list_without_outliers = [sliceFeature(slice_k.get_stackIndex(),slice_k.get_indexSlice()) for slice_k in listOfSlices] 
     update_features(listOfSlices,list_without_outliers,square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrx) 
     features= ['inter','nmse','dice'] 

     #convert the data into input for the classifier
     X=data_to_classifier_real(list_without_outliers,features)
     
     #classifier for prediction
     proba=loaded_model.predict_proba(X)
     estimated_y = proba[:,0]<threshold
     
     return abs(estimated_y)    

def detect_slices_out_of_images(listFeatures : 'list[SliceObject]',
                                intersection_matrix : array) -> (array,array):
    """
    This function detect slices that don't intersect any other slice and therefore are badly register.
    However the cost on those slices can be computed because they don't intersect other slices.
    """

    nbSlice=len(listFeatures)
    list_far_far_away=zeros(nbSlice)
    
    i_slice=0
    index=0
    while i_slice<(nbSlice):
        
        nbpoint = somme(intersection_matrix[:,i_slice])+somme(intersection_matrix[i_slice,:])
        mask = listFeatures[index].get_mask_point()
       
        if  nbpoint < mask/2 :
            far_far_away = (listFeatures[index].get_stack_index(),listFeatures[index].get_index())
            list_far_far_away[i_slice]=1
            listFeatures.remove(listFeatures[index])
            index=index-1
            
        index=index+1
        i_slice=i_slice+1
    
    return list_far_far_away,listFeatures


def save_features_in_csv(listFeatures : 'list[sliceFeature]',
                         output : str) :
    
    labels = ['stack','position','tre','nmse','dice','mprop','std','diff']
    
    with open(output,'w',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(labels)
        for feature in listFeatures:
            values = [
                feature.get_stack(),
                feature.get_index(),
                feature.get_error(),
                feature.get_mse(),
                feature.get_dice(),
                feature.get_mask_proportion(),
                feature.get_std_intensity(),
                feature.get_inter()
            ]
            writer.writerow(values)

