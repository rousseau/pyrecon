import array
from cmath import isinf, isnan, nan
from math import ceil
import pickle

from numpy import concatenate, median, reshape, squeeze, std, where, zeros
from ..intersection import cost_between_2slices
from ..outliers_detection.outliers import sliceFeature
from ..sliceObject import SliceObject
from ..tools import separate_slices_in_stacks, somme


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
    
    union = somme(union_matrix[:,k])+somme(union_matrix[k,:])
    inter = somme(intersection_matrix[:,k])+somme(intersection_matrix[k,:])
    n=somme(union_matrix[:,k]>0)+somme(union_matrix[k,:]>0)

    if n==0:
        return 0
    return (union-inter)/n

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
    for k in slices_index:
        
        current_feature = listOfFeatures[k]
        slicei=listOfSlice[k]
        fk=slicei.get_index_image()
        var_fk=variance[fk]
     
        #Update MSE
        #Correspond to eqaution : 
        #sum _{i=1,i!=fk} (1/ (var_fk + var_i)) *( ( sum_{k',f(k)=i} S^2(k,k') ) /  ( sum_{k',f(k)=i} N^2(k,k') ) )
        stack = stacks[fk]
        mse=0
        for i_stack in range(0,len(stacks)):
            if i_stack != fk:

                var_i = variance[i_stack]
                interested_values  = [(slicei.get_index_image()==fk or slicei.get_index_image()==i_stack) for slicei in listOfSlice]
                interested_values = array(interested_values)
                zeros_values = slices_index[where(interested_values==False)[0]]
                mse_tmp=compute_outliers_mse(k,listOfSlice,zeros_values,square_error_matrix,nbpoint_matrix)
                mse = mse + (1/(var_fk+var_i))*mse_tmp 

        current_feature.set_mse(mse)
        
        
        #Update DICE
        dice=cost_slice(intersection_matrix,union_matrix,k)
        current_feature.set_dice(dice)
        
        #Update DIFF
        inter=mask_difference(union_matrix,intersection_matrix,k)
        current_feature.set_inter(inter)
        
        #Update mask_proportion
        fk=slicei.get_index_image()
        mtot=pmasktot[fk]
        mask=slicei.get_mask()
        mprop=somme(mask)
        current_feature.set_mask_point(mprop)
        current_feature.set_mask_proportion(mprop/mtot)
        
        #Update the distance to the mask
        stack=stacks[fk]
        i_slice_in_stack = [stack[index_slice].get_index_slice()==slicei.get_index_slice() for index_slice in range(0,len(stack))].index(True)
        center_dist = distance_to_center(stack,i_slice_in_stack,center_volume[fk])
        current_feature.set_center_distance(center_dist)
        
        #Update the std intensity of the mask
        data=slicei.get_slice().get_fdata()
        std_in_image=std_intensity(data,mask)
        std_norm = std_total[fk]
        current_feature.set_std_intensity(std_in_image/std_norm)
        
        

def compute_noise_variance(stack : 'list[SliceObject]') -> float:
    """
    Compute the noise variance for each volume of the stack. (Used to normalized the MSE for outliers detection)
    some help on computed the noise : 
    Reference: J. Immerkær, “Fast Noise Variance Estimation”, Computer Vision and Image Understanding, Vol. 64, No. 2, pp. 300-302, Sep. 1996
    """
    
    data=[slice_k.get_slice().get_fdata().squeeze() for slice_k in stack]
    mask=[slice_k.get_mask().squeeze() for slice_k in stack]
 
    laplacien=array([[0,1,0],[1,-4,1],[0,1,0]]) #laplacien filter
    laplacien_convolution = [scipy.ndimage.convolve(data_slice*mask_slice,laplacien).reshape(-1) for (data_slice,mask_slice) in zip(data,mask)] #convolve with laplacien

    
    data_mask = [mask_slice.reshape(-1) for mask_slice in mask]
    data_mask = concatenate(data_mask)
     
    noise = concatenate(laplacien_convolution)
    noise=noise[data_mask>0]

    #Estimate variance using mad estimator
    med = median(noise)
    mad = median(abs(noise - med))
    k=1.4826
    sigma=(k*mad)
    variance=sigma**2
        
    return variance/20


def compute_ncc(k : int,listSlice : 'list[SliceObject]'):
    """
    Compute the normalized cross corelation (NCC)
    """
    
    ncc_moy=[]
    slicei=listSlice[k]
    for slice2 in listSlice:
        if slice2 != slicei:
            _,_,_,_,ncc,_ = cost_between_2slices(slicei,slice2)
            ncc_moy.append(ncc)
    ncc_moy=array(ncc_moy)
    ncc_moy>0
    n = somme(ncc_moy>0)
    ncc_moy=somme(ncc_moy[ncc_moy>0])
    
    return ncc_moy/n

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
                mse_moy.append(square_error_matrix[k,kprime])
                nbpoint.append(nbpoint_matrix[k,kprime])
    
    mse_moy=array(mse_moy)
    nbpoint=array(nbpoint)
    
    mse_moy=somme(mse_moy[mse_moy>-1])
    points_moy=somme(nbpoint[nbpoint>-1])

    if somme(nbpoint[nbpoint>-1])==0:
        return nan
    
    return mse_moy/points_moy
        
def data_to_classifier(listFeatures : 'list[sliceFeature]',
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
    X=zeros((nb_point,nb_features))
    
    for i_feature in range(0,len(features)):
        fe = features[i_feature]
        vaules_features = array([getattr(currentError,'_'+fe) for currentError in listFeatures])
        vaules_features=squeeze(vaules_features)
        print(vaules_features.shape)
        vaules_features[isnan(vaules_features)]=0
        vaules_features[isinf(vaules_features)]=0
        X[:,i_feature]=vaules_features
    
    return X,Y #X contains the features for all points and Y a classification of bad and good slices
        
def detect_misregistered_slice(listOfSlices : 'list[SliceObject]',
                               cost_matrix : array,
                               loaded_model : pickle) -> (array,array):
     """
     Labels each slices to well-registered or badly-registered
     """
     square_error_matrix = cost_matrix[0,:,:]
     nbpoint_matrix = cost_matrix[1,:,:]
     intersection_matrix = cost_matrix[2,:,:]
     union_matrx= cost_matrix[3,:,:]
     
     
     #Create a list of features and update the features 
     #note à moi meme : si possible udapte directement les features
     list_without_outliers = [sliceFeature(slice_k.get_index_image(),slice_k.get_index_slice()) for slice_k in listOfSlices] 
     update_features(listOfSlices,list_without_outliers,square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrx) 
     features=['mse','inter','dice','mask_proportion','std_intensity']

     #convert the data into input for the classifier
     X,Y=data_to_classifier(list_without_outliers,features,1.5)
     
     #classifier for prediction
     estimated_y=loaded_model.predict(X)
     out_image,list_without_outliers = detect_slices_out_of_images(list_without_outliers,intersection_matrix)
     
     return abs(estimated_y),abs(out_image)    

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


