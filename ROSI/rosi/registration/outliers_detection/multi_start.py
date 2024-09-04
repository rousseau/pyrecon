import array
from cmath import nan
import copy
from functools import partial
from multiprocessing import Pool
import pickle
from numpy import eye, ones, shape, where, zeros, array, sum 
from numba import jit
import numpy as np
from ..intersection import compute_cost_from_matrix, compute_cost_matrix, update_cost_matrix, updateResults, cost_fct
from ..optimisation import algo_optimisation, nelder_mead_optimisation, multi_optimisation, translation_optimisation
from .feature import detect_misregistered_slice
from ..sliceObject import SliceObject

from ..tools import line, somme
from ..transformation import ParametersFromRigidMatrix, log_cplx, rigidMatrix, rotation_diff

from numpy.linalg import inv, eig
from scipy.linalg import expm

from rosi.simulation.validation import theorical_misregistered, tre_for_each_slices
from skimage.measure import block_reduce


#find the best closer solutions
def good_neighboors(listOfSlice,k,set_r,nb_neigbhoors) :

    i_before=nb_neigbhoors*[None]
    i_after=nb_neigbhoors*[None]

    number_slices = len(listOfSlice)
    stack = listOfSlice[k].get_indexVolume()
    it=0
    pre_before = k
    pre_after = k
    while it<nb_neigbhoors:
        #step = (it+1)*2
        step=2
        for n1 in range(pre_before-step,0,-step):
            if set_r[n1]==0 and stack==listOfSlice[n1].get_indexVolume() and not n1 in i_before :
                i_before[it]=n1
                pre_before=n1
                break
        for n2 in range(pre_after+step,number_slices,step) :
            if set_r[n2]==0 and stack==listOfSlice[n2].get_indexVolume() and not n2 in i_after:
                i_after[it]=n2
                pre_after=n2
                break
        it+=1
    
    return i_before,i_after

#estimate a new initial postion for the slice
def estimate_new_position(listSlice,i_before,i_slice,i_after):

    center_in_world=listSlice[i_slice].get_centerOfRotation()
    corner_to_center=eye(4)
    corner_to_center[0:3,3]=center_in_world
    center_to_corner=eye(4)
    center_to_corner[0:3,3]=-center_in_world
    slice_transformation = listSlice[i_slice].get_slice().affine

    if i_before != None and i_after != None :

        M1=listSlice[i_before].get_estimatedTransfo()
        #print(M1)
        M2=listSlice[i_after].get_estimatedTransfo()
        #print(M2)
        #p1=listSlice[i_before].get_parameters()
        #print(p1)
        #p2=listSlice[i_after].get_parameters()
        #print(p2)
        #M1=rigidMatrix(p1)
        #M2=rigidMatrix(p2)

        r1=M1[0:3,0:3]
        r2=M2[0:3,0:3]

        T1=M1[0:3,3]
        T2=M2[0:3,3]

        d1=np.abs(i_slice-i_before)
        d2=np.abs(i_slice-i_after)
        d1=1
        d2=1

        estimated_rotation = computeMeanRotation(r1,d1,r2,d2)
        estimated_translation = computeMeanTranslation(T1,d1,T2,d2)

        estimated_transfo = eye(4)
        estimated_transfo[0:3,0:3]=estimated_rotation
        estimated_transfo[0:3,3]=estimated_translation
        estimated_rigid = inv(center_to_corner) @ ((estimated_transfo @ inv(slice_transformation)) @ inv(corner_to_center))
        #self.__estimated_transfo = self.__center_to_corner @ (self.__rigid_matrix @ (self.__corner_to_center @ slice_transformation))
        new_x = ParametersFromRigidMatrix(estimated_rigid)
        check_transfo = rigidMatrix(new_x)
        
        #print('determinant :',np.linalg.det(estimated_rigid))
        #print('estimated :',estimated_rigid)
        #print('transfo :',check_transfo)
        
        listSlice[i_slice].set_parameters(new_x)
        #print(listSlice[i_slice].get_estimatedTransfo())
        
    
    if i_before == None and i_after != None :

        p1=listSlice[i_after].get_parameters()
        G1 = rigidMatrix(p1)
        M1 = listSlice[i_after].get_estimatedTransfo()

        center_in_world=listSlice[i_after].get_centerOfRotation()
        corner_to_center_m1=eye(4)
        corner_to_center_m1[0:3,3]=center_in_world
      
        center_to_corner_m1=eye(4)
        center_to_corner_m1[0:3,3]=-center_in_world

        G2 = corner_to_center @ center_to_corner_m1 @ G1 @ corner_to_center_m1 @ center_to_corner
        p2 = ParametersFromRigidMatrix(G2)

        new_x = p2

    if i_after == None  and i_before != None:

        p1=listSlice[i_before].get_parameters()
        G1 = rigidMatrix(p1)
        M1 = listSlice[i_before].get_estimatedTransfo()

        center_in_world=listSlice[i_before].get_centerOfRotation()
        corner_to_center_m1=eye(4)
        corner_to_center_m1[0:3,3]=center_in_world
      
        center_to_corner_m1=eye(4)
        center_to_corner_m1[0:3,3]=-center_in_world

        G2 = corner_to_center @ center_to_corner_m1 @ G1 @ corner_to_center_m1 @ center_to_corner
        p2 = ParametersFromRigidMatrix(G2)

        new_x = p2
    
    if i_after == None and i_before == None:

        new_x = listSlice[i_slice].get_parameters()

    return new_x


def multi_start(hyperparameters,i_slice,listSlice,cost_matrix,set_o,x0,valstart,Vmx,optimisation,index):
    #x0,valstart
    """
    Function to try different initial position for optimisation. 

    Parameters
    ----------
    hyperparameters : parameters for optimisation : simplex size, xatol, fatol, epsilon, gauss, lamb
    i_slice : slice we want to correct
    listSlice : set of slices
    grid_slices : matrix costs
    set_o : set of outliers
    x0 : initial postion of the slice
    valstart : value for mutlistart

    Returns
    -------
    cost :
        cost after optimisation
    
    x_opt :
        best parameters of slices obtained with optimisation

    """
    x=x0+valstart[index,:]
    listSlice[i_slice].set_parameters(x)
    x_opt = multi_optimisation(hyperparameters,listSlice,cost_matrix,set_o,set_o,Vmx,i_slice,x,optimisation)
  

    
    return x_opt

def multi_start2(hyperparameters,i_slice,listSlice,cost_matrix,set_o,x0,Vmx,optimisation,index):
    #x0,valstart
    """
    Function to try different initial position for optimisation. 

    Parameters
    ----------
    hyperparameters : parameters for optimisation : simplex size, xatol, fatol, epsilon, gauss, lamb
    i_slice : slice we want to correct
    listSlice : set of slices
    grid_slices : matrix costs
    set_o : set of outliers
    x0 : initial postion of the slice
    valstart : value for mutlistart

    Returns
    -------
    cost :
        cost after optimisation
    
    x_opt :
        best parameters of slices obtained with optimisation

    """
    x=x0[index]
    listSlice[i_slice].set_parameters(x)
    x_opt = multi_optimisation(hyperparameters,listSlice,cost_matrix,set_o,Vmx,i_slice,x,optimisation)


    
    return x_opt


def removeBadSlice(listSlice,set_o):
    """
    return a list of bad-registered slices and their corresponding stack
    """
    removelist=[]

   
    for i_slice in range(len(listSlice)):
      if set_o[i_slice]==1:
          removelist.append((listSlice[i_slice].get_indexVolume(),listSlice[i_slice].get_indexSlice()))
   
    
    return removelist
        


def computeMeanRotation(R1 : np.array,
                        dist1 : int,
                        R2 : np.array,
                        dist2 : int) -> np.array:
    
    """
    Weigted mean between two rotations
    """
    M = R2 @ inv(R1)
    d,v = eig(M)
    tmp = log_cplx(d)
    A = v @ np.diag(tmp) @ inv(v)
    R_mean=expm(A*dist1/(dist1+dist2)) @ R1
    R_mean=np.real(R_mean)
    
    return R_mean

def computeMeanTranslation(T1 : np.array,
                           dist1 : int,
                           T2 : np.array,
                           dist2 : int) -> np.array:
    
    """
    Weigted mean between two translations
    """
    T_mean=((dist2) * T1 + (dist1) * T2)/(dist1 + dist2)
    
    return T_mean


def grid_search(x0,hyperparameters,listOfSlices,cost_matrix,set_o,Vmx,k,optimisation):

    a,b,c = x0[0:3]
    #x_opt,cost = translation_optimisation(hyperparameters,listOfSlices,cost_matrix,set_r,set_o,Vmx,k,x0)
    
    #print(x_t)
    vect_a = np.array([a-6,a-3,a,a+3,a+6])
    vect_b = np.array([b-6,b-3,b,b+3,b+6])
    vect_c = np.array([c-6,c-3,c,c+3,c+6])
    x=np.zeros(6)
    grid = np.zeros((len(vect_a),len(vect_b),len(vect_c)))
    estimated_translation = np.zeros((len(vect_a),len(vect_b),len(vect_c),3))

    for i in range(0,len(vect_a)):
        for j in range(0,len(vect_b)):
            for z in range(0,len(vect_c)):                
                x[:3]=[vect_a[i],vect_b[j],vect_c[z]]
                x[3:6]=x0[3:6]
                x_opt,cost=translation_optimisation(hyperparameters,listOfSlices,cost_matrix,set_o,Vmx,k,x,optimisation)
                x_t=x_opt[3:6]
                x = np.array([vect_a[i],vect_b[j],vect_c[z],x_t[0],x_t[1],x_t[2]])
                listOfSlices[k].set_parameters(x)
                cost=cost_fct(x,k,listOfSlices,cost_matrix,set_o,hyperparameters['omega'],Vmx)
                grid[i,j,z]=cost
                estimated_translation[i,j,z,:]=x_t

    return grid,estimated_translation

@jit(nopython=True,fastmath=True)
def local_minimum(grid):
    
    local_minium = []
    cost_minimum = []
    X,Y,Z = np.shape(grid)
    padding = np.ones((X+2,Y+2,Z+2))*10
    padding[1:(X+1),1:(Y+1),1:(Z+1)] = grid
    
    for i in range(1,X+1):
        for j in range(1,Y+1):
            for z in range(1,Z+1):
                
                current = padding[i,j,z]

                prev_x = padding[i-1,j,z]
                prev_y = padding[i,j-1,z]
                prev_z = padding[i,j,z-1]
                next_x = padding[i+1,j,z]
                next_y = padding[i,j+1,z]
                next_z = padding[i,j,z+1]

                if current < prev_x and current<prev_y and current<prev_z and current<next_x and current<next_y and current<next_z:
                        
                        cost_minimum.append(current)
                        local_minium.append((i-1,j-1,z-1))   
                
    minima = [x for _,x in sorted(zip(cost_minimum,local_minium))]

    return minima[0:5]



def find_minimum(x0,index,estimated_translation):
    

    a,b,c = x0[0:3]
    vect = np.array([-6,-3,0,+3,+6])
    starts = []

    for i in range(0,len(index)):

        r = np.array([a+vect[index[i][0]],b+vect[index[i][1]],c+vect[index[i][2]]])
        t = estimated_translation[index[i][0],index[i][1],index[i][2],:]
        starts.append(np.concatenate((r,t)))

    return starts

def best_value(hyperparameters,listSlice,cost_matrix,set_o,Vmx,i_slice,starts,optimisation):

    cost = np.zeros(5)
    optimise_parameters = np.zeros((5,6))
    
    x_min=listSlice[i_slice].get_parameters()
    square_error_matrix = cost_matrix[0,:,:]
    nbpoint_matrix = cost_matrix[1,:,:]
    intersection_matrix = cost_matrix[2,:,:]
    union_matrix = cost_matrix[3,:,:]
    update_cost_matrix(i_slice,listSlice,square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix)
    cost_matrix = array([square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix])
    cost_pre=compute_cost_from_matrix(cost_matrix[0,:,:],cost_matrix[1,:,:]) - hyperparameters['omega']*np.sum(cost_matrix[2,:,:])/Vmx + hyperparameters['omega']*np.sum(cost_matrix[3,:,:])/Vmx
    index = np.linspace(0,len(starts)-1,len(starts),dtype=int)
    with Pool(processes=6) as p:
        tmpfun=partial(multi_start2,hyperparameters,i_slice,listSlice,cost_matrix,set_o,starts,Vmx,optimisation)
        res=p.map(tmpfun,index)

    
    for x0 in res:
                
        cost = cost_fct(x0,i_slice,listSlice,cost_matrix,set_o,hyperparameters['omega'],Vmx)

        if cost<cost_pre:
            x_min=x0
            cost_pre=cost
    
    square_error_matrix = cost_matrix[0,:,:]
    nbpoint_matrix = cost_matrix[1,:,:]
    intersection_matrix = cost_matrix[2,:,:]
    union_matrix = cost_matrix[3,:,:]
    listSlice[i_slice].set_parameters(x_min)        

    update_cost_matrix(i_slice,listSlice,square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix)
    cost_matrix = array([square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix])

    return x_min,cost_pre



def correct_slice(set_r,set_o,listOfSlice,hyperparameters,optimisation,Vmx,matrix,listFeatures,loaded_model,dicRes,threshold):


    square_error_matrix,nboint_matrix,intersection_matrix,union_matrix = compute_cost_matrix(listOfSlice)
    matrix = np.array([square_error_matrix,nboint_matrix,intersection_matrix,union_matrix])
    mask = [e.get_mask_proportion()[0] for e in listFeatures]
    mask = np.array(mask)

    while it < len(listFeatures):
            if listFeatures[it].get_mask_proportion()<0.1:
                    del listFeatures[it]
                    del listOfSlice[it]
            else:
                    it+=1

    i=0
    nb_outliers = np.sum(set_r)
    nb_slice = len(listOfSlice)
    if nb_outliers> np.int64(nb_slice/3):
        return set_r

    while i<1 :

        set_pre = set_r
        for i_slice in range(0,len(listOfSlice)):  
            if set_r[i_slice]==1 : 

                    nb_neigbhoors=3
                    i_before,i_after=good_neighboors(listOfSlice,i_slice,set_r,nb_neigbhoors)

                    new_x=np.zeros((15,6))
                    it=0
                    for it_b in range(0,len(i_before)):
                        for it_a in range(0,len(i_after)):
                            new_x[it,:] = estimate_new_position(listOfSlice,i_before[it_b],i_slice,i_after[it_a])
                            it+=1
                    for it_a in range(0,len(i_after)):
                        new_x[it,:] = estimate_new_position(listOfSlice,None,i_slice,i_after[it_a])
                        it+=1
                    for it_b in range(0,len(i_before)):
                        new_x[it,:] = estimate_new_position(listOfSlice,i_before[it_b],i_slice,None)
                        it+=1
                    
                    new_x = choose_postion(new_x)
                    print(len(new_x))
                    estimated_x=np.zeros((len(new_x),6))
                    cost=np.zeros((len(new_x)))
                    for it in range(0,len(new_x)):
                        grid,estimated_trans = grid_search(new_x[it,:],hyperparameters,listOfSlice,matrix,set_o,Vmx,i_slice,optimisation)
                        minimum = local_minimum(grid)
                        starts = find_minimum(new_x[it,:],minimum,estimated_trans)
                        estimated_x[it,:],cost[it] = best_value(hyperparameters,listOfSlice,matrix,set_o,Vmx,i_slice,starts,optimisation)
                    
                    min_index = np.argmin(cost)
                    min = estimated_x[min_index,:]
                    listOfSlice[i_slice].set_parameters(min)
                    update_cost_matrix(i_slice,listOfSlice,square_error_matrix,nboint_matrix,intersection_matrix,union_matrix)
                    matrix = array([square_error_matrix,nboint_matrix,intersection_matrix,union_matrix])
                    
                    new_set = detect_misregistered_slice(listOfSlice,matrix,loaded_model,0.8)
                    set_r = np.logical_and(set_r,new_set)

                    new_set = detect_misregistered_slice(listOfSlice,matrix,loaded_model,0.5)
                    set_o = np.logical_and(set_o,new_set)

        new_set = detect_misregistered_slice(listOfSlice,matrix,loaded_model,0.8)
        set_r = np.logical_and(set_r,new_set)
        i+=1
        
        if np.all(set_pre==set_r) and i>1:
            break
        else:
            set_r=new_set.copy()
            hyperparameters['omega']=1
            i+=1
    
    new_set = detect_misregistered_slice(listOfSlice,matrix,loaded_model,0.5)

    return set_r

def choose_postion(new_position):
    
    for iter in range(0,len(new_position)):
        if iter!=0:
            pA = new_position[iter,:]
            A = rigidMatrix(pA)
            RA = A[0:3,0:3]
        for pre in range(0,iter):
            pB = new_position[pre,:]
            B = rigidMatrix(pB)
            RB = B[0:3,0:3]
            theta = rotation_diff(RA,RB)
            dist = np.linalg.norm(pA[3:6]-pB[3:6])
            if theta<0.2 : #if we have more than 11 degrees of difference
                new_position[iter,:]=new_position[pre,:]
    
    new_position = np.unique(new_position,axis=0)
    return new_position

def check_distance(i_slice1,i_slice2,listOfSlice,M1prime,M2prime):
    
    slice1 = listOfSlice[i_slice1].copy()
    slice2 = listOfSlice[i_slice2].copy()

    slice1.set_parameters([0,0,0,0,0,0])
    slice2.set_parameters([0,0,0,0,0,0])

    M1 = slice1.get_estimatedTransfo()
    M2 = slice2.get_estimatedTransfo()

    print('dist before :',np.linalg.norm(M1[0:3,3] - M2[0:3,3]))
    print('dist before :',np.linalg.norm(M1prime[0:3,3] - M2prime[0:3,3]))
    print(M1prime[0:3,0:3],M2prime[0:3,0:3])


    