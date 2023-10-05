import array
from cmath import nan
import copy
from functools import partial
from multiprocessing import Pool
import pickle
from numpy import eye, ones, shape, where, zeros
from numba import jit
import numpy as np
from ..intersection import compute_cost_from_matrix, compute_cost_matrix, update_cost_matrix, updateResults
from ..optimisation import algo_optimisation, nelder_mead_optimisation
from .feature import detect_misregistered_slice
from ..sliceObject import SliceObject

from ..tools import line, somme
from ..transformation import ParametersFromRigidMatrix, log_cplx

from numpy.linalg import inv, eig
from scipy.linalg import expm

load_model = pickle.load(open('ROSI/rosi/registration/outliers_detection/my_new_model.pickle','rb'))

def correction_out_images(listSlice,hyperparameters,set_o,set_r,grid_slices,dicRes,Vmx):
    """
    This function aims to correct mis-registered slices

    Parameters
    ----------
    listSlice : set of the slices, from the three volumes
    hyperparameters : parameters for optimisation, it includes : simplex initial size, xatol, fatol, epsilon, gaussian parameter, lamnda for dice
    set_o : set of outliers or mis-registered slices
    grid_slices : matrix of cost
    dicRes : results save for representation

    Returns
    -------
    grid_slices : matrix of cost
    dicRes : results save for representation

    """
    print(set_o.shape)
    ge = grid_slices[0,:,:]
    gn = grid_slices[1,:,:]
    gi = grid_slices[2,:,:]
    gu= grid_slices[3,:,:]
    before_correction = sum(set_o)
    print(before_correction)
     
    nbSlice = len(listSlice)     
    while 1:
                
        #number of slices total
        print(set_o.shape)
        for i_slice in range(0,nbSlice):
            
        #     #print('before:',listSlice[i_slice].get_parameters())
        #     #print('Slice Cost Before:', oneSliceCost(ge,gn,set_o,i_slice))
            
            n_0=somme(gn[:,i_slice])+somme(gn[i_slice,:])
            okpre=False;okpost=False;nbSlice1=0;nbSlice2=0;dist1=0;dist2=0
            slicei=listSlice[i_slice]
            x_pre=slicei.get_parameters()
            print(set_o[i_slice])
            if set_o[i_slice]==1:
                it=0
                while it<5:
                    print('i_slice',i_slice,'it',it) 
                    for i in range(i_slice-2,0,-2):
                        if listSlice[i].get_stackIndex()==listSlice[i_slice].get_stackIndex(): #check if the previous slice is from the same volume
                            if set_r[i]==0 : #if the previous slice is well registered 
                                nbSlice1=i
                                dist1=abs(i_slice-nbSlice1)//2
                                okpre=True
                                break
                    for j in range(i_slice+2,nbSlice,2):
                        if set_r[j]==0: #if the previous slice is well registered
                            if listSlice[j].get_stackIndex()==listSlice[i_slice].get_stackIndex(): 
                                nbSlice2=j
                                dist2=abs(i_slice-nbSlice2)//2
                                okpost=True
                                break
                    if okpre==True and okpost==True: #if there is two close slice well-register, we do a mean between them
                        Slice1=listSlice[nbSlice1];Slice2=listSlice[nbSlice2]
                        ps1=Slice1.get_parameters().copy();ps2=Slice2.get_parameters().copy();
                        print('ps1:',ps1,'dist1:',dist1)
                        print('ps2:',ps2,'dist2:',dist2)
                        #MS1=rigidMatrix(ps1);MS2=rigidMatrix(ps2)
                        MS1=Slice1.get_transfo();MS2=Slice2.get_transfo()
                        RotS1=MS1[0:3,0:3];RotS2=MS2[0:3,0:3]
                        TransS1=MS1[0:3,3];TransS2=MS2[0:3,3]
                        Rot=computeMeanRotation(RotS1,dist1,RotS2,dist2)
                        Trans=computeMeanTranslation(TransS1,dist1,TransS2,dist2)
                        Mtot=eye(4)
                        Mtot[0:3,0:3]=Rot;Mtot[0:3,3]=Trans
                        #x0=ParametersFromRigidMatrix(Mtot)
                        center = Slice1.get_center()
                        center_mat = eye(4)
                        center_mat[0:3,3] = center
                        center_inv = eye(4)
                        center_inv[0:3,3] = -center
                        M_est = center_mat @ Mtot @ inv(Slice1.get_slice().affine) @ center_inv
                        x0 = ParametersFromRigidMatrix(M_est)
                        print('x0',x0)
                            #slicei.set_parameters(x0)
                            
                    elif okpre==True and okpost==False: #if there is only one close slice well-register, we take the parameters of this slice
                        Slice1=listSlice[nbSlice1]
                        x0=Slice1.get_parameters().copy()
                        print('x0',x0)
                        #slicei.set_parameters(p)
                    elif okpost==True and okpre==False: 
                        Slice2=listSlice[nbSlice2]
                        x0=Slice2.get_parameters().copy()
                        print('x0',x0)
                        #slicei.set_parameters(p)
                    else :
                        x0=slicei.get_parameters()
                        print('x0',x0)
                        break
                            
                #         print(i_slice,slicei.get_stackIndex(),slicei.get_index_slice(),okpre,okpost,x0)
            
            
                    #We do a multistart optimisation based on the new initialisation
                    #print('x0:',x0) 
                    multistart =zeros((6,6))
                    multistart[:5,:]=(line(-20,20,5)*ones((6,5))).T
                    #print(multistart)
                    index = array([0,1,2,3,4,5])
                    with Pool(processes=16) as p:
                        tmpfun=partial(multi_start,hyperparameters,i_slice,listSlice.copy(),grid_slices,set_o,x0,multistart,Vmx)
                        res=p.map(tmpfun,index)
                            
                        #
                    #cost=opti_res[0]
                    #x_opt=opti_res[3]
                    current_cost=cost_from_matrix(ge,gn,set_o,i_slice)
                    #p=array([p[0] for p in res])
                        
                    #print(p)
                    #p.append(cost)
                        
                    x=res
                    n = zeros(len(x))
                    p = zeros(len(x))
                    i_n=0
                    for x_tmp in x:
                        listSlice[i_slice].set_parameters(x_tmp)
                        getmp=ge.copy()
                        gntmp=gn.copy()
                        gitmp=gi.copy()
                        gutmp=gu.copy()
                        update_cost_matrix(i_slice,listSlice,getmp,gntmp,gitmp,gutmp)
                        grid_tmp=array([getmp,gntmp,gitmp,gutmp])
                        p[i_n]=cost_fct(x_tmp,i_slice,listSlice,grid_tmp,set_o,1,Vmx)
                        n[i_n]=somme(gitmp[i_slice,:]+somme(gitmp[:,i_slice]))
                        i_n=i_n+1
                        
                        
                    good_index = where(n>n_0)[0]
                    print('n_0',n_0,n[good_index])

                    p = p[good_index]
                    #si p est vide, on relance avec un simplex plus petit, sinon on garde les même valeurs.
                    if len(p)==0:
                        delta = hyperparameters[0]
                        d = delta/2
                        hyperparameters[0]=d
                        it=it+1
                        print('je suis ici')
                    else :
                        print('je suis la')
                        p = p.tolist()
                        print(x)
                        x=array([x])[0,:,:]
                        print(x.shape)
                        print(good_index)
                        x = x[good_index]
                        x = x.tolist()
                        n = n[good_index]
                        n = n.tolist()
                        mincost = min(p)
                        x_opt = x[p.index(mincost)]
                        n_1 = n[p.index(mincost)]
                        print('new_n',n_1)
                        print('corrected x',x_opt)
                        #print('i_slice',i_slice)
                        print('After:', x_opt)
                        print('mincost :',mincost)
                        x_opt = np.array([x_opt])
                        print(x_opt)
                        slicei.set_parameters(x_opt[0,:])
                            
                            
                        update_cost_matrix(i_slice,listSlice,ge,gn,gi,gu)
                            
                        grid_slices = array([ge,gn,gi,gu])
                        #print('Slice Cost After:',i_slice, oneSliceCost(ge,gn,set_o,i_slice))
                        costMse=compute_cost_from_matrix(ge,gn)
                        #print('cost :', costMse)
                        costDice=compute_cost_from_matrix(gi,gu)
                        
                        ge,gn,gi,gu,dicRes=algo_optimisation(hyperparameters,listSlice,set_r,set_o,grid_slices,dicRes,Vmx,10)  
                        grid_slices=array([ge,gn,gi,gu]) 
                        
                        new_set_o1,new_set_o2 = detect_misregistered_slice(listSlice,grid_slices,loaded_model)
                        print('La coupe est-elle corrigé ?',i_slice,new_set_o2[i_slice])
                        #set_o = logical_or(new_set_o1,new_set_o2)
                            
                        
                        break 

        #intersection=zeros((nbSlice,nbSlice))
        #union=zeros((nbSlice,nbSlice))
        #mW = matrixOfWeight(ge, gn,gi,gu, intersection, union, listSlice,t_inter,t_mse, t_inter, t_mse)
        ge,gn,gi,gu = compute_cost_matrix(listSlice)
        print(compute_cost_from_matrix(ge,gn))
        #grid_slices = array([ge,gn,gi,gu])
        #ge,gn,gi,gu,dicRes=algo_optimisation(hyperparameters,listSlice,set_o,set_o,grid_slices,dicRes)  
        
        #print("badly register : ", sum(new_set_o))
    
        new_set_o1,new_set_o2 = detect_misregistered_slice(listSlice,grid_slices,loaded_model)
        set_o = logical_or(new_set_o1,new_set_o2)
        print('mis-registered-before :', removeBadSlice(listSlice,set_o))
        print('mis-registered-after :', removeBadSlice(listSlice,new_set_o2))
        
        if all(new_set_o2 == set_r):
            break
            
        set_r = new_set_o2.copy()
    
    set_r = new_set_o2
    after_correction = sum(set_r)
    saved = before_correction - after_correction
    #print('slices saved with multistart :', saved)
    ge,gn,gi,gu=compute_cost_matrix(listSlice)
    costMse=compute_cost_from_matrix(ge,gn)
    costDice=compute_cost_from_matrix(gi,gu)
    updateResults(dicRes, ge, gn, gi, gu, costMse, costDice, listSlice, nbSlice)
    grid_slices = array([ge,gn,gi,gu])
        
    return grid_slices,set_o,dicRes

##deux fonctions pour voir si ça marche mais après il faudra factoriser
def correction_misregistered(listSlice,hyperparameters,set_o,set_r,grid_slices,dicRes,Vmx):
    """
    This function aims to correct mis-registered slices

    Parameters
    ----------
    listSlice : set of the slices, from the three volumes
    hyperparameters : parameters for optimisation, it includes : simplex initial size, xatol, fatol, epsilon, gaussian parameter, lamnda for dice
    set_o : set of outliers or mis-registered slices
    grid_slices : matrix of cost
    dicRes : results save for representation

    Returns
    -------
    grid_slices : matrix of cost
    dicRes : results save for representation

    """
    print(set_o.shape)
    ge = grid_slices[0,:,:]
    gn = grid_slices[1,:,:]
    gi = grid_slices[2,:,:]
    gu = grid_slices[3,:,:]
    before_correction = sum(set_o)
    print(before_correction)
     
    nbSlice = len(listSlice)     
    while True:
        
                
         #number of slices total
        print(set_o.shape)
        for i_slice in range(0,nbSlice):
            
        #     #print('before:',listSlice[i_slice].get_parameters())
        #     #print('Slice Cost Before:', oneSliceCost(ge,gn,set_o,i_slice))
             n_0=somme(gn[:,i_slice])+somme(gn[i_slice,:])
             okpre=False;okpost=False;nbSlice1=0;nbSlice2=0;dist1=0;dist2=0
             slicei=listSlice[i_slice]
             x_pre=slicei.get_parameters()
             print(set_o[i_slice])
             if set_o[i_slice]==1:
                 #print(i_slice)
                 for i in range(i_slice-2,0,-2):
                     if listSlice[i].get_stackIndex()==listSlice[i_slice].get_stackIndex(): #check if the previous slice is from the same volume
                         if set_r[i]==0 : #if the previous slice is well registered 
                             nbSlice1=i
                             dist1=abs(i_slice-nbSlice1)//2
                             okpre=True
                             break
                 for j in range(i_slice+2,nbSlice,2):
                     if set_r[j]==0: #if the previous slice is well registered
                         if listSlice[j].get_stackIndex()==listSlice[i_slice].get_stackIndex(): 
                              nbSlice2=j
                              dist2=abs(i_slice-nbSlice2)//2
                              okpost=True
                              break
                 if okpre==True and okpost==True: #if there is two close slice well-register, we do a mean between them
                      Slice1=listSlice[nbSlice1];Slice2=listSlice[nbSlice2]
                      ps1=Slice1.get_parameters().copy();ps2=Slice2.get_parameters().copy();
                      print('ps1:',ps1,'dist1:',dist1)
                      print('ps2:',ps2,'dist2:',dist2)
                      #MS1=rigidMatrix(ps1);MS2=rigidMatrix(ps2)
                      MS1=Slice1.get_transfo();MS2=Slice2.get_transfo()
                      RotS1=MS1[0:3,0:3];RotS2=MS2[0:3,0:3]
                      TransS1=MS1[0:3,3];TransS2=MS2[0:3,3]
                      Rot=computeMeanRotation(RotS1,dist1,RotS2,dist2)
                      Trans=computeMeanTranslation(TransS1,dist1,TransS2,dist2)
                      Mtot=eye(4)
                      Mtot[0:3,0:3]=Rot;Mtot[0:3,3]=Trans
                      #x0=ParametersFromRigidMatrix(Mtot)
                      center = Slice1.get_center()
                      center_mat = eye(4)
                      center_mat[0:3,3] = center
                      center_inv = eye(4)
                      center_inv[0:3,3] = -center
                      M_est = center_mat @ Mtot @ inv(Slice1.get_slice().affine) @ center_inv
                      x0 = ParametersFromRigidMatrix(M_est)
                      print('x0',x0)
                      #slicei.set_parameters(x0)
                      
                 elif okpre==True and okpost==False: #if there is only one close slice well-register, we take the parameters of this slice
                      Slice1=listSlice[nbSlice1]
                      x0=Slice1.get_parameters().copy()
                      print('x0',x0)
                      #slicei.set_parameters(p)
                 elif okpost==True and okpre==False: 
                      Slice2=listSlice[nbSlice2]
                      x0=Slice2.get_parameters().copy()
                      print('x0',x0)
                      #slicei.set_parameters(p)
                 else :
                      x0=slicei.get_parameters()
                      print('x0',x0)
                     
        #         print(i_slice,slicei.get_stackIndex(),slicei.get_index_slice(),okpre,okpost,x0)
    
    
                #We do a multistart optimisation based on the new initialisation
                 #print('x0:',x0) 
                 multistart = zeros((6,6))
                 multistart[:5,:]=(line(-20,20,5)*ones((6,5))).T
                 #print(multistart)
                 index = array([0,1,2,3,4,5])
                 with Pool(processes=16) as p:
                     tmpfun=partial(multi_start,hyperparameters,i_slice,listSlice.copy(),grid_slices,set_o,x0,multistart,Vmx)
                     res=p.map(tmpfun,index)
                 print(res)
                #
                #cost=opti_res[0]
                #x_opt=opti_res[3]
                 current_cost=cost_from_matrix(ge,gn,set_o,i_slice)
                 #p=array([p[0] for p in res])
                 
                 #print(p)
                 #p.append(cost)
                 
                 x=res#array([x[0] for x in res]) 
                 n = zeros(len(x))
                 p = zeros(len(x))
                 i_n=0
                 for x_tmp in x:
                     print(x_tmp)
                     listSlice[i_slice].set_parameters(x_tmp)
                     ge_tmp=ge.copy()
                     gn_tmp=gn.copy()
                     gi_tmp=gi.copy()
                     gu_tmp=gu.copy()
                     update_cost_matrix(i_slice,listSlice,ge_tmp,gn_tmp,gi_tmp,gu_tmp)
                     grid_tmp = array([ge_tmp,gn_tmp,gi_tmp,gu_tmp])
                     p[i_n]=cost_fct(x_tmp,i_slice,listSlice,grid_tmp,set_o,1,Vmx)
                     n[i_n]=somme(gn[i_slice,:]+somme(gn[:,i_slice]))
                     i_n=i_n+1
                 
            
                 mincost = min(p)
                 p=p.tolist()
                 x=x
                 x_opt = x[p.index(mincost)]
                 print('corrected x',x_opt)
                 #print('i_slice',i_slice)
                 print('After:', x_opt)
                 print('mincost :',mincost)
                 slicei.set_parameters(x_opt)
                
                
                 update_cost_matrix(i_slice,listSlice,ge,gn,gi,gu)
                
                
                 grid_slices = array([ge,gn,gi,gu])
                 #print('Slice Cost After:',i_slice, oneSliceCost(ge,gn,set_o,i_slice))
                 costMse=compute_cost_from_matrix(ge,gn)
                 #print('cost :', costMse)
                 costDice=compute_cost_from_matrix(gi,gu)
                 
                 new_set_o1,new_set_o2 = detect_misregistered_slice(listSlice,grid_slices,loaded_model)
                 set_o = logical_or(new_set_o1,new_set_o2) 
                 
                 hyperparameters[0]=4
                 ge,gn,gi,gu,dicRes=algo_optimisation(hyperparameters,listSlice,set_o,set_o,grid_slices,dicRes,Vmx,10)  
                 grid_slices=array([ge,gn,gi,gu])
                 
        
        #intersection=zeros((nbSlice,nbSlice))
        #union=zeros((nbSlice,nbSlice))
        #mW = matrixOfWeight(ge, gn,gi,gu, intersection, union, listSlice,t_inter,t_mse, t_inter, t_mse)
        ge,gn,gi,gu = compute_cost_matrix(listSlice)
        print(compute_cost_from_matrix(ge,gn))
        #grid_slices = array([ge,gn,gi,gu])
        ge,gn,gi,gu,dicRes=algo_optimisation(hyperparameters,listSlice,set_o,set_o,grid_slices,dicRes,Vmx,10)  
        
        #print("badly register : ", sum(new_set_o))
    
        new_set_o1,new_set_o2 = detect_misregistered_slice(listSlice,grid_slices,loaded_model)
        set_r = logical_or(new_set_o1,new_set_o2)
        print('mis-registered-before :', removeBadSlice(listSlice,set_o))
        print('mis-registered-after :', removeBadSlice(listSlice,set_r))

        
        
        if all(set_r == set_o):
            break
            
        set_o = set_r.copy()
    
    
    new_set_o1,new_set_o2 = detect_misregistered_slice(listSlice,grid_slices,loaded_model)
    set_o = logical_or(new_set_o1,new_set_o2)
    after_correction = sum(set_o)
    #set_o = new_set_o1
    saved = before_correction - after_correction
    #print('slices saved with multistart :', saved)
    ge,gn,gi,gu=compute_cost_matrix(listSlice)
    costMse=compute_cost_from_matrix(ge,gn)
    costDice=compute_cost_from_matrix(gi,gu)
    updateResults(dicRes, ge, gn, gi, gu, costMse, costDice, listSlice, nbSlice)
    grid_slices = array([ge,gn,gi,gu])
        
    return grid_slices,set_o,dicRes



def multi_start(hyperparameters,i_slice,listSlice,grid_slices,set_o,x0,valstart,Vmx,index):
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
    print(listSlice[i_slice].get_index_slice(),listSlice[i_slice].get_stackIndex())
    #print('index multistart:',valstart[index,:],'index:',index)
    x_opt = nelder_mead_optimisation(x,hyperparameters,listSlice,grid_slices,set_o,i_slice,Vmx)
    #cost=opti_res[0] #-1*opti_res[1]
    #x_opt=opti_res[3]

    
    return x_opt



def removeBadSlice(listSlice,set_o):
    """
    return a list of bad-registered slices and their corresponding stack
    """
    removelist=[]

   
    for i_slice in range(len(listSlice)):
      if set_o[i_slice]==1:
          removelist.append((listSlice[i_slice].get_stackIndex(),listSlice[i_slice].get_index_slice()))
   
    
    return removelist
        

@jit(nopython=True)
def cost_from_matrix(numerator,denumerator,set_o,i_slice):
    """
    Function to compute the cost, either mse or dice. Cost is computed only on well registered slices and depend on the slice we want ot make the optimisation on.
    """
    
    nbslice,nbslice = shape(numerator)
    
    grid_numerator_no_o = numerator.copy()
    grid_denumerator_no_o = denumerator.copy()
    
    set_outliers = 1-set_o
    set_outliers[i_slice]=1

    numerator = sum(grid_numerator_no_o)
    denumerator = sum(grid_denumerator_no_o)

    if denumerator==0:
        cost=nan
    cost=numerator/denumerator
    return cost
            

def cost_fct(x0 : array,
             k : int,
             listSlice : 'list[SliceObject]',
             cost_matrix : array,
             set_o : array,
             omega : float,
             Vmx):
    
    """
    function we want to minimize. 
    L = sum _ {k,k' ; k>k'} (S^2(k,k')) / sum _ {k,k' ; k>k'} N(k,k') + omega * ( sum _ {k,k' ; k>k'} M(k,k') /Vmx)
    """
    
    square_error_matrix = cost_matrix[0,:,:]
    nbpoint_matrix = cost_matrix[1,:,:]
    intersection_matrix = cost_matrix[2,:,:]
    union_matrix = cost_matrix[3,:,:]
    
    x = copy(x0) #copy to use bound in miminization fonction
  
    slice_k = listSlice[k]
    slice_k.set_parameters(x)
    
    update_cost_matrix(k,listSlice,square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix)
    cost_matrix = array([square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix])
   
    mse = cost_from_matrix(square_error_matrix,nbpoint_matrix,set_o,k)

    intersection = sum(intersection_matrix)
    intersection=intersection/Vmx
   
    cost = mse - omega*intersection
    
    return cost

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

