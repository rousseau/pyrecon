import array
from cmath import nan
import copy
from functools import partial
from multiprocessing import Pool
import pickle
from numpy import eye, ones, shape, where, zeros, array, sum 
from numba import jit
import numpy as np
from ..intersection import compute_cost_from_matrix, compute_cost_matrix, update_cost_matrix, updateResults, cost_fct,cost_fct2
from ..optimisation import algo_optimisation, nelder_mead_optimisation, newton_optimisation, translation_optimisation
from .feature import detect_misregistered_slice
from ..sliceObject import SliceObject

from ..tools import line, somme
from ..transformation import ParametersFromRigidMatrix, log_cplx, rigidMatrix, rotation_diff

from numpy.linalg import inv, eig
from scipy.linalg import expm

from rosi.simulation.validation import theorical_misregistered, tre_for_each_slices
from skimage.measure import block_reduce




#load_model = pickle.load(open('ROSI/rosi/registration/outliers_detection/my_new_model.pickle','rb'))

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
                        MS1=Slice1.get_estimatedTransfo();MS2=Slice2.get_estimatedTransfo()
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
                            
                #         print(i_slice,slicei.get_stackIndex(),slicei.get_indexSlice(),okpre,okpost,x0)
            
            
                    #We do a multistart optimisation based on the new initialisation
                    #print('x0:',x0) 
                    multistart =zeros((6,6))
                    multistart[:5,:]=(line(-20,20,10)*ones((6,10))).T
                    #print(multistart)
                    index = array([0,1,2,3,4,5])
                    with Pool(processes=5) as p:
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
                        #print(x_opt)
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

#find the best closer solutions
def good_neighboors(listOfSlice,k,set_r,nb_neigbhoors) :

    i_before=nb_neigbhoors*[None]
    i_after=nb_neigbhoors*[None]

    number_slices = len(listOfSlice)
    stack = listOfSlice[k].get_stackIndex()
    it=0
    while it<nb_neigbhoors:
        step = (it+1)*2
        for n1 in range(k-step,0,-step):
            if set_r[n1]==0 and stack==listOfSlice[n1].get_stackIndex() and not n1 in i_before :
                i_before[it]=n1
                break
        for n2 in range(k+step,number_slices,step) :
            if set_r[n2]==0 and stack==listOfSlice[n2].get_stackIndex() and not n2 in i_after:
                i_after[it]=n2
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


#correct the position of the slice with multi-start, using new initial parameters : 
def correct_with_new_initialisation(listOfSlices,hyperparameters,i_slice,new_parameters,cost_matrix,set_r,Vmx,optimisation):

    #print('optimisation :',optimisation)
    
    multistart=zeros((10,6))
    multistart[:9,:]=(np.random.rand(9,6)*20)-10#(line(-10,10,9)*ones((6,9))).T
    index=array([0,1,2,3,4,5,6,7,8,9])
    with Pool(processes=3) as p:
        tmpfun=partial(multi_start,hyperparameters,i_slice,listOfSlices,cost_matrix,set_r,new_parameters,multistart,Vmx,optimisation)
        res=p.map(tmpfun,index)

    cost_pre=10
    for x0 in res:
        
        #x0=x[0]
        
        cost = cost_fct(x0,i_slice,listOfSlices,cost_matrix,set_r,hyperparameters[4],Vmx)
        #print('The cost is :', cost, 'The cost_pre is :', cost_pre)
        #print(cost<cost_pre)
        print('the cost :', cost)

        if cost<cost_pre:
            x_min=x0
            cost_pre=cost
    print('the best cost :', cost_pre)
    square_error_matrix = cost_matrix[0,:,:]
    nbpoint_matrix = cost_matrix[1,:,:]
    intersection_matrix = cost_matrix[2,:,:]
    union_matrix = cost_matrix[3,:,:]
    listOfSlices[i_slice].set_parameters(x_min)        
    
    update_cost_matrix(i_slice,listOfSlices,square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix)
    cost_matrix = array([square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix])
   
    #print('The small cost is : ', cost_pre)

    return x_min


#correct the position of the slice with multi-start, using new initial parameters : 
def correct_with_new_initialisation3(listOfSlices,hyperparameters,i_slice,new_parameters,cost_matrix,set_r,Vmx,optimisation,W):

    #print('optimisation :',optimisation)
    
    multistart=zeros((10,6))
    multistart[:9,:]=(np.random.rand(9,6)*20)-10#(line(-10,10,9)*ones((6,9))).T
    index=array([0,1,2,3,4,5,6,7,8,9])
    with Pool(processes=3) as p:
        tmpfun=partial(multi_start,hyperparameters,i_slice,listOfSlices,cost_matrix,set_r,new_parameters,multistart,Vmx,optimisation,W)
        res=p.map(tmpfun,index)

    cost_pre=10
    for x0 in res:
        
        #x0=x[0]
        
        cost = cost_fct2(x0,i_slice,listOfSlices,cost_matrix,set_r,hyperparameters[4],Vmx,W)
        #print('The cost is :', cost, 'The cost_pre is :', cost_pre)
        #print(cost<cost_pre)
        print('the cost :', cost)

        if cost<cost_pre:
            x_min=x0
            cost_pre=cost
    print('the best cost :', cost_pre)
    square_error_matrix = cost_matrix[0,:,:]
    nbpoint_matrix = cost_matrix[1,:,:]
    intersection_matrix = cost_matrix[2,:,:]
    union_matrix = cost_matrix[3,:,:]
    listOfSlices[i_slice].set_parameters(x_min)        
    
    update_cost_matrix(i_slice,listOfSlices,square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix)
    cost_matrix = array([square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix])
   
    #print('The small cost is : ', cost_pre)

    return x_min

def correct_with_new_initialisation2(listOfSlices,hyperparameters,i_slice,new_parameters,cost_matrix,set_r,Vmx,optimisation):

    

    multistart=zeros((10,6))
    multistart[:9,:3]=(np.random.rand(9,3)*90)-45#(line(-10,10,9)*ones((6,9))).T
    multistart[:9,3:]=(np.random.rand(9,3)*90)-45
    index=array([0,1,2,3,4,5,6,7,8,9])#10,11,12,13,14,15,16,17,18,19])
    with Pool(processes=6) as p:
        tmpfun=partial(multi_start,hyperparameters,i_slice,listOfSlices,cost_matrix,set_r,new_parameters,multistart,Vmx,optimisation)
        res=p.map(tmpfun,index)

    cost_pre=10
    for x0 in res:
        
        #x0=x[0]
        
        cost = cost_fct(x0,i_slice,listOfSlices,cost_matrix,set_r,hyperparameters[4],Vmx)
        #print('The cost is :', cost, 'The cost_pre is :', cost_pre)
        #print(cost<cost_pre)
        print('the cost :', cost)

        if cost<cost_pre:
            x_min=x0
            cost_pre=cost
    print('the best cost :', cost_pre)
    square_error_matrix = cost_matrix[0,:,:]
    nbpoint_matrix = cost_matrix[1,:,:]
    intersection_matrix = cost_matrix[2,:,:]
    union_matrix = cost_matrix[3,:,:]
    listOfSlices[i_slice].set_parameters(x_min)        
    
    update_cost_matrix(i_slice,listOfSlices,square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix)
    cost_matrix = array([square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix])

    multistart=zeros((10,6))
    multistart[:9,:3]=(np.random.rand(9,3)*6)-3#(line(-10,10,9)*ones((6,9))).T
    multistart[:9,3:]=(np.random.rand(9,3)*6)-3
    index=array([0,1,2,3,4,5,6,7,8,9])#10,11,12,13,14,15,16,17,18,19])
    with Pool(processes=6) as p:
        tmpfun=partial(multi_start,hyperparameters,i_slice,listOfSlices,cost_matrix,set_r,x_min,multistart,Vmx,optimisation)
        res=p.map(tmpfun,index)

    cost_pre=10
    for x0 in res:
        
        #x0=x[0]
        
        cost = cost_fct(x0,i_slice,listOfSlices,cost_matrix,set_r,hyperparameters[4],Vmx)
        #print('The cost is :', cost, 'The cost_pre is :', cost_pre)
        #print(cost<cost_pre)
        print('the cost :', cost)

        if cost<cost_pre:
            x_min=x0
            cost_pre=cost


   
    #print('The small cost is : ', cost_pre)

    return x_min


##deux fonctions pour voir si ça marche mais après il faudra factoriser
""" def correction_misregistered(listSlice,hyperparameters,set_o,set_r,grid_slices,dicRes,Vmx):
    """
"""     This function aims to correct mis-registered slices

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
                      MS1=Slice1.get_estimatedTransfo();MS2=Slice2.get_estimatedTransfo()
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
                     
        #         print(i_slice,slicei.get_stackIndex(),slicei.get_indexSlice(),okpre,okpost,x0)
    
    
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
        
    return grid_slices,set_o,dicRes """



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

    #print('index : ',index)
    #x=x0[index]
    #print(x)
    #print(x)
    #print(listSlice[i_slice].get_indexSlice(),listSlice[i_slice].get_stackIndex())
    listSlice[i_slice].set_parameters(x)
    #print('index',index)
    #print('index multistart:',valstart[index,:],'index:',index)
    x_opt = newton_optimisation(hyperparameters,listSlice,cost_matrix,set_o,set_o,Vmx,i_slice,x,optimisation)
    #print('x_opt',x_opt)
    #cost=opti_res[0] #-1*opti_res[1]
    #x_opt=opti_res[3]

    
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
    print(x)
    #x=x0[index]
    #print(x)
    #print(x)
    #print(listSlice[i_slice].get_indexSlice(),listSlice[i_slice].get_stackIndex())
    listSlice[i_slice].set_parameters(x)
    #print('index',index)
    #print('index multistart:',valstart[index,:],'index:',index)
    x_opt = newton_optimisation(hyperparameters,listSlice,cost_matrix,set_o,set_o,Vmx,i_slice,x,optimisation)
    #print('x_opt',x_opt)
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
          removelist.append((listSlice[i_slice].get_stackIndex(),listSlice[i_slice].get_indexSlice()))
   
    
    return removelist
        

#@jit(nopython=True)
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
    print(denumerator)

    if denumerator==0:
        cost=nan
    cost=numerator/denumerator
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


#""""
#Plan pour la méthode de FSL  : 
#0. Créer une fonction qui permet d'optimiser uniquement les paramètres des translatiions -> OK
#1. Créer une fonction grid_search qui calcule la valeur de la fonction cout sur la grille et qui estime les paramètres de rotation associé. 
#2. Créer une fonction qui permet de calculer les valeurs des 5 minima locaux : pour ça : séléctionner les minima en fonction de leur entourage avec un filtre (probablement possible). Filtre jusqu'a ce que on ai plus que 5 valeurs
#3. Minimier sur ces 5 valeurs et choisir le cout le plus petit
#4. Verifier que on corrige bien la coupe.
#""""


def grid_search(x0,hyperparameters,listOfSlices,cost_matrix,set_r,set_o,Vmx,k,optimisation):

    a,b,c = x0[0:3]
    #x_opt,cost = translation_optimisation(hyperparameters,listOfSlices,cost_matrix,set_r,set_o,Vmx,k,x0)
    
    #print(x_t)
    vect_a = np.array([a,a-3,a-6,a+3,a+6])
    vect_b = np.array([b,b-3,b-6,b+3,b+6])
    vect_c = np.array([c,c-3,c-6,c+3,c+6])
    x=np.zeros(6)
    grid = np.zeros((len(vect_a),len(vect_b),len(vect_c)))
    estimated_translation = np.zeros((len(vect_a),len(vect_b),len(vect_c),3))

    for i in range(0,len(vect_a)):
        for j in range(0,len(vect_b)):
            for z in range(0,len(vect_c)):
                
                
                x[:3]=[vect_a[i],vect_b[j],vect_c[z]]
                x[3:6]=x0[3:6]
                x_opt,cost=translation_optimisation(hyperparameters,listOfSlices,cost_matrix,set_r,set_o,Vmx,k,x,optimisation)
                x_t=x_opt[3:6]
                x = np.array([vect_a[i],vect_b[j],vect_c[z],x_t[0],x_t[1],x_t[2]])
                print('x_grid',x)
                listOfSlices[k].set_parameters(x)
                cost=cost_fct(x,k,listOfSlices,cost_matrix,set_r,hyperparameters[4],Vmx)
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
                
                #print(i,j,z)
                current = padding[i,j,z]
                #print(current)

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
    vect = np.array([0,-3,-6,+3,+6])
    starts = []

    for i in range(0,len(index)):

        r = np.array([a+vect[index[i][0]],b+vect[index[i][1]],c+vect[index[i][2]]])
        t = estimated_translation[index[i][0],index[i][1],index[i][2],:]
        starts.append(np.concatenate((r,t)))

    return starts

def best_value(hyperparameters,listSlice,cost_matrix,set_o,Vmx,i_slice,starts,optimisation):

    cost = np.zeros(5)
    optimise_parameters = np.zeros((5,6))

    #for i in range(0,5) : multi_start2
    #    listSlice[i_slice].set_parameters(starts[i])
    #    optimise_parameters[i,:] = newton_optimisation(hyperparameters,listSlice,cost_matrix,set_o,set_o,Vmx,i_slice,starts[i])
    #    sqr_matrix = cost_matrix[0,:,:]
    #    nbpoint_matrix = cost_matrix[1,:,:]
    #    cost[i] = cost_from_matrix(sqr_matrix,nbpoint_matrix,set_o,i_slice)

    index = np.linspace(0,len(starts)-1,len(starts),dtype=int)
    with Pool(processes=6) as p:
        tmpfun=partial(multi_start2,hyperparameters,i_slice,listSlice,cost_matrix,set_o,starts,Vmx,optimisation)
        res=p.map(tmpfun,index)

    x_min=np.zeros(6)
    cost_pre=10
    for x0 in res:
        
        #x0=x[0]
        
        cost = cost_fct(x0,i_slice,listSlice,cost_matrix,set_o,hyperparameters[4],Vmx)
        #print('The cost is :', cost, 'The cost_pre is :', cost_pre)
        #print(cost<cost_pre)
        print('the cost :', cost)

        if cost<cost_pre:
            x_min=x0
            cost_pre=cost
    
    print('the best cost :', cost_pre)
    square_error_matrix = cost_matrix[0,:,:]
    nbpoint_matrix = cost_matrix[1,:,:]
    intersection_matrix = cost_matrix[2,:,:]
    union_matrix = cost_matrix[3,:,:]
    listSlice[i_slice].set_parameters(x_min)        
    
    update_cost_matrix(i_slice,listSlice,square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix)
    cost_matrix = array([square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix])

    #min = np.argmin(cost)
    #x_opt=optimise_parameters[min,:]

    return x_min,cost_pre



def correct_misregisterd(listOfSlice, set_r, matrix, hyperparameters, Vmx,loaded_model) : 

    number_slice = len(listOfSlice)
    while True :
        for i_slice in range(0,number_slice):   
                if set_r[i_slice]==1:

                    i_before,i_after=good_neighboors(listOfSlice,i_slice,set_r)
                    new_x = estimate_new_position(listOfSlice,i_before,i_slice,i_after)
                    estimated_x = correct_with_new_initialisation(listOfSlice,hyperparameters,i_slice,new_x,matrix,np.zeros(number_slice),Vmx)
                    listOfSlice[i_slice].set_parameters(estimated_x)
                    
        
        new_set =  detect_misregistered_slice(listOfSlice,matrix,loaded_model)
        new_set = np.logical_and(set_r,new_set)
        
        if np.all(set_r==new_set):
            break
        else:
            set_r=new_set.copy()


    return set_r


def correct_slice_with_theorical_error(set_r,listOfSlice,hyperparameters,optimisation,Vmx,matrix,listFeatures,transfolist,listNomvt):

    number_slice = len(listOfSlice)
    set_r = theorical_misregistered(listOfSlice,listFeatures,transfolist)
    
    tre_for_each_slices(listNomvt,listOfSlice,listFeatures,transfolist,[])
    tre = [e.get_error() for e in listFeatures]
    tre = np.array(tre) 
    mask = [e.get_mask_proportion()[0] for e in listFeatures]
    mask = np.array(mask)
    index = np.array([np.where(set_r==1)])[0][0]
    ordered_tre = np.argsort(-mask[index],axis=-1)

    
    while True :

        
        for iv in ordered_tre:  
            i_slice = index[iv] 
            if set_r[i_slice]==1 and mask[iv]>0.1:
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
                    print((new_x))
                    estimated_x=np.zeros((len(new_x),6))
                    cost=np.zeros((len(new_x)))
                    for it in range(0,len(new_x)):
                        hyperparameters[4]=1
                        grid,estimated_trans = grid_search(new_x[it,:],hyperparameters,listOfSlice,matrix,set_r,set_r,Vmx,i_slice,optimisation)
                        minimum = local_minimum(grid)
                        starts = find_minimum(new_x[it,:],minimum,estimated_trans)
                        estimated_x[it,:],cost[it] = best_value(hyperparameters,listOfSlice,matrix,set_r,Vmx,i_slice,starts,optimisation)
                    
                    min_index = np.argmin(cost)
                    min = estimated_x[min_index,:]
                    listOfSlice[i_slice].set_parameters(min)
                    print('estimated :',estimated_x)
            
        new_set = theorical_misregistered(listOfSlice,listFeatures,transfolist)
        new_set = np.logical_and(set_r,new_set)
        #i+=1
        if np.all(set_r==new_set):
            break
        else:
            set_r=new_set.copy()

    tre_for_each_slices(listNomvt,listOfSlice,listFeatures,transfolist,[])
    tre = [e.get_error() for e in listFeatures]
    tre_new = np.array(tre) 
    #algo_optimisation(hyperparameters,listOfSlice,set_r,set_r,matrix,None,Vmx,10,optimisation) 

    return set_r,tre_new

def correct_slice(set_r,listOfSlice,hyperparameters,optimisation,Vmx,matrix,listFeatures,loaded_model):

    mask = [e.get_mask_proportion()[0] for e in listFeatures]
    mask = np.array(mask)
    index = np.array([np.where(set_r==1)])[0][0]
    ordered_slice = np.argsort(-mask[index],axis=-1)

    
    while True :

        for iv in ordered_slice:  
            i_slice = index[iv] 
            if set_r[i_slice]==1 and mask[iv]>0.1:
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
                    print((new_x))
                    estimated_x=np.zeros((len(new_x),6))
                    cost=np.zeros((len(new_x)))
                    for it in range(0,len(new_x)):
                        hyperparameters[4]=1
                        grid,estimated_trans = grid_search(new_x[it,:],hyperparameters,listOfSlice,matrix,set_r,set_r,Vmx,i_slice,optimisation)
                        minimum = local_minimum(grid)
                        starts = find_minimum(new_x[it,:],minimum,estimated_trans)
                        estimated_x[it,:],cost[it] = best_value(hyperparameters,listOfSlice,matrix,set_r,Vmx,i_slice,starts,optimisation)
                    
                    min_index = np.argmin(cost)
                    min = estimated_x[min_index,:]
                    listOfSlice[i_slice].set_parameters(min)
                    print('estimated :',estimated_x)
            
            
        new_set = detect_misregistered_slice(listOfSlice,matrix,loaded_model)
        new_set = np.logical_and(set_r,new_set)
        #i+=1
        if np.all(set_r==new_set):
            break
        else:
            set_r=new_set.copy()
        
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
            if theta<0.2: #if we have more than 11 degrees of difference
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


    