
from time import perf_counter
from numpy import float64, array,max
from numpy import zeros, sqrt, asarray

from .outliers_detection.outliers import sliceFeature

from .optimisation import algo_optimisation
from .intersection import compute_cost_from_matrix, compute_cost_matrix
from .outliers_detection.feature import detect_misregistered_slice, update_features
from .outliers_detection.multi_start import  removeBadSlice, correct_slice
from .tools import computeMaxVolume
import os

import pickle

root = os.getcwd()
#load_model = pickle.load(open('ROSI/my_model_nmse_inter_dice_std.pickle','rb'))

def global_optimisation(listSlice,optimisation='Nelder-Mead',classifier='ROSI/my_model_nmse_inter_dice.pickle',multi_start=0,hyperparameters={'ds':4,'fs':0.25,'T':2,'omega':0}):
    """
    Compute the optimised parameters for each slice. At the end of the function parameters of each slice must be the optimised parameters. The function returns the evolution of the registration on each iterarion.
    
    Input
    
    listSlice : list of sliceObjects
        list of slices from all staks

    Returns
    
    dicRes : A dictionnay wich contains informations on the evolution of the registration during all iteration (ex : evolution of MSE, evolution of DICE, evolution of parameters ....)
    rejectedSlice : list of rejected slices and their corresponding stack
    
    """

    print(classifier)
    tps = perf_counter()
    ##Preprocessing : 
    ##Remove slices with a really small mask proportion
    listErrorSlice = [sliceFeature(slicei.get_stackIndex(),slicei.get_indexSlice()) for slicei in listSlice]
    
    squarre_error,number_point,intersection,union=compute_cost_matrix(listSlice) 
    update_features(listSlice,listErrorSlice,squarre_error,number_point,intersection,union)
    it=0
    while it < len(listErrorSlice):
            if listErrorSlice[it].get_mask_proportion()<0.1:
                    del listErrorSlice[it]
                    del listSlice[it]
            else:
                    it+=1
        
    
    squarre_error,number_point,intersection,union=compute_cost_matrix(listSlice) 

 
    #Results initialisation
    nbSlice=len(listSlice) 
    EvolutionError=[] 
    EvolutionDice=[]
    EvolutionGridError=[] 
    EvolutionGridNbpoint=[]
    EvolutionGridInter=[]
    EvolutionGridUnion=[]
    EvolutionParameters=[] 
    Previous_parameters=zeros((6,nbSlice))
    EvolutionTransfo=[]
    
   
    dicRes={}
    for i in range(nbSlice): 
        slicei=listSlice[i]
        EvolutionParameters.extend(slicei.get_parameters())
        Previous_parameters[:,i]=slicei.get_parameters()
        EvolutionTransfo.extend(slicei.get_estimatedTransfo())
        
    dicRes["evolutionparameters"] = EvolutionParameters
    dicRes["evolutiontransfo"] = EvolutionTransfo
    squarre_error,number_point,intersection,union=compute_cost_matrix(listSlice) 
    EvolutionGridError.extend(squarre_error.copy())
    dicRes["evolutiongriderror"] = EvolutionGridError
    EvolutionGridNbpoint.extend(number_point.copy())
    dicRes["evolutiongridnbpoint"] = EvolutionGridNbpoint
    EvolutionGridInter.extend(intersection.copy())
    dicRes["evolutiongridinter"] = EvolutionGridInter
    EvolutionGridUnion.extend(union.copy())
    dicRes["evolutiongridunion"] = EvolutionGridUnion
    costMse=compute_cost_from_matrix(squarre_error, number_point)
    costDice=compute_cost_from_matrix(intersection,union)
    
    EvolutionError.append(costMse)
    dicRes["evolutionerror"] = EvolutionError
    print('The MSE before optimization is :', costMse)
    EvolutionDice.append(costDice)
    dicRes["evolutiondice"] = EvolutionDice
    print('The DICE before optimisation is :', costDice)
    print('\n')
    

    nbslice=len(listSlice)
    set_o=zeros(nbSlice)
    
    grid_slices=array([squarre_error,number_point,intersection,union])
    set_r=zeros(nbSlice)
  
    #nbstack=max([slicei.get_stackIndex() for slicei in listSlice])

    Vmx=computeMaxVolume(listSlice)

    v = [1,2,4,8]
 
    iteration = range(0,len(v))

    for iter in iteration:

        print('Iteration %d : '%(iter)) 
        tpsg1 =perf_counter()
        if optimisation=="Nelder-Mead":
            print('ds = ',hyperparameters["ds"]/v[iter])
            print('fs =',hyperparameters["fs"]/v[iter])
        print('th =',sqrt(6*(hyperparameters["T"]/v[iter])**2))
        

        if iter==(len(v)-1):
              omega=0
        else :
              omega=hyperparameters["omega"]
              
        print('omega =',hyperparameters["omega"])
        
        new_hyperparameters={'ds':hyperparameters['ds']/v[iter],'fs':hyperparameters['fs']/v[iter],'T':sqrt(6*(hyperparameters["T"]/v[iter])**2),'omega':omega}
        squarre_error,number_point,intersection,union,dicRes=algo_optimisation(new_hyperparameters,listSlice,set_o,set_r,grid_slices,dicRes,Vmx,10,optimisation)
        grid_slices=array([squarre_error,number_point,intersection,union])
        set_r=zeros(nbSlice)

        tpeg1 = perf_counter()
        tp = tpeg1 - tpsg1
        hours = int(tp//(60*60))
        minutes = int((tp%(60*60))//60)
        secondes = int(tp%60)
        print('Iteration %d was done in %s hours %s minutes and %s seconds' %(iter,hours,minutes,secondes))
        print('\n')
        

        squarre_error=grid_slices[0,:,:]
        number_point=grid_slices[1,:,:]
        intersection=grid_slices[2,:,:]
        union=grid_slices[3,:,:]
    

    new_hyperparameters={'ds':hyperparameters['ds']/v[iter],'fs':hyperparameters['fs']/v[iter],'T':sqrt(6*(hyperparameters["T"]/v[iter])**2),'omega':hyperparameters['omega']/v[iter]}
    print('Outliers\'detection  :')
    load_model = pickle.load(open(classifier,'rb'))
    grid_slices=array([squarre_error,number_point,intersection,union])
    set_o = detect_misregistered_slice(listSlice, grid_slices, load_model,0.5)
    

    if multi_start==0:
        set_r = detect_misregistered_slice(listSlice, grid_slices, load_model,0.8)
        print(sum(set_r),' slices are poorly registered')
        set_o = correct_slice(set_o,set_r,listSlice,new_hyperparameters,'Nelder-Mead',Vmx,grid_slices,listErrorSlice,load_model,dicRes,0.8)
    rejectedSlices=removeBadSlice(listSlice, set_o)
    tpe = perf_counter()
    tp = tpe-tps
    hours = int(tp//(60*60))
    minutes = int((tp%(60*60))//60)
    secondes = int(tp%60)
    print('The global optimisation was done in %s hours %s minutes and %s seconds' %(hours,minutes,secondes))
    
    return dicRes, rejectedSlices
    

