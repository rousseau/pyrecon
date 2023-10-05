
from time import perf_counter
from numpy import float, array,max
from numpy import linspace, where, zeros, sqrt, asarray

from .optimisation import algo_optimisation
from .intersection import compute_cost_from_matrix, compute_cost_matrix
from .outliers_detection.feature import detect_misregistered_slice
from .outliers_detection.multi_start import correction_misregistered, removeBadSlice
from .tools import computeMaxVolume

import pickle

load_model = pickle.load(open('ROSI/rosi/registration/outliers_detection/my_new_model.pickle','rb'))

def global_optimisation(hyperparameters,listSlice,ablation):

    print('taille list : ',len(listSlice))    
    tps = perf_counter()

    hyperparameters = asarray(hyperparameters,dtype=float)

    """
    Compute the optimised parameters for each slice. At the end of the function parameters of each slice must be the optimised parameters. The function returns the evolution of the registration on each iterarion.
    
    Input
    
    listSlice : list of sliceObjects
        list of slices from all staks

    Returns
    
    dicRes : A dictionnay wich contains informations on the evolution of the registration during all iteration (ex : evolution of MSE, evolution of DICE, evolution of parameters ....)
    rejectedSlice : list of rejected slices and their corresponding stack
    
    """
    print('ablattion 1 :',ablation[0])
    print('ablation 2 :',ablation[1] )
    #Initialisation
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
    gridError,gridNbpoint,gridInter,gridUnion=compute_cost_matrix(listSlice) 
    EvolutionGridError.extend(gridError.copy())
    dicRes["evolutiongriderror"] = EvolutionGridError
    EvolutionGridNbpoint.extend(gridNbpoint.copy())
    dicRes["evolutiongridnbpoint"] = EvolutionGridNbpoint
    EvolutionGridInter.extend(gridInter.copy())
    dicRes["evolutiongridinter"] = EvolutionGridInter
    EvolutionGridUnion.extend(gridUnion.copy())
    dicRes["evolutiongridunion"] = EvolutionGridUnion
    costMse=compute_cost_from_matrix(gridError, gridNbpoint)
    costDice=compute_cost_from_matrix(gridInter,gridUnion)
    
    EvolutionError.append(costMse)
    dicRes["evolutionerror"] = EvolutionError
    print('The MSE before optimization is :', costMse)
    EvolutionDice.append(costDice)
    dicRes["evolutiondice"] = EvolutionDice
    print('The DICE before optimisation is :', costDice)
    

    nbslice=len(listSlice)
    set_o=zeros(nbSlice)
    

    grid_slices=array([gridError,gridNbpoint,gridInter,gridUnion])
    set_r=zeros(nbSlice)
  
    if ablation[0]=='no_gaussian':
         hyperparameters[5] = 0
    elif ablation[0]=='no_dice':
         hyperparameters[4] = 0
    #hyperparameters[4]=0 #no dice
    #hyperparameters[5]=0 #no gaussian filtering -> we find out it have no interest in the algorithm

    nbstack=max([slicei.get_stackIndex() for slicei in listSlice])
    #print(nbstack)
    res=zeros(nbstack+1)
    nbslice=len(listSlice)
  

    Vmx=computeMaxVolume(listSlice)
    #if hyperparameters[5] != 0 :
    print('--------------motion estimation with gaussian filtering and dice-----------------')
    print('---First Iteration -----')
    tpsg1 =perf_counter()
    print('Hyperparameters :')
    print('Initial Simplex :',hyperparameters[0])
    print('Final Simplex :',hyperparameters[1])
    print('Epsilon :',sqrt(6*hyperparameters[3]**2))
    print('Dice weight :',hyperparameters[4])
    print('Gaussian blurring :',hyperparameters[5])
    print('taille list :',len(listSlice))
    new_hyperparameters = array([hyperparameters[0]/8,hyperparameters[1],hyperparameters[2],sqrt(6*hyperparameters[3]**2),hyperparameters[4],hyperparameters[5]])
    ge,gn,gi,gu,dicRes=algo_optimisation(new_hyperparameters,listSlice,set_o,set_r,grid_slices,dicRes,Vmx,10)
    grid_slices=array([ge,gn,gi,gu])
    set_r=zeros(nbSlice)
    #     #Second Step : sigma = 2.0, d=b/2, x_opt, omega
    #     #epsilon = sqrt(6*(erreur/2)^2)
    tpeg1 = perf_counter()
    tp = tpeg1 - tpsg1
    hours = int(tp//(60*60))
    minutes = int((tp%(60*60))//60)
    secondes = int(tp%60)
    print('Fist Iteration was done in %s hours %s minutes and %s seconds' %(hours,minutes,secondes))
       
    print('----Second Iteration----')
    tpsg2 = perf_counter()
    print('Hyperparameters :')
    print('Initial Simplex :',hyperparameters[0]/2)
    print('Final Simplex :',hyperparameters[1]/2)
    print('Epsilon :',sqrt(6*(hyperparameters[3]/2)**2))
    print('Dice Weight :',hyperparameters[4])
    print('Gaussien blurring :',hyperparameters[5]/2)
    new_hyperparameters = array([hyperparameters[0]/8,hyperparameters[1]/2,hyperparameters[2],sqrt(6*(hyperparameters[3]/2)**2),hyperparameters[4],hyperparameters[5]/2])
    ge,gn,gi,gu,dicRes=algo_optimisation(new_hyperparameters,listSlice,set_o,set_r,grid_slices,dicRes,Vmx,10)
    grid_slices=array([ge,gn,gi,gu])
    tpeg2 = perf_counter()
    tp = tpeg2 - tpsg2
    hours = int(tp//(60*60))
    minutes = int((tp%(60*60))//60)
    secondes = int(tp%60)
    print('Second Iteration was done in %s hours %s minutes and %s seconds' %(hours,minutes,secondes))
    # #Third step : sigma = 0.0, d=b/4, x_opt, omega
    # #epsilon = sqrt(6*(erreur)^2/4)
    #if hyperparameters[4] != 0 :
    print('---motion estimation without gaussian filtering but with dice----')
    tpsd = perf_counter()
    print('Hyperparameters :')
    print('Initial Simplex :',hyperparameters[0]/4)
    print('FInal Simplex :',hyperparameters[1]/4)
    print('Epsilon :',sqrt(6*(hyperparameters[3]/4)**2))
    print('Dice Weight :',hyperparameters[4])
    print('Gaussien blurirng :',0)
    new_hyperparameters = array([hyperparameters[0]/8,hyperparameters[1]/4,hyperparameters[2],sqrt(6*(hyperparameters[3]/4)**2),hyperparameters[4],0])
    ge,gn,gi,gu,dicRes=algo_optimisation(new_hyperparameters,listSlice,set_o,set_r,grid_slices,dicRes,Vmx,10)
    grid_slices=array([ge,gn,gi,gu])
    tped = perf_counter()
    tp = tped - tpsd
    hours = int(tp//(60*60))
    minutes = int((tp%(60*60))//60)
    secondes = int(tp%60)
    print('Third Iteration was done in %s hours %s minutes and %s seconds' %(hours,minutes,secondes))
    # #Fourth step : sigma = 0.0 , d=b/8, x_opt, omega
    # #epsilon = sqrt(6*(erreur)^2/8)
    print('---motion estmisation without gaussien filtering and without dice----')
    tpsnod = perf_counter()
    print('Hyperparameters :')
    print('Initial simplex :',hyperparameters[0]/8)
    print('Final simplex :',hyperparameters[1]/8)
    print('Epsilon :',sqrt(6*(hyperparameters[3]/8)**2))
    print('Dice Weight :',0)
    print('Gaussien blurring :', 0)
    if ablation[1]=="no_dice":
        new_hyperparameters = array([hyperparameters[0]/8,hyperparameters[1]/8,hyperparameters[2],sqrt(6*(hyperparameters[3]/8)**2),hyperparameters[4],0])
    else :
        new_hyperparameters = array([hyperparameters[0]/8,hyperparameters[1]/8,hyperparameters[2],sqrt(6*(hyperparameters[3]/8)**2),0,0])
    ge,gn,gi,gu,dicRes=algo_optimisation(new_hyperparameters,listSlice,set_o,set_r,grid_slices,dicRes,Vmx,10)
    grid_slices=array([ge,gn,gi,gu])
    tpenod = perf_counter()
    tp = tpenod - tpsnod
    hours = int(tp//(60*60))
    minutes = int((tp%(60*60))//60)
    secondes = int(tp%60)
    print('Fourth Iteration was done in %s hours %s minutes and %s seconds' %(hours,minutes,secondes))
    
    ge=grid_slices[0,:,:]
    gn=grid_slices[1,:,:]
    gi=grid_slices[2,:,:]
    gu=grid_slices[3,:,:]
    
    #union=zeros((nbSlice,nbSlice))
    #intersection=zeros((nbSlice,nbSlice))
    nbSlice=len(listSlice)
    #union=zeros((nbSlice,nbSlice))
    #intersection=zeros((nbSlice,nbSlice))

    #set_o = zeros(nbSlice)
    #print("badly register : ",sum(set_o))
    #print(set_o)
    #ge,gn,gi,gu,dicRes=algo_optimisation(new_hyperparameters,listSlice,set_o,set_o,grid_slices,dicRes)  
    #grid_slices=array([ge,gn,gi,gu])
      
    if ablation[0]!='no_multistart':
    #set_o1,set_o2 = detect_misregistered_slice(listSlice, grid_slices, loaded_model)
    #set_r = logical_or(set_o1,set_o2) 
    #print(set_o)
        #before = removeBadSlice(listSlice,set_o)
        new_hyperparameters = array([hyperparameters[0],hyperparameters[1],hyperparameters[2],sqrt(6*hyperparameters[3]**2),hyperparameters[4],0])
        #grid_slices,set_r,dicRes=correction_out_images(listSlice,new_hyperparameters,set_o1.copy(),set_r,grid_slices,dicRes)
        print('---------------Outliers\'detection -------')
        set_o1,set_o2 = detect_misregistered_slice(listSlice, grid_slices, loaded_model)
        set_o = np.logical_or(set_o1,set_o2)
        ge,gn,gi,gu,dicRes=algo_optimisation(new_hyperparameters,listSlice,set_o,set_o,grid_slices,dicRes,Vmx,10)
    #set_r = logical_or(set_o1,set_o2) 
   
        #grid_slices,set_r,dicRes=correction_out_images(listSlice,new_hyperparameters,set_o.copy(),set_o2,grid_slices,dicRes)
        #set_o1,set_o2 = detect_misregistered_slice(listSlice, grid_slices, loaded_model)
        #set_o = np.logical_or(set_o1,set_o2)
        grid_slices,set_r,dicRes=correction_misregistered(listSlice,new_hyperparameters,set_o.copy(),set_o.copy(),grid_slices,dicRes,Vmx)
        ge,gn,gi,gu,dicRes=algo_optimisation(new_hyperparameters,listSlice,set_r,set_r,grid_slices,dicRes,Vmx,10)
        rejectedSlices=removeBadSlice(listSlice, set_r)
        
    else:
        rejectedSlices=[]
    #rejectedSlices=[]
    tpe = perf_counter()
    tp = tpe-tps
    hours = int(tp//(60*60))
    minutes = int((tp%(60*60))//60)
    secondes = int(tp%60)
    print('The global optimisation was done in %s hours %s minutes and %s seconds' %(hours,minutes,secondes))
    
    return dicRes, rejectedSlices
    
