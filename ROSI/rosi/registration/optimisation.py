from random import shuffle
from .tools import somme, apply_gaussian_filtering
from scipy.optimize import minimize,least_squares
from numpy import linspace, ones, eye, concatenate, array
from .intersection import compute_cost_matrix, update_cost_matrix, compute_cost_from_matrix, cost_fct, updateResults, cost_from_matrix, cost_multi_start
from .sliceObject import SliceObject
from numpy.linalg import norm




#load_model = pickle.load(open('ROSI/rosi/registration/outliers_detection/my_new_model.pickle','rb'))

def nelder_mead_optimisation(hyperparameters : dict,
                            listOfSlices : 'list[SliceObject]',
                            cost_matrix : array,
                            set_r : array,
                            set_o : array,
                            Vmx : float,
                            k : int,
                            ablation : str) -> array(6):
    
    """
    The function used to optimise the cost_function. It applied the Nelder-Mead to optimise parameters of one slice. 
    """
    
    #print('taille list :',len(listOfSlices))
    #the hyperparameters of the function are : the inital size of the simplex, tolerance on x, tolerance in f and omega
    registration_state=0
    

    #previous_parameters
    #print('shape list :', len(listOfSlices))
    x0 = listOfSlices[k].get_parameters()
    
    #initial simplex
    delta=hyperparameters['ds']
    p=(ones((x0.size,x0.size))*x0.T)+eye(6)*delta
    initial_s=concatenate(([x0],p),axis=0)

    x_tol = hyperparameters['fs']
    omega = hyperparameters['omega']
    epsilon = hyperparameters['T']
    maxiter=2000

    #optimisation with Nelder-Mead
    ftol=1e-4
    #print('delta',delta,'xatol',x_tol)
    if ablation == 'Nelder-Mead':
        NM = minimize(cost_fct,x0,args=(k,listOfSlices,cost_matrix,set_o,omega,Vmx),method='Nelder-Mead',options={"disp" : False, "maxiter" : maxiter, "maxfev":1e3, "fatol" : ftol,"xatol" : x_tol, "initial_simplex" : initial_s , "adaptive" :  False})
        x_opt = NM.x
    elif ablation != "LM":
        NM = minimize(cost_fct,x0,args=(k,listOfSlices,cost_matrix,set_o,omega,Vmx),method=ablation,options={"disp" : False})
        x_opt = NM.x
    else : 
        NM = least_squares(cost_fct, x0,args=(k,listOfSlices,cost_matrix,set_o,omega,Vmx),xtol=1e-10,ftol=1e-10)
        x_opt = NM.x
    #print(NM)
    #update the parameters of the slice and the cost
    #x_opt = NM.x 
    #print(NM.fun)
    listOfSlices[k].set_parameters(x_opt)
    square_error_matrix,nboint_matrix,intersection_matrix,union_matrix=[cost_matrix[i,:,:] for i in range(0,4)]
    update_cost_matrix(k,listOfSlices,square_error_matrix,nboint_matrix,intersection_matrix,union_matrix)
    cost_matrix = array([square_error_matrix,nboint_matrix,intersection_matrix,union_matrix])
    
    if norm(x0-x_opt)<epsilon:
        registration_state=1
    
   
    return x_opt,registration_state


def multi_optimisation(hyperparameters : dict,
                            listOfSlices : 'list[SliceObject]',
                            cost_matrix : array,
                            set_o : array,
                            Vmx : float,
                            k : int,
                            x0 : array,
                            optimisation) -> array(6):
    
    """
    The function used to optimise the cost_function. It applied the Nelder-Mead to optimise parameters of one slice. 
    """
    

    delta=hyperparameters['ds']
    x_tol=hyperparameters['fs']
    omega=hyperparameters['omega']
    epsilon=hyperparameters['T']
   

    ftol=1e-4
  
    print(optimisation)
    #print(dtype(cost_matrix[0][0,0]))
    if optimisation == 'Nelder-Mead':
        p=(ones((x0.size,x0.size))*x0.T)+eye(6)*delta
        initial_s=concatenate(([x0],p),axis=0)
        NM = minimize(cost_fct,x0,args=(k,listOfSlices,cost_matrix,set_o,omega,Vmx),method='Nelder-Mead',options={"disp" : True, "maxiter" : 1000, "maxfev":1e3, "fatol" : ftol,"xatol" : x_tol, "initial_simplex" : initial_s , "adaptive" :  False, "return_all" :True })
        x_opt = NM.x
    elif optimisation != "LM":
       
    
        NM = minimize(cost_fct,x0,args=(k,listOfSlices,cost_matrix,set_o,omega,Vmx),method=optimisation,jac=None,options={"disp" : False, "return_all" : False, "gtol" : 6*(1e-3), "eps" : 1e-4})
  
        x_opt = NM.x
        print(NM)
        print(NM.message)
        print(x_opt)
        #else :
        #    x_opt=x0
            
        #print(NM.success)
    else : 
        NM = least_squares(cost_fct, x0,args=(k,listOfSlices,cost_matrix,set_o,omega,Vmx),xtol=1e-10,ftol=1e-10)
        x_opt = NM.x
        print(NM.message)
  
    listOfSlices[k].set_parameters(x_opt)
    square_error_matrix,nboint_matrix,intersection_matrix,union_matrix=[cost_matrix[i,:,:] for i in range(0,4)]
    update_cost_matrix(k,listOfSlices,square_error_matrix,nboint_matrix,intersection_matrix,union_matrix)
    cost_matrix = array([square_error_matrix,nboint_matrix,intersection_matrix,union_matrix])
    print(x_opt)

    return x_opt




def translation_optimisation(hyperparameters : array(6),
                            listOfSlices : 'list[SliceObject]',
                            cost_matrix : array,
                            set_o : array,
                            Vmx : float,
                            k : int,
                            x0 : array,
                            optimisation) -> array(6):
    
    """
    The function used to optimise the cost_function. It applied the Nelder-Mead to optimise parameters of one slice. 
    """
    
    #print('taille list :',len(listOfSlices))
    #the hyperparameters of the function are : the inital size of the simplex, tolerance on x, tolerance in f and omega
    delta=hyperparameters['ds']
    x_tol=hyperparameters['fs']
    omega=hyperparameters['omega']

    xr = x0[0:3]
    xt = x0[3:6]
    #print('xt init :', xt)

    #initial simplex
    p=(ones((xt.size,xt.size))*xt.T)+eye(3)*delta
    initial_s=concatenate(([xt],p),axis=0)

    
    #optimisation with Nelder-Mead
    ftol=1e-4
    #print('delta',delta,'xatol',x_tol)
    if optimisation=='Nelder-Mead' :
        NM = minimize(cost_multi_start,xt,args=(xr,k,listOfSlices,cost_matrix,set_o),method='Nelder-Mead',options={"disp" : False, "maxiter" : 1000, "maxfev": 1e3, "fatol" : ftol,"xatol" : x_tol, "initial_simplex" : initial_s, "adaptive" :  False, "return_all" : False})
        x_est = NM.x
    elif optimisation != 'LM':
        NM = minimize(cost_multi_start,xt,args=(xr,k,listOfSlices,cost_matrix,set_o),method=optimisation,jac=None,options={"disp" : False,"gtol" : 1e-10, "eps" : 1e-10})
        x_est = NM.x
        print(NM.message)

    else :
        print('LM')
        NM = least_squares(cost_multi_start, xt,args=(xr,k,listOfSlices,cost_matrix,set_o),xtol=1e-10,ftol=1e-10)
        x_est = NM.x
        print(NM.message)
    #update the parameters of the slice and the cost
    
    #print('x_est',x_est)
    x_opt = array([xr[0],xr[1],xr[2],x_est[0],x_est[1],x_est[2]]) 

    listOfSlices[k].set_parameters(x_opt)
    square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix=[cost_matrix[i,:,:] for i in range(0,4)]
    update_cost_matrix(k,listOfSlices,square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix)
    cost_matrix = array([square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix])
    cost = cost_from_matrix(square_error_matrix,nbpoint_matrix,set_o,k)
    

    return x_opt,cost
    

def algo_optimisation(hyperparameters : dict,
                      listOfSlice : 'list[SliceObject]',
                      set_o : array,
                      set_r : array,
                      cost_matrix : array,
                      dicRes,
                      Vmx : int,
                      iteration_max : int,
                      ablation : str):
    
    """
    Optimise cost fuction for each slices in the list. Repeat optimisation until convergence
    """
                                     
    number_slices = len(listOfSlice)    
    
    
    #print('taille list :',len(blur_slices))
    square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix=compute_cost_matrix(listOfSlice)
    #print('taille list :',len(blur_slices))
    cost_matrix=array([square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix])
    
    global_iteration = 0
    convergence_globale=False

    while global_iteration<iteration_max and not convergence_globale: 
        
        convergence_globale=True
         
        set_to_register=abs(set_r.copy()) #all slice has to be register on the first iteration

        local_iteration = 0
        local_convergence=False

        while  not local_convergence and local_iteration < iteration_max :
            
            print('local_iteration :', local_iteration,'global_iteration :', global_iteration)
           
            eval_index=linspace(0,number_slices-1,number_slices,dtype=int)
            shuffle(eval_index)
            
                        
            for i_slice in eval_index: #nelder-mead is applied to optimise parameters for each slices
                if not(set_to_register[i_slice]):
                    #print('taille list :',len(blur_slices))
                    x_opt,bool_register=nelder_mead_optimisation(hyperparameters,listOfSlice,cost_matrix,set_to_register,set_o,Vmx,i_slice,ablation) 
                    listOfSlice[i_slice].set_parameters(x_opt)
                    set_to_register[i_slice]=bool_register
                    if not(bool_register):
                        convergence_globale=False #if one slice wasn't register in the local iteration, we start again
                    update_cost_matrix(i_slice,listOfSlice,square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix)
                    cost_matrix=array([square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix])
                
            
            costMse=compute_cost_from_matrix(square_error_matrix,nbpoint_matrix)
            print('costMse:', costMse)
            print('Dice:',somme(intersection_matrix)/Vmx)
             
            local_iteration+=1
            local_convergence=(set_to_register.all()==1) 
          
        
        global_iteration+=1 
        
   
    
    #compute the error on the original slices
    square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix=compute_cost_matrix(listOfSlice)
    costMse=compute_cost_from_matrix(square_error_matrix, nbpoint_matrix)
    costDice=compute_cost_from_matrix(intersection_matrix,union_matrix)
    if dicRes is not None:
        updateResults(dicRes,square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix,costMse,costDice,listOfSlice,number_slices)
        
    return square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix,dicRes



