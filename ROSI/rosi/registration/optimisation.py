from random import shuffle
from .tools import somme, apply_gaussian_filtering, separate_slices_in_stacks
from scipy.optimize import minimize,approx_fprime, newton,least_squares, check_grad
from numpy import linspace, ones, eye, concatenate, array, zeros, dtype
from .intersection import compute_cost_matrix, update_cost_matrix, compute_cost_from_matrix, cost_fct, cost_fct2,updateResults, cost_from_matrix, cost_multi_start
from .sliceObject import SliceObject
from numpy.linalg import norm



import pickle

#load_model = pickle.load(open('ROSI/rosi/registration/outliers_detection/my_new_model.pickle','rb'))

def nelder_mead_optimisation(hyperparameters : array(6),
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
    delta,x_tol,f_tol,epsilon,omega=hyperparameters[0],hyperparameters[1],hyperparameters[2],hyperparameters[3],hyperparameters[4]

    #previous_parameters
    #print('shape list :', len(listOfSlices))
    x0 = listOfSlices[k].get_parameters()
    
    #initial simplex
    p=(ones((x0.size,x0.size))*x0.T)+eye(6)*delta
    initial_s=concatenate(([x0],p),axis=0)

    #optimisation with Nelder-Mead
    ftol=1e-4
    #print('delta',delta,'xatol',x_tol)
    if ablation == 'Nelder-Mead':
        NM = minimize(cost_fct,x0,args=(k,listOfSlices,cost_matrix,set_o,omega,Vmx),method='Nelder-Mead',options={"disp" : False, "maxiter" : f_tol, "maxfev":1e3, "fatol" : ftol,"xatol" : x_tol, "initial_simplex" : initial_s , "adaptive" :  False, "return_all" :True})
        x_opt = NM.x
    elif ablation != "LM":
        NM = minimize(cost_fct,x0,args=(k,listOfSlices,cost_matrix,set_o,omega,Vmx),method=ablation,options={"disp" : True})
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


def newton_optimisation(hyperparameters : array(6),
                            listOfSlices : 'list[SliceObject]',
                            cost_matrix : array,
                            set_r : array,
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
    delta,x_tol,f_tol,epsilon,omega=hyperparameters[0],hyperparameters[1],hyperparameters[2],hyperparameters[3],hyperparameters[4]

    #previous_parameters
    #print('shape list :', len(listOfSlices))
    #x0 = listOfSlices[k].get_parameters()
    
    #initial simplex
    #p=(ones((x0.size,x0.size))*x0.T)+eye(6)*delta
    #initial_s=concatenate(([x0],p),axis=0)

    #optimisation with Nelder-Mead
    ftol=1e-4
    #print('delta',delta,'xatol',x_tol)
    #print(x0)
    #fprime=approx_fprime(x0,cost_fct,ones(6),k,listOfSlices,cost_matrix,set_o,omega,Vmx)
    print(optimisation)
    #print(dtype(cost_matrix[0][0,0]))
    if optimisation == 'Nelder-Mead':
        p=(ones((x0.size,x0.size))*x0.T)+eye(6)*delta
        initial_s=concatenate(([x0],p),axis=0)
        NM = minimize(cost_fct,x0,args=(k,listOfSlices,cost_matrix,set_o,omega,Vmx),method='Nelder-Mead',options={"disp" : False, "maxiter" : f_tol, "maxfev":1e3, "fatol" : ftol,"xatol" : x_tol, "initial_simplex" : initial_s , "adaptive" :  False, "return_all" :True })
        x_opt = NM.x
    elif optimisation != "LM":
        #jacob  = approx_fprime(x0, cost_fct,[1e-8,1e-8,1e-8,1e-8,1e-8,1e-8],k,listOfSlices,cost_matrix,set_o,omega,Vmx)
    
        NM = minimize(cost_fct,x0,args=(k,listOfSlices,cost_matrix,set_o,omega,Vmx),method=optimisation,jac=None,options={"disp" : False, "return_all" : False, "gtol" : 6*(1e-3), "eps" : 1e-4})
            #"gtol" : 6*(1e-3), "eps" : 1e-4
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
    #NM = newton(cost_fct,x0,args=(k,listOfSlices,cost_matrix,set_o,omega,Vmx))
    #print(NM)
    #update the parameters of the slice and the cost"
    #x_opt = NM.x 
    #print(x_opt)
    #print(NM.fun)
    listOfSlices[k].set_parameters(x_opt)
    square_error_matrix,nboint_matrix,intersection_matrix,union_matrix=[cost_matrix[i,:,:] for i in range(0,4)]
    update_cost_matrix(k,listOfSlices,square_error_matrix,nboint_matrix,intersection_matrix,union_matrix)
    cost_matrix = array([square_error_matrix,nboint_matrix,intersection_matrix,union_matrix])
       
    return x_opt




def translation_optimisation(hyperparameters : array(6),
                            listOfSlices : 'list[SliceObject]',
                            cost_matrix : array,
                            set_r : array,
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
    delta,x_tol,f_tol,epsilon,omega=hyperparameters[0],hyperparameters[1],hyperparameters[2],hyperparameters[3],hyperparameters[4]

    xr = x0[0:3]
    xt = x0[3:6]
    print('xt init :', xt)

    #initial simplex
    p=(ones((xt.size,xt.size))*xt.T)+eye(3)*delta
    initial_s=concatenate(([xt],p),axis=0)

    
    #optimisation with Nelder-Mead
    ftol=1e-4
    #print('delta',delta,'xatol',x_tol)
    if optimisation=='Nelder-Mead' :
        NM = minimize(cost_multi_start,xt,args=(xr,k,listOfSlices,cost_matrix,set_o,omega,Vmx),method='Nelder-Mead',options={"disp" : True, "maxiter" : f_tol, "maxfev":1e3, "fatol" : ftol,"xatol" : x_tol, "initial_simplex" : initial_s , "adaptive" :  False, "return_all" :False})
        x_est = NM.x
    elif optimisation != 'LM':
        #print('Gradient')
        #def fprime(x : array(3),*args):
        #    return approx_fprime(x, cost_multi_start, 1e-30,*args)
        NM = minimize(cost_multi_start,xt,args=(xr,k,listOfSlices,cost_matrix,set_o,omega,Vmx),method=optimisation,jac=None,options={"disp" : True,"gtol" : 1e-10, "eps" : 1e-10})
        x_est = NM.x
        print(NM.message)

    else :
        print('LM')
        NM = least_squares(cost_multi_start, xt,args=(xr,k,listOfSlices,cost_matrix,set_o,omega,Vmx),xtol=1e-10,ftol=1e-10)
        x_est = NM.x
        print(NM.message)
    #update the parameters of the slice and the cost
    
    print('x_est',x_est)
    x_opt = array([xr[0],xr[1],xr[2],x_est[0],x_est[1],x_est[2]]) 
    #print(NM.fun)
    listOfSlices[k].set_parameters(x_opt)
    square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix=[cost_matrix[i,:,:] for i in range(0,4)]
    update_cost_matrix(k,listOfSlices,square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix)
    cost_matrix = array([square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix])
    cost = cost_from_matrix(square_error_matrix,nbpoint_matrix,set_r,k)
    

    return x_opt,cost
    

def algo_optimisation(hyperparameters : array(6),
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
    
    #hyperparameters : 
    blur = hyperparameters[5]

    #gaussian filtering on slices
    if blur != 0 :
        blur_slices = apply_gaussian_filtering(listOfSlice.copy(),blur)
    else : 
        blur_slices = listOfSlice.copy()
    
    #print('taille list :',len(blur_slices))
    square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix=compute_cost_matrix(blur_slices)
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
                    x_opt,bool_register=nelder_mead_optimisation(hyperparameters,blur_slices,cost_matrix,set_to_register,set_o,Vmx,i_slice,ablation) 
                    blur_slices[i_slice].set_parameters(x_opt)
                    set_to_register[i_slice]=bool_register
                    if not(bool_register):
                        convergence_globale=False #if one slice wasn't register in the local iteration, we start again
                    update_cost_matrix(i_slice,blur_slices,square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix)
                    cost_matrix=array([square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix])
                
            
            costMse=compute_cost_from_matrix(square_error_matrix,nbpoint_matrix)
            print('costMse:', costMse)
            print('Dice:',somme(intersection_matrix)/Vmx)
             
            local_iteration+=1
            local_convergence=(set_to_register.all()==1) 
          
        
        global_iteration+=1 
        
    for i_slice in range(number_slices):
        listOfSlice[i_slice].set_parameters(blur_slices[i_slice].get_parameters())
    
    #compute the error on the original slices
    square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix=compute_cost_matrix(listOfSlice)
    costMse=compute_cost_from_matrix(square_error_matrix, nbpoint_matrix)
    costDice=compute_cost_from_matrix(intersection_matrix,union_matrix)
    if dicRes is not None:
        updateResults(dicRes,square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix,costMse,costDice,listOfSlice,number_slices)
        
    return square_error_matrix,nbpoint_matrix,intersection_matrix,union_matrix,dicRes



