import joblib
from rosi.registration.load import convert2Slices
from rosi.registration.sliceObject import SliceObject
import numpy as np
from rosi.registration.intersection import cost_fct2, compute_cost_matrix, update_cost_matrix,cost_fct
from rosi.registration.tools import computeMaxVolume,apply_gaussian_filtering
from rosi.simulation.validation import same_order
from rosi.registration.transformation import ParametersFromRigidMatrix,rigidMatrix
import nibabel as nib
from rosi.registration.outliers_detection.feature import update_features
from rosi.simulation.validation import tre_for_each_slices
from rosi.registration.outliers_detection.multi_start import good_neighboors,removeBadSlice,estimate_new_position,correct_with_new_initialisation,grid_search, find_minimum,best_value, local_minimum, correct_slice_with_theorical_error
from rosi.registration.optimisation import algo_optimisation
from rosi.registration.outliers_detection.feature import detect_misregistered_slice
import pickle
import matplotlib.pyplot as plt
from rosi.registration.intersection import compute_cost_matrix, compute_cost_from_matrix
from rosi.registration.outliers_detection.multi_start import choose_postion

res = joblib.load('../../res/multi_5/Nelder-Mead/value/simul_data/Grand1/Nelder-Mead/res_test_Nelder-Mead_v2.joblib.gz')
                  #/mnt/Data/Chloe/res/omega/value0/simul_data/Grand1/omega0/res_test_omega0.joblib.gz')
                  #
                  #/mnt/Data/Chloe/res/omega/value0/simul_data/Grand1/omega0/res_test_omega0.joblib.gz')
                 #../../res/multi/Nelder-Mead/value/simul_data/Grand1/Nelder-Mead/res_test_Nelder-Mead_v2.joblib.gz')
                  #/mnt/Data/Chloe/res/omega/value0/simul_data/Grand1/omega0/res_test_omega0.joblib.gz')
                  #../../res/multi/Nelder-Mead/value/simul_data/Grand1/Nelder-Mead/res_test_Nelder-Mead_v2.joblib.gz')
                  #/mnt/Data/Chloe/res/omega/value0/simul_data/Grand1/omega0/res_test_omega0.joblib.gz')
                  #/mnt/Data/Chloe/res/multi/CG/value/simul_data/Grand1/CG/res_test_CG_v2.joblib.gz')
                  #)
                  #
                  #
                  #
                  #
                    #/mnt/Data/Chloe/res/omega/value0/simul_data/grand1/omega0/res_test_omega0.joblib.gz')
                  
                  #/mnt/Data/Chloe/res/omega/value0/simul_data/grand1/omega0/res_test_omega0.joblib.gz

key = [data[0] for data in res]
element = [data[1] for data in res]

listFeatures = element[-1]
listOfSlice = element[0]

new_data = res.copy()


listOfSlice = element[0]
nbslice=len(listOfSlice)
Vmx = computeMaxVolume(listOfSlice)

def check_multistart(num_slice1,num_slice2,listSlice,transfo,matrix):
 
    cost_matrix=matrix.copy()
    ge=cost_matrix[0,:,:].copy()
    gn=cost_matrix[1,:,:].copy()
    gi=cost_matrix[2,:,:].copy()
    gu=cost_matrix[3,:,:].copy()

    c2 = listSlice[num_slice2]
    print(type(c2))
    c1 = listSlice[num_slice1]
    print(type(c1))

    Mest_1 = c1.get_estimatedTransfo()
    Mest_2 = c2.get_estimatedTransfo()
    print(c1.get_indexSlice(),c2.get_indexSlice())
    M1 = transfo[c1.get_indexSlice()] @ c1.get_slice().affine
    M2 = transfo[c2.get_indexSlice()] @ c2.get_slice().affine
    T = np.dot(Mest_1,np.linalg.inv(M1))
    M2new = T @ M2
    M1new = T @ M1
    
    center = c1.get_centerOfRotation()
    center_mat = np.eye(4)
    center_mat[0:3,3] = center
    center_inv = np.eye(4)
    center_inv[0:3,3] = -center

    M1_est = center_mat @ M1new @ np.linalg.inv(c1.get_slice().affine) @ center_inv
    x1_est = ParametersFromRigidMatrix(M1_est)

    #print("estimation quality :", np.sum((c1.get_parameters()-x1_est)**2/6))

    center = c2.get_centerOfRotation()
    center_mat = np.eye(4)
    center_mat[0:3,3] = center
    center_inv = np.eye(4)
    center_inv[0:3,3] = -center

    update_cost_matrix(num_slice2,listOfSlice,ge,gn,gi,gu)
    cost_matrix = np.array([ge,gn,gi,gu])
    M_theorique = center_mat @ M2new @ np.linalg.inv(c2.get_slice().affine) @ center_inv
    x_theorique = ParametersFromRigidMatrix(M_theorique)
   
    
    return x_theorique,cost_matrix



transfo1 = '../../simu/Grand1/transfoAx_grand1.npy'
transfo2 = '../../simu/Grand1/transfoCor_grand1.npy'
transfo3 = '../../simu/Grand1/transfoSag_grand1.npy'
transfo_str=np.array([transfo1,transfo2,transfo3])

listTheorique=[]
output=convert2Slices(nib.load('../../simu/Grand1/LrAxNifti_grand1.nii.gz'),nib.load('../../simu/Grand1/brain_mask/LrAxNifti_grand1.nii.gz'),[],0,0)
listTheorique.extend(output)
output=convert2Slices(nib.load('../../simu/Grand1/LrCorNifti_grand1.nii.gz'),nib.load('../../simu/Grand1/brain_mask/LrCorNifti_grand1.nii.gz'),[],1,1)
listTheorique.extend(output)
output=convert2Slices(nib.load('../../simu/Grand1/LrSagNifti_grand1.nii.gz'),nib.load('../../simu/Grand1/brain_mask/LrSagNifti_grand1.nii.gz'),[],2,2)
listTheorique.extend(output)

listNomvt=[]
output=convert2Slices(nib.load('../../simu/Grand1/LrAxNifti_nomvt.nii.gz'),nib.load('../../simu/Grand1/brain_mask/LrAxNifti_grand1.nii.gz'),[],0,0)
listNomvt.extend(output)
output=convert2Slices(nib.load('../../simu/Grand1/LrCorNifti_nomvt.nii.gz'),nib.load('../../simu/Grand1/brain_mask/LrCorNifti_grand1.nii.gz'),[],1,1)
listNomvt.extend(output)
output=convert2Slices(nib.load('../../simu/Grand1/LrSagNifti_nomvt.nii.gz'),nib.load('../../simu/Grand1/brain_mask/LrSagNifti_grand1.nii.gz'),[],2,2)
listNomvt.extend(output)

image,theorique,features,transfolist=same_order(listOfSlice,listTheorique,listFeatures,transfo_str)
print([len(image[i]) for i in range(0,3)],[len(theorique[i]) for i in range(0,3)],[len(features[i]) for i in range(0,3)])

listOfSlice=np.concatenate(image)
squarre_error,nbpoint_matrix,intersection_matrix,union_matrix=compute_cost_matrix(listOfSlice)
matrix = np.array([squarre_error,nbpoint_matrix,intersection_matrix,union_matrix])
matrixcopy = np.array([squarre_error,nbpoint_matrix,intersection_matrix,union_matrix])
listFeatures=np.concatenate(features)

update_features(listOfSlice,listFeatures,squarre_error,nbpoint_matrix,intersection_matrix,union_matrix)

image,nomvt,features,transfolist=same_order(listOfSlice,listNomvt,listFeatures,transfo_str)
listOfSlice=np.concatenate(image)
listNomvt=np.concatenate(nomvt)
listFeatures=np.concatenate(features)

number_slice = len(listOfSlice)
set_r = np.zeros(number_slice)

#image,theorique,features,transfolist=same_order(listOfSlice,listNomvt,listFeatures,transfo_str)
tre_for_each_slices(listOfSlice,listOfSlice,listFeatures,transfolist,[])
list_theorique = np.zeros((number_slice,6))
er_2 = [e.get_error() for e in listFeatures]
for fk in range(0,len(image)):
    
    tre=[e.get_error() for e in features[fk]]
    #print(len(image[fk]),len(features[fk]))
    bad_registered_slices=np.where(np.array(tre)>3)[0]
    #print(bad_registered_slices)
    good_slice=np.where(np.array([f.get_error() for f in features[fk]])<3)[0][3]

    #print(good_slice)
    #print('fk',fk)
    transfo=np.load(transfolist[fk])

    for bad_slice in bad_registered_slices:
        #print(fk,bad_slice)
        
        if fk>0:
            stack_size = sum([len(image[i]) for i in range(0,fk)])
            k=bad_slice + (stack_size)
            kprime = good_slice + stack_size
        else :
            k=bad_slice
            kprime=good_slice

        set_o = np.zeros(nbslice)
        lamb=0
        
        x0=listOfSlice[k].get_parameters()
        #print(k)
        #print(k,listOfSlice[k].get_indexSlice(),listOfSlice[k].get_stackIndex(),image[fk][bad_slice].get_indexSlice(),image[fk][bad_slice].get_stackIndex())
        #
        #print("slice :",k, kprime)
        x_theorique,cost_matrix=check_multistart(kprime,k,listOfSlice,transfo,matrixcopy)
        #listOfSlice[k].set_parameters(x_theorique)
        #print('slice k',k)

    
        
        set_r[k]=1
        c=cost_fct(x0,k,listOfSlice,matrix,set_o,lamb,Vmx)
        #c_theorique=cost_fct(x_theorique,k,listOfSlice,matrix,set_o,lamb,Vmx)
        if k!=12:
            listOfSlice[k].set_parameters(x0)    
        else :
            listOfSlice[k].set_parameters(x0)
        
        #print(k,c,c_theorique,c<c_theorique)
        #else : 
        #listOfSlice[k].set_parameters(x_theorique) 
        #print('x_theorique :',x_theorique) 
        list_theorique[k,:]=x_theorique
        #listOfSlice[k].set_parameters(x_theorique)
        #update_cost_matrix(k,listOfSlice,squarre_error,nbpoint_matrix,intersection_matrix,union_matrix)
        
        
        
        #if c<c_theorique:
        #    data = listOfSlice[k].get_slice().get_fdata()
        #    mask = listOfSlice[k].get_mask()
        #    plt.figure()
        #    plt.imshow(data*mask)
        #    plt.savefig('image_%s.png' %(k))
        #listOfSlice[k].set_parameters(x_theorique)
        #print(x0,x_theorique)
        #print(k,c_theorique<c,listFeatures[k].get_mse(),listFeatures[k].get_dice(),listFeatures[k].get_error())  

squarre_error,nbpoint_matrix,intersection_matrix,union_matrix=compute_cost_matrix(listOfSlice)
matrix = np.array([squarre_error,nbpoint_matrix,intersection_matrix,union_matrix])
hyperparameters = np.array([4,0.01,2000,0.25,1,0]) 
i=0
rejected_slice = element[-2]
set_r = np.zeros(len(listFeatures))
for i_slice in range(0,len(listFeatures)):
    slice = listFeatures[i_slice]
    if (slice.get_stack(),slice.get_index()) in rejected_slice:
        set_r[i_slice]=1


#load_model = pickle.load(open('my_model.pickle','rb'))
#set_o = detect_misregistered_slice(listOfSlice, matrix, load_model)
#W = np.zeros(number_slice)
#for i_slice in range(0,len(listOfSlice)):
#    W[i_slice] = (listFeatures[i_slice].get_mask_proportion())
    #print(w,listOfSlice[i_slice].get_indexSlice())
    #matrix[0,i_slice, :] = w * matrix[0,i_slice,:]
    #matrix[0,:, i_slice] = w * matrix[0,:,i_slice]


correct_slice_with_theorical_error(set_r,listOfSlice,hyperparameters,'Nelder-Mead',Vmx,matrix,listFeatures,transfolist,listNomvt)

iter=0
while iter<1 :
    #algo_optimisation(hyperparameters,listOfSlice,set_r,set_r,matrix,None,Vmx,1) 
    tre_for_each_slices(listOfSlice,listOfSlice,listFeatures,transfolist,[])
    tre = [e.get_error() for e in listFeatures]
    tre = np.array(tre) 
    mask = [e.get_mask_proportion()[0] for e in listFeatures]
    mask = np.array(mask)
    index = np.array([np.where(set_r==1)])[0][0]
    ordered_tre = np.argsort(-mask[index],axis=-1)
    
    
    for iv in ordered_tre:  
            i_slice = index[iv] 
            print(i_slice)
            if set_r[i_slice]==1:
            
                
                if mask[i_slice]<0.1:
                    lamb=2
                    hyperparameters[4]=2
                else : 
                    lamb=0
                    listblur = apply_gaussian_filtering(listOfSlice,sigma=3)
                    hyperparameters[4]=0
                
                x0 = listOfSlice[i_slice].get_parameters()
                i_before,i_after=good_neighboors(listOfSlice,i_slice,set_r,3)
                print(i_before,i_after)
                title = 'slice_%s_grand1.png' %(i_slice)
                plt.figure()
                plt.imshow(listOfSlice[i_slice].get_slice().get_fdata()*listOfSlice[i_slice].get_mask(),cmap='gray')
                plt.savefig(title)
                iter=1
                new_x = np.zeros((15,6))
                id=0
                for id_b in range(0,3):
                    for id_a in range(0,3):
                        print(i_before[id_b],i_after[id_a])
                        new_x[id,:] = estimate_new_position(listOfSlice,i_before[id_b],i_slice,i_after[id_a])
                        id+=1
                for id_b2 in range(0,3):
                    new_x[id,:] = estimate_new_position(listOfSlice,i_before[id_b2],i_slice,None)
                    id+=1
                for id_a2 in range(0,3):
                    new_x[id,:] = estimate_new_position(listOfSlice,None,i_slice,i_after[id_a2])
                    id+=1
                for id in range(0,len(new_x)):
                    print(id)
                    if id != 0:
                        pA = new_x[id,:]
                        A = rigidMatrix(pA)
                        RA = A[0:3,0:3]
                        
                        for prev in range(0,id):
                            print(prev)
                            pR = new_x[prev,:]
                            R = rigidMatrix(pR)
                            RR = R[0:3,0:3]
                            tr = np.trace(RA@RR.T)
                            theta = np.arccos((tr-1)/2)
                            print(id,prev,theta)
                            if theta<0.2:
                                new_x[id,:]=new_x[prev,:]
                                break
                                
                new_x=np.unique(new_x,axis=0)

                x_opt = np.zeros((len(new_x),6))
                cost = np.zeros(len(new_x))
                for id in range(0,len(new_x)):
                    #x_opt = correct_with_new_initialisation(listOfSlice,hyperparameters,i_slice,new_x[id],matrix,np.zeros(number_slice),Vmx,'Nelder-Mead')
                    #listOfSlice[i_slice].set_parameters(new_x)
                    #x0 = listOfSlice[i_slice].get_parameters()
                    grid,estimated_trans = grid_search(new_x[id,:],hyperparameters,listOfSlice,matrix,set_r,set_r,Vmx,i_slice,'Nelder-Mead')
                    minimum = local_minimum(grid)
                    starts = find_minimum(new_x[id,:],minimum,estimated_trans)
                    x_opt[id,:], cost[id] = best_value(hyperparameters,listOfSlice,matrix,set_r,Vmx,i_slice,starts,'Nelder-Mead')
                    #print(x_opt[id,:],cost_fct(x_opt[id,:],i_slice,listOfSlice,matrix,set_r,0,Vmx))
                
                #print(new_x)
                #theo = list_theorique[i_slice,:]
                
                #x=new_x
                #x = theo#np.zeros(6)
                #x[0:3] = theo[0:3]
                #x[3:6] = new_x[3:6]
                #print(x)
                #print('thoeorique :',theo)
                #print('new_x',new_x)
                        #print('new_x',new_x)
                # 
                #grid,estimated_trans = grid_search(new_x,hyperparameters,listOfSlice,matrix,set_r,set_r,Vmx,i_slice,'Nelder-Mead')
                #minimum = local_minimum(grid)
                #starts = find_minimum(new_x,minimum,estimated_trans)
                #x_opt = best_value(hyperparameters,listOfSlice,matrix,set_r,Vmx,i_slice,starts,'Nelder-Mead')
                #listOfSlice[i_slice].set_parameters(x_opt)
                #
                #estimated_x=new_x;
               
                
                #listOfSlice[i_slice].set_parameters(new_x)
                
                #c2=cost_fct(x0,i_slice,listOfSlice,matrix,np.zeros(number_slice),lamb,Vmx)
                #c2=cost_fct2(x0,i_slice,listOfSlice,matrix,set_r,lamb,Vmx,W)
                #c2_theorique=cost_fct(theo,i_slice,listOfSlice,matrix,np.zeros(number_slice),lamb,Vmx)
                #c1_theorique=cost_fct(theo,i_slice,listOfSlice,matrix,np.zeros(number_slice),lamb,Vmx)
                #c2_estimated=cost_fct(x_opt,i_slice,listOfSlice,matrix,np.zeros(number_slice),lamb,Vmx)
               # c1_estimated=cost_fct(x_opt,i_slice,listOfSlice,matrix,set_o,lamb,Vmx)

                #print('initialisation :', new_x)
                #print('estimated :',x_opt)
                #print('thoerical x :', theo)
                #print('cost new :', c1 , 'cost theo :', c1_theorique,'cost estimated :', c1_estimated)
                #print('cost new :', c2 , 'cost theo :', c2_theorique, 'cost estimated :', c2_estimated)
        
    tre_for_each_slices(listNomvt,listOfSlice,listFeatures,transfolist,[])
    tre_new = np.array([e.get_error() for e in listFeatures])
    new_set = tre_new>3
    new_set = np.logical_and(set_r,new_set)
    #i+=1
    if np.all(set_r==new_set):
        break
    else:
        set_r=new_set.copy()
    iter=1

rs = []
while True:
    tre_for_each_slices(listOfSlice,listOfSlice,listFeatures,transfolist,rs)
    tre = [e.get_error() for e in listFeatures]
    tre = np.array(tre)
    print(np.sum(np.array(tre)>1.5))
    print(tre)
    if sum(np.array(tre)>1.5)==0:
        break
    maxs = np.argmax(tre)
    el = (listFeatures[maxs].get_stack(),listFeatures[maxs].get_index())
    rs.append(el)
    print(rs)

print(np.where(tre>3))


dicRes={}
dicRes["evolutionparameters"] =np.reshape(element[key.index('EvolutionParameters')][-1,:,:],-1).tolist() 
dicRes["evolutiontransfo"] = np.reshape(element[key.index('EvolutionTransfo')][-1,:,:],(4*number_slice,4)).tolist() 
dicRes["evolutiongriderror"] = element[key.index('EvolutionGridError')][-1,:,:].tolist()  
dicRes["evolutiongridnbpoint"] = element[key.index('EvolutionGridNbpoint')][-1,:,:].tolist() 
dicRes["evolutiongridinter"] = element[key.index('EvolutionGridInter')][-1,:,:].tolist()  
dicRes["evolutiongridunion"] = element[key.index('EvolutionGridUnion')][-1,:,:].tolist() 
dicRes["evolutionerror"] = element[key.index('EvolutionGridError')][-1,:,:].tolist()
dicRes["evolutionnbpoint"] = element[key.index('EvolutionGridNbpoint')][-1,:,:].tolist()
dicRes["evolutionGridInter"] = element[key.index('EvolutionGridInter')][-1,:,:].tolist()
dicRes["evolutionGridUnion"] = element[key.index('EvolutionGridUnion')][-1,:,:].tolist()
costMse=compute_cost_from_matrix(squarre_error, nbpoint_matrix)
print("Cost Before we start multi-start",costMse)
costDice=compute_cost_from_matrix(intersection_matrix,union_matrix)
dicRes["evolutionerror"] = [] 
dicRes["evolutiondice"] = []
dicRes["evolutionerror"].append(costMse)
dicRes["evolutiondice"].append(costDice)
    

ErrorEvolution=dicRes["evolutionerror"]
DiceEvolution=dicRes["evolutiondice"]
nbit = len(ErrorEvolution)

    
    #strEGE = file + '/EvolutionGridError.npz'
EvolutionGridError = np.reshape(dicRes["evolutiongriderror"],[nbit,number_slice,number_slice])
    #np.savez_compressed(strEGE,EvolutionGridError)
    
    #strEGN = file + '/EvolutionGridNbpoint.npz'
EvolutionGridNbpoint = np.reshape(dicRes["evolutiongridnbpoint"],[nbit,number_slice,number_slice])
    #np.savez_compressed(strEGN,EvolutionGridNbpoint)
    
    #strEGI = file + '/EvolutionGridInter.npz'
EvolutionGridInter = np.reshape(dicRes["evolutiongridinter"],[nbit,number_slice,number_slice])
    #np.savez_compressed(strEGI,EvolutionGridInter)
    
    #strEGU = file + '/EvolutionGridUnion.npz'
EvolutionGridUnion = np.reshape(dicRes["evolutiongridunion"],[nbit,number_slice,number_slice])
    #np.savez_compressed(strEGU,EvolutionGridUnion)
    
    #strEP = file + '/EvolutionParameters.npz'
EvolutionParameters = np.reshape(dicRes["evolutionparameters"],[nbit,number_slice,6])
    #np.savez_compressed(strEP,EvolutionParameters)
    
    #strET = file + '/EvolutionTransfo.npz'
EvolutionTransfo = np.reshape(dicRes["evolutiontransfo"],[nbit,number_slice,4,4])
    #np.savez_compressed(strET,EvolutionTransfo)
    #strLM = file + '/CostGlobal.npz'
    #costGlobal.tofile(strLM)
    
    #transfo = args.simulation
tre_for_each_slices(listNomvt,listOfSlice,listFeatures,transfolist,[])
    #tre_new = np.array([e.get_error() for e in listFeatures])

bad_slices = removeBadSlice(listOfSlice,set_r)
res_obj = [('listSlice',listOfSlice),('ErrorEvolution',ErrorEvolution), ('DiceEvolution',DiceEvolution), ('EvolutionGridError',EvolutionGridError), ('EvolutionGridNbpoint',EvolutionGridNbpoint), ('EvolutionGridInter',EvolutionGridInter), ('EvolutionGridUnion',EvolutionGridUnion), ('EvolutionParameters',EvolutionParameters),('EvolutionTransfo',EvolutionTransfo),('RejectedSlices',bad_slices),('ListError',listFeatures)]
    
joblib_name = '../../res/multi/test_grand1_v4.joblib.gz' 
joblib.dump(res_obj,open(joblib_name,'wb'), compress=True)
    
res = joblib.load(open(joblib_name,'rb'))
key=[p[0] for p in res]
element=[p[1] for p in res]
#listSlice=element[key.index('listSlice')]

#print(tre)
for slicei in listOfSlice:
    new_slice = nib.Nifti1Image(slicei.get_slice().get_fdata(),slicei.get_estimatedTransfo())
    print(type(slicei.get_indexSlice()),type(slicei.get_stackIndex()))
    nib.save(new_slice,'../../res/multi/all_slices_corrected/slice_%s_in_stack_%s'%(slicei.get_indexSlice(),slicei.get_stackIndex())) 
