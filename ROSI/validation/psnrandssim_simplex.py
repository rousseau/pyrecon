from os import listdir, isfile, join
from skimage.metrics import structural_similarity as ssim
import numpy as np
import re
import ants

val_simplex=['1','2','4','6']

for test_simplex in val_simplex : 

    dir_path = '../res/simplex_test/value%d/simul_data/' #lien vers les fichiers résultats
    fichiers = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]


#fonction pour calculer les métriques les plus courament utilisée entre les images reconstruites et les images originales

def EQM(reference_image,moving_image):
    
    X,Y,Z = reference_image.shape
    sum_EQM = np.sum((reference_image-moving_image)**2)
    EQM = sum_EQM/(X*Y*Z)
    
    return EQM
    
def PSNR(reference_image,moving_image):
    
    moving_image=moving_image
    eqm = EQM(reference_image,moving_image)
    print(eqm)
    d=np.max(reference_image)-np.min(reference_image)
    print(d)
    psnr = 10*np.log10(d**2/eqm)
    return psnr

def SSIM(reference_image,moving_image):
    
    # 5. Compute the Structural Similarity Index (SSIM) between the two images
    (score, diff) = ssim(reference_image, moving_image, full=True)
    diff = (diff * 255).astype("uint8")
    
    return score
    

moyen_psnr = []
moyen_ssim = []
petit_psnr = []
petit_ssim = []
tres_petit_psnr = []
tres_petit_ssim = []


#calcul du SSIM et du PSNR sur chaque image résultat
for file in fichiers: #j'ai appelé mes résultats avec le suffix 'ours', tu peux modifier le code en fonction de tes fichiers
        
        print('moving_image :',file)
        number_image = re.findall(r"\d+", file)
        i = number_image[0]
        print('reference_image :', 'image_%s.nii.gz' %(i), 'mask_%s.nii.gz' %(i))
        reference_image = ants.image_read('/home/mercier/Documents/donnee/DHCP/image_%s.nii.gz' %(i)) #je récupère l'image originale qui correspond à mon image reconstruite
        print('/home/mercier/Documents/donnee/DHCP/binmask_%s.nii.gz' %(i))
        reference_mask = ants.image_read('/home/mercier/Documents/donnee/DHCP/binmask_%s.nii.gz' %(i)) #le mask de l'image originale
        
        #register image to a reference image to compute PSNR, using ants
        moving_image = ants.image_read(dir_path + file)

        moving_mask = ants.image_read(dir_path + file.replace('_atlas_space','').replace('.nii.gz','') + '_mask.nii.gz')
        
        transform = ants.registration(reference_image,moving_image,'Rigid')
        moving_image_registered = ants.apply_transforms(reference_image,moving_image,transformlist=transform['fwdtransforms'])
        mask_registered = ants.apply_transforms(reference_mask,moving_mask,transformlist=transform['fwdtransforms'])
        
        #convert image from ants to numpy, in order to compute PSNR
        numpy_moving_image = moving_image_registered.numpy()
        numpy_moving_mask = mask_registered.numpy()
        numpy_reference_image = reference_image.numpy()
        numpy_reference_mask = reference_mask.numpy()
        
        numpy_reference_image = numpy_reference_image*numpy_reference_mask
        numpy_moving_image = numpy_moving_image*numpy_moving_mask
        
        #Compute PSNR and SSIM
        psnr = PSNR(numpy_reference_image,numpy_moving_image)
        print('Intersection', file, ', PSNR :',  psnr)
        
        #list_psnr_inter.append(psnr)
        ssim_res = SSIM(numpy_reference_image,numpy_moving_image)
        #list_ssim_inter.append(ssim_res)
        print('Intersection', file, ', SSIM :',  ssim_res)
        
        
        if 'moyen' in file: #je simule trois niveau de mouvement, ici, les résultats des SSIM et PSNR sont classifiés par niveau de mouvements
            
            moyen_psnr.append(psnr)
            moyen_ssim.append(ssim_res)
            
        elif 'trespetit' in file:
                
            tres_petit_psnr.append(psnr)
            tres_petit_ssim.append(ssim_res)
                
        elif 'petit' in file:
            
            petit_psnr.append(psnr)
            petit_ssim.append(ssim_res)

            
        ebner_path = dir_path + file.replace('ours.nii.gz','') + 'intersection.nii.gz' #j'ai des choses bizard dans mes nom de fichiers, les images reconstruite avec Ebner s'apelle '_intersection', ^^
        image_ebner = ants.image_read(ebner_path)
        transform = ants.registration(reference_image,image_ebner,'Rigid')
        ebner_registred = ants.apply_transforms(reference_image,image_ebner,transformlist=transform['fwdtransforms'])
        mask_path  = dir_path + file.replace('ours.nii.gz','') + 'intersection_mask.nii.gz'
        mask = ants.image_read(mask_path)
        transform = ants.registration(reference_image,mask,'Rigid')
        mask_registred = ants.apply_transforms(reference_image,mask,transformlist=transform['fwdtransforms'])
            
        numpy_mask = mask_registred.numpy() 
        numpy_ebner = ebner_registred.numpy()
        image_ebner = numpy_ebner * numpy_mask
            
        psnr_ebner = PSNR(numpy_reference_image,image_ebner)
        print('Ebner', file, ', PSNR :',  psnr_ebner)
            
        #list_psnr_ebner.append(psnr_ebner)
        ssim_ebner = SSIM(numpy_reference_image,image_ebner)
        #list_ssim_ebner.append(ssim_ebner)
        print('Ebner', file, ', SSIM :',  ssim_ebner)
            
        if 'moyen' in file:
                
            moyen_psnr_ebner.append(psnr_ebner)
            moyen_ssim_ebner.append(ssim_ebner)
                
        elif 'trespetit' in file:
                    
            tres_petit_psnr_ebner.append(psnr_ebner)
            tres_petit_ssim_ebner.append(ssim_ebner)
                    
        elif 'petit' in file:
                
            petit_psnr_ebner.append(psnr_ebner)
            petit_ssim_ebner.append(ssim_ebner)

psnr = np.zeros((3,4)) #First colume mean inter, second column std inter, third column mean ebner, fourth column std ebner
#first row  : tres_petit
#second row : petit
#third row : medium
ssim_vect = np.zeros((3,4))

psnr[0,0] = np.mean(tres_petit_psnr)
psnr[0,1] = np.std(tres_petit_psnr) 
psnr[0,2] = np.mean(tres_petit_psnr_ebner)
psnr[0,3] = np.std(tres_petit_psnr_ebner)

psnr[1,0] = np.mean(petit_psnr)
psnr[1,1] = np.std(petit_psnr) 
psnr[1,2] = np.mean(petit_psnr_ebner)
psnr[1,3] = np.std(petit_psnr_ebner)

psnr[2,0] = np.mean(moyen_psnr)
psnr[2,1] = np.std(moyen_psnr) 
psnr[2,2] = np.mean(moyen_psnr_ebner)
psnr[2,3] = np.std(moyen_psnr_ebner)

np.save('/home/mercier/Documents/res/figures/psnr',psnr) #les résultats sont enregistrer dans un tableau numpy

ssim_vect[0,0] = np.mean(tres_petit_ssim)
ssim_vect[0,1] = np.std(tres_petit_ssim) 
ssim_vect[0,2] = np.mean(tres_petit_ssim_ebner)
ssim_vect[0,3] = np.std(tres_petit_ssim_ebner)

ssim_vect[1,0] = np.mean(petit_ssim)
ssim_vect[1,1] = np.std(petit_ssim) 
ssim_vect[1,2] = np.mean(petit_ssim_ebner)
ssim_vect[1,3] = np.std(petit_ssim_ebner)

ssim_vect[2,0] = np.mean(moyen_ssim)
ssim_vect[2,1] = np.std(moyen_ssim) 
ssim_vect[2,2] = np.mean(moyen_ssim_ebner)
ssim_vect[2,3] = np.std(moyen_ssim_ebner)

np.save('/home/mercier/Documents/res/figures/ssim',ssim_vect)
