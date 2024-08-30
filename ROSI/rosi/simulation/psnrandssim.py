from os import listdir
from skimage.metrics import structural_similarity as ssim
import numpy as np
import re
import nibabel as nib
from scipy.ndimage import map_coordinates
#import ants


#fonction pour calculer les métriques les plus courament utilisée entre les images reconstruites et les images originales

def EQM(reference_image,moving_image):
    size = reference_image.size
    sum_EQM = np.sum((reference_image-moving_image)**2)
    EQM = sum_EQM/(size)
    return EQM
    
def PSNR(reference_image,moving_image):
    eqm = EQM(reference_image,moving_image)
    d=max(np.max(reference_image),np.max(moving_image))-min(np.min(reference_image),np.min(moving_image))
    print(d,max(np.max(reference_image),np.max(moving_image)),min(np.min(reference_image),np.min(moving_image)))
    psnr = 10*np.log10(d**2/eqm)
    return psnr

def SSIM(reference_image,moving_image):
    # 5. Compute the Structural Similarity Index (SSIM) between the two images
    (score, diff) = ssim(reference_image, moving_image, full=True)
    diff = (diff * 255).astype("uint8")
    return score

def norm_data(data):
    mean_data=np.mean(data)
    std_data=np.std(data,ddof=1)
    print(mean_data,std_data)
    return (data-mean_data)/std_data  

def NCC(reference_image,moving_image):
    #index = moving_image>0
    data0 = norm_data(moving_image)
    data0 = data0
    data1= norm_data(reference_image)
    data1= data1
    #print(len(data1),len(data0))
    if np.all(data0 == 0) or np.all(data1 == 0) or np.isnan(np.sum(data0*data1)):
        return 0
    return ((1.0/(data0.size-1))* np.sum(data0*data1))

from sklearn.mixture import GaussianMixture

path_test = "/home/mercier/Documents/res/Donnees_articles/all_ebner/value/sub-0018/ses-0021/Ebner/recon_template_space/srr_template.nii.gz"
mask_test = "/home/mercier/Documents/res/Donnees_articles/all_ebner/value/sub-0018/ses-0021/Ebner/recon_template_space/srr_template_mask.nii.gz"
#path_test= "/home/mercier/Documents/res/marseille_results/sub-0002/ses-0003/volume.nii"
image_test = nib.load(path_test)
mask = nib.load(mask_test)

image = image_test.get_fdata()*mask.get_fdata()

def snr(reconstructed_hr_image):
    data = reconstructed_hr_image.reshape(-1)
    data = data[np.where(data>0)]
    gm = GaussianMixture(n_components=3,random_state=0).fit(data.reshape((-1,1)))
    mu1,mu2,mu3 = gm.means_
    mu1=mu1[0]; mu2=mu2[0]; mu3=mu3[0]
    var1,var2,var3 = gm.covariances_
    var1=var1[0][0]; var2=var2[0][0]; var3=var3[0][0]
    w1,w2,w3 = gm.weights_
    mu = w1*mu1 + w2*mu2 + w3*mu3
    var = w1*var1 + w2*var2 + w3*var3 + w1*mu1**2 + w2*mu2**2 + w3*mu3**2 - (w1*mu1 + w2*mu2 + w3*mu3)**2
    #std = np.sqrt(var1) + np.sqrt(var2) + np.sqrt(var3)
    return 20*np.log10(mu/np.sqrt(var))

def pve(reconstructed_hr_image):
    data = reconstructed_hr_image.reshape(-1)
    data = data[np.where(data>0)]
    nb_voxel=data.size
    gm = GaussianMixture(n_components=3,random_state=0).fit(data.reshape((-1,1)))
    mu1,mu2,mu3 = gm.means_
    mu1=mu1[0]; mu2=mu2[0]; mu3=mu3[0]
    var1,var2,var3 = gm.covariances_
    var1=var1[0][0]; var2=var2[0][0]; var3=var3[0][0]
    fhwm1=2.355*np.sqrt(var1) ; fhwm2=2.355*np.sqrt(var2) ; fhwm3=2.355*np.sqrt(var3)
    t1 = [max(0,mu1-fhwm1),mu1+fhwm1]; t2 = [max(0,mu2-fhwm2),mu2+fhwm2]; t3 = [max(0,mu3-fhwm3),mu3+fhwm3]
    pve=0
    for v in data:
        if not ((v>t1[0] and v<t1[1]) or (v>t2[0] and v<t2[1]) or (v>t3[0] and v<t3[1])):
            pve+=1
    return pve/nb_voxel*100


br_directory = "/home/mercier/Documents/res/nesvor_rosi/sub-0004/ses-0006/"
hr_image = nib.load("/home/mercier/Documents/res/nesvor_rosi/sub-0004/ses-0006/volume.nii")
metric = "PSNR"

def QualityMetric(br_directory, hr_image ,metric):
    """
    compute residus between one low resolution image (br) and one high resolution image (hr)
    """
    #result image
    if not os.path.exists(br_directory):
        print("This directory does not exists")
        return 0
    list_slice = [file for file in os.listdir(br_directory) if not "mask" in file and not "volume" in file]
    m_value=[]
    for slicei in range(0,len(list_slice)) : 
        path_to_slicei = os.path.join(br_directory,list_slice[slicei])
        dataslice = nib.load(path_to_slicei)
        path_to_mask = os.path.join(br_directory,'mask_'+list_slice[slicei])
        mask = nib.load(path_to_mask)
        brdata = dataslice.get_fdata()*mask.get_fdata()
        if not np.all(brdata==0): 
            X,Y,_ = dataslice.shape
            hr_affine=hr_image.affine
            hr_data=hr_image.get_fdata()
            brdata = brdata.squeeze()
            coordinate_in_lr = np.zeros((4,X*Y)) #initialisation of coordinate in the low resolution image, with 6 points per voxels
                #create the coordinate of points of the slice i in the LR image, with 6 points per voxel, center in 0
            ii = np.arange(0,X) 
            jj = np.arange(0,Y)
            iv,jv = np.meshgrid(ii,jj,indexing='ij')
            iv = np.reshape(iv, (-1))
            jv = np.reshape(jv, (-1))
            coordinate_in_lr[0,:] = iv
            coordinate_in_lr[1,:] = jv
            coordinate_in_lr[2,:] = 0
            coordinate_in_lr[3,:] = 1
            br_affine = dataslice.affine
            coordinate_in_world = br_affine @ coordinate_in_lr
            coordinate_in_hr = (np.linalg.inv(hr_affine) @ coordinate_in_world) #np.linalg.inv(image.affine) @ coordinate_in_world
            interpolate = np.zeros(X*Y)
            map_coordinates(hr_data,coordinate_in_hr[0:3,:],output=interpolate,order=3,mode='constant',cval=0,prefilter=False)
            value_in_hr = np.reshape(interpolate,(X,Y))
            hrvalue = value_in_hr / np.quantile(value_in_hr,0.99)
            hrvalue[np.isnan(hrvalue)] = 0
            brdata[np.isnan(brdata)] = 0
            print(hrvalue[np.isnan(hrvalue)])
            print(brdata[np.isnan(brdata)])
            #print('values :',mask_in_hr[np.where(mask_in_hr>0)])
            #print('data :',data[np.where(data>0)])
            if metric=="NCC" :
                m_value.append(NCC(hrvalue,brdata))
            if metric=="PSNR" :
                m_value.append(PSNR(hrvalue,brdata))
            if metric=="SSIM" :
                m_value.append(SSIM(brdata,hrvalue))
    return m_value



