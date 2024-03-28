from os import listdir
from skimage.metrics import structural_similarity as ssim
import numpy as np
import re
#import ants


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
    


