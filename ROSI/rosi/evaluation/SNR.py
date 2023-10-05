import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as gmm
import scipy.stats

#Aide pour l'estimation des gaussiennes
#https://itecnote.com/tecnote/python-fit-mixture-of-two-gaussian-normal-distributions-to-a-histogram-from-one-set-of-data-python/


image = nib.load('../DHCP/image_3.nii.gz')
mask = nib.load('../DHCP/binmask_3.nii.gz')

data = image.get_fdata()*mask.get_fdata()
#data = data.reshape(-1)
data = data[data>0]

histo = plt.hist(data,bins=1000,density=True,stacked=True)
plt.show()
plt.savefig('histogramme.png')

X = data.reshape(-1,1)


model = gmm(n_components=3).fit(X)

#print(model.means_)
#print(mu)
#print(model.covariances_)
#print(sigma)

mu1,mu2,mu3 = model.means_
sig1,sig2,sig3 = np.sqrt(model.covariances_)
w1,w2,w3 = model.weights_
#print(weight)
#print(sum(weight))

#m = [weight[i]*vmu[i] for i in range(0,3)]
mu = w1*mu1+w2*mu2+w3*mu3#np.sum(m)
#print(vmu)
print(mu)
var1,var2,var3 = model.covariances_
var = w1*var1+w2*var2+w3*var3+w1*mu1**2+w2*mu2**2+w3*mu3**2-(w1*mu1+w2*mu2+w3*mu3)**2
sigma = np.sqrt(var)
print(sigma)

x=np.linspace(0,300,300)
y0 = w1*scipy.stats.norm.pdf(x,mu1,sig1)
y1 = w2*scipy.stats.norm.pdf(x,mu2,sig2)
y2 = w3*scipy.stats.norm.pdf(x,mu3,sig3)

#print(y0[0])
plt.plot(x,y0[0],color='red')
plt.plot(x,y1[0],color='green')
plt.plot(x,y2[0],color='blue')
plt.savefig('GMM.png')

print(mu)
snr = 20*np.log(mu/sigma)

print('snr :',snr)

