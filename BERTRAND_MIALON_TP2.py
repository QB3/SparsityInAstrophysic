import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from astropy.io import fits
import scipy.misc
import BERTRAND_MIALON_TP1 as tp1

###########################################################
#on charge les données
hdul = fits.open('simu_sky.fits')
hdul.info()
xStar = hdul[0].data
plt.figure()
plt.imshow(xStar, cmap='gray')
path = "imagesTP2/simu_sky.jpg"
scipy.misc.imsave(path, xStar)

#########################################################################
############## Estimation of a multiresolution mask ####################
#########################################################################

###########################################################"
#on ajoute un bruit gaussien
n = xStar.shape[0]
y = xStar + np.random.normal(0, 50, (n, n))
plt.figure()
plt.imshow(y, cmap='gray')
path = "imagesTP2/noisy_simu_sky.jpg"
scipy.misc.imsave(path, y)

###########################################################
########################### calcul de sigma MAD

#d'abord on calcule la tranformée en ondelettes
h = np.array([1, 4, 6, 4, 1])/16
n = 4 
res = tp1.getCjWj(n, y, h)
i=1
for image in res:
    plt.figure()
    plt.imshow(image, cmap='gray')
    print('moyenne = ',np.mean(image))
    #path = 'galaxie_w_' + str(i) + '.jpg' 
    #scipy.misc.imsave(path, image)
    i =i +1

#ensuite on calcule sigma MAD
def getSigmaMAD(w0):
    lho = 1.48826
    median = np.median(w0)
    sigmaMAD = lho * np.median(np.abs(w0 - median))
    return sigmaMAD
w0 = res[0]
sigmaMAD = getSigmaMAD(w0)
print("sigmaMAD = " + str(sigmaMAD))

#############################################################
########################## calcul du seuil pour chaque wavelet

def getDetectionLevels(y, k, nbLevels):
    h = np.array([1, 4, 6, 4, 1])/16
    wavelets = tp1.getCjWj(nbLevels, y, h)
    wavelets.pop()
    res = []
    for w in wavelets:
        sigmaMAD = getSigmaMAD(w)
        res.append(k*sigmaMAD)
    return res
k = 3
nbLevels = 5
sigmasMAD = getDetectionLevels(y, k, nbLevels)
print("sigmasMAD : ")
print(sigmasMAD) 

#############################################################
######################### derive the multiresolution mask



####################################################################
#########Denoising with sparsity constraint in the starlet transform
####################################################################

#############################################################
################# implémentation de soft et hard thresholding

def softThrd(x, gamma):
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)

def hardThrd(x, gamma):
    bool0 = x > -np.sqrt(gamma)
    bool1 = x < np.sqrt(gamma)
    bool2 = np.logical_and(bool0, bool1)
    res = x * np.logical_not(bool2)
    return res

############################################################
################# reconstruction à partir de l'image bruitée
    
def reconstruction(y, n, h, k, prior):
    CjWj = tp1.getCjWj(n, y, h)
    cn = CjWj.pop()
    res = []
    for w in CjWj:
        gamma = k * getSigmaMAD(w)
        if(prior == "softThrd"):
            sparseW = softThrd(w, gamma)
        elif(prior == "hardThrd"):
            sparseW = hardThrd(w, gamma)
        else:
            print("Error this prior is not implemented")
        res.append(sparseW)
        plt.figure()
        plt.imshow(w, cmap='gray')
        plt.figure()
        plt.imshow(sparseW, cmap='gray')
    res.append(cn)
    reconst = tp1.reconstruct(res)
    return reconst

h = np.array([1, 4, 6, 4, 1])/16
n = 4
k=3

softReconst = reconstruction(y, n, h, k, "softThrd")
hardReconst = reconstruction(y, n, h, k, "hardThrd")
plt.figure()
plt.imshow(y, cmap='gray')
plt.figure()
plt.imshow(xStar, cmap='gray')
plt.figure()
plt.imshow(softReconst, cmap='gray')
plt.figure()
plt.imshow(hardReconst, cmap='gray')

###########################################################
############## implémentation du masque multirésolution

# à faire

###########################################################
############# comparaison des méthodes

def error(xStar, xHat):
    return np.linalg.norm(xStar - xHat) / np.linalg.norm(xStar)

print(error(xStar, xStar))
print(error(xStar, y))
print(error(xStar, softReconst))
print(error(xStar, hardReconst))
#grosse erreur de soft reconstruction
#à vérifier !

###########################################################
############## sparsity based deblurring




