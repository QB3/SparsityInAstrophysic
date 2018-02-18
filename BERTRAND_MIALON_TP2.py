import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from astropy.io import fits
import scipy.misc
from scipy import signal
import sys
sys.path.insert(0, '~/Bureau/SparsityEnAstrophysique/SparsityInAstrophysic/')
import BERTRAND_MIALON_TP1 as tp1


from scipy.signal import convolve as scipy_convolve
from astropy.convolution import convolve


###########################################################
#on charge les données
hdul = fits.open('qbe/Bureau/SparsityEnAstrophysique/SparsityInAstrophysic/simu_sky.fits')
hdul.info()
xStar = hdul[0].data
plt.figure()
plt.imshow(xStar, cmap='gray')
path = "qbe/Bureau/SparsityEnAstrophysique/SparsityInAstrophysic/imagesTP2/simu_sky.jpg"
scipy.misc.imsave(path, xStar)

#########################################################################
############## Estimation of a multiresolution mask ####################
#########################################################################

###########################################################"
#on ajoute un bruit gaussien
n = xStar.shape[0]
sigma = 500
y = xStar + np.random.normal(0, sigma, (n, n))
plt.figure()
plt.imshow(y, cmap='gray')
path = "qbe/Bureau/SparsityEnAstrophysique/SparsityInAstrophysic/imagesTP2/noisy_simu_sky.jpg"
scipy.misc.imsave(path, y)

###########################################################
########################### calcul de sigma MAD

#d'abord on calcule la tranformée en ondelettes
h = np.array([1, 4, 6, 4, 1])/16
nbLevel = 4 
res = tp1.getCjWj(nbLevel, y, h)
i=1
for image in res:
    plt.figure()
    plt.imshow(image, cmap='gray')
    print('moyenne = ',np.mean(image))
    #path = 'qbe/Bureau/SparsityEnAstrophysique/SparsityInAstrophysic/galaxie_w_' + str(i) + '.jpg' 
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
        """plt.figure()
        plt.imshow(w, cmap='gray')
        plt.figure()
        plt.imshow(sparseW, cmap='gray')"""
    res.append(cn)
    reconst = tp1.reconstruct(res)
    return reconst

h = np.array([1, 4, 6, 4, 1])/16
n = 4
k=3

softReconst = reconstruction(y, n, h, k, "softThrd")
hardReconst = reconstruction(y, n, h, k, "hardThrd")
plt.figure()
plt.title('image non bruitée', fontsize=18)
plt.imshow(xStar, cmap='gray')
plt.figure()
plt.title('image + bruit additif gaussien', fontsize=18)
plt.imshow(y, cmap='gray')
path = 'qbe/Bureau/SparsityEnAstrophysique/SparsityInAstrophysic/imagesTP2/simu_sky_noisy_gauss.jpg'
scipy.misc.imsave(path, y)
plt.figure()
plt.title('image debruitée par softThrd', fontsize=18)
plt.imshow(softReconst, cmap='gray')
path = 'qbe/Bureau/SparsityEnAstrophysique/SparsityInAstrophysic/imagesTP2/simu_sky_noisy_gauss_soft_reconst.jpg'
scipy.misc.imsave(path, softReconst)
plt.figure()
plt.title('image debruitée par hardThrd', fontsize=18)
plt.imshow(hardReconst, cmap='gray')
path = 'qbe/Bureau/SparsityEnAstrophysique/SparsityInAstrophysic/imagesTP2/simu_sky_noisy_gauss_hard_reconst.jpg'
scipy.misc.imsave(path, hardReconst)

###########################################################
############# comparaison des méthodes

def error(xStar, xHat):
    return np.linalg.norm(xStar - xHat) / np.linalg.norm(xStar)

print(error(xStar, xStar))
print("erreur(xStar, y) : ", error(xStar, y))
print("erreur(xStar, softReconst) : ", error(xStar, softReconst))
print("erreur(xStar, hardReconst) : ", error(xStar, hardReconst))
#grosse erreur de soft reconstruction
#à vérifier !

###########################################################
############## sparsity based deblurring

####################################################
# on charge la matrice de convolution
"""hdul = fits.open('simu_psf.fits')
hdul.info()
H = hdul[0].data
plt.figure()
plt.imshow(H, cmap='gray')
"""
###################################
#on convolue
"""
convol = np.dot(H, xStar)
plt.figure()
plt.imshow(convol, cmap='gray')
"""

###############################
# il faut passer en Fourier
# et faire une multiplication entrée par entrée



######################################################
##Applications of proximal algorithms to inpainting

#fonction pour créer un masque avec p coefficient nuls choisi au hasard
def getMask(p, n):
    mask = np.array([0] * p + [1] * (n**2-p))
    np.random.shuffle(mask)
    mask = np.reshape(mask, (n,n))
    return mask

#on bruite l'image
n = xStar.shape[0]
p = 20000
mask = getMask(p, n)
sigma = 50
y = mask * (xStar + np.random.normal(0, sigma, (n,n)))
plt.figure()
plt.title('image non bruitée', fontsize=18)
plt.imshow(xStar, cmap='gray')
plt.figure()
plt.title('image bruitée (inpainting)', fontsize=18)
plt.imshow(y, cmap='gray')

#on calcule la fonction gradient
def getGrad(alpha, mask, y, nLevel, h):
    res = tp1.reconstruct(alpha)
    res = mask * res -y
    res = mask *res
    res = tp1.getCjWj(nLevel, res, h)
    return res

def multiply(alphas, gamma):
    copyAlphas = list(alphas)
    cn = copyAlphas.pop()
    res = []
    for image in copyAlphas:
        newElement = gamma*image
        res.append(newElement)  
    res.append(cn)
    return res

def diffLists(liste1, liste2):
    l1 = len(liste1)
    l2 = len(liste2)
    if(l1!=l2):
        print(" lists do not have the same length")
    res = []
    for im1, im2 in zip(liste1, liste2):
        newElement = im1 - im2
        res.append(newElement)
    return res

def add(liste1, liste2):
    l1 = len(liste1)
    l2 = len(liste2)
    if(l1!=l2):
        print(" lists do not have the same length")
    res = []
    for im1, im2 in zip(liste1, liste2):
        newElement = im1 + im2
        res.append(newElement)
    return res    

def softThrdOnList(yn, gamma):
    liste = list(yn)
    cn = liste.pop()
    res = []
    for image in liste:
        newElement = softThrd(image, gamma)
        res.append(newElement)
    res.append(cn)
    return res    
        
def forwardBackwardInpainting(Niter, alpha0, mask, y, nLevel, h, Lambda):
    alpha = alpha0
    gamma =0.1
    theta = 1.5
    for i in range(Niter):
        grad = getGrad(list(alpha), mask, y, nLevel, h)
        mult = multiply(grad, gamma)
        yn = diffLists(alpha, mult)
        DIFF = diffLists(softThrdOnList(yn, gamma * Lambda), alpha)
        alpha = add(alpha,  multiply(DIFF, theta))
    return alpha

h = np.array([1, 4, 6, 4, 1])/16
nLevel = 4
k=3

softReconst = reconstruction(y, nLevel, h, k, "softThrd")
alpha0 = tp1.getCjWj(nLevel, image, h)
Niter = 50
Lambda = 1

FBalphas = forwardBackwardInpainting(Niter, alpha0, mask, y, nLevel, h, Lambda)
FBreconst = tp1.reconstruct(FBalphas)

################################################################
#on plotte les résultats
plt.figure()
plt.title('image non bruitée', fontsize=18)
plt.imshow(xStar, cmap='gray')
plt.figure()
plt.title('image bruitée (inpainting)', fontsize=18)
plt.imshow(y, cmap='gray')
print(error(xStar, y))
path = 'qbe/Bureau/SparsityEnAstrophysique/SparsityInAstrophysic/imagesTP2/inpainted_simu_sky.jpg'
scipy.misc.imsave(path, y)
plt.figure()
plt.title('image débruitée par softThrd brutal (inpainting)', fontsize=18)
plt.imshow(softReconst , cmap='gray')
print(error(xStar, softReconst))
path = 'qbe/Bureau/SparsityEnAstrophysique/SparsityInAstrophysic/imagesTP2/soft_reconst_simu_sky.jpg'
scipy.misc.imsave(path, softReconst)
plt.figure()
plt.title('image débruitée par FB (inpainting)', fontsize=18)
plt.imshow(FBreconst, cmap='gray')
print(error(FBreconst, softReconst))
path = 'qbe/Bureau/SparsityEnAstrophysique/SparsityInAstrophysic/imagesTP2/FB_reconst_simu_sky.jpg'
scipy.misc.imsave(path, FBreconst)