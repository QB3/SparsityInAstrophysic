import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from astropy.io import fits
import scipy.misc
from scipy import signal
import Starlet2D as tp1
import copy as cp  
from copy import deepcopy as dp

from scipy.signal import convolve as scipy_convolve
from astropy.convolution import convolve


###########################################################
#on charge les données
hdul = fits.open('simu_sky.fits')
hdul.info()
xStar = hdul[0].data
plt.figure()
plt.imshow(xStar, cmap='gray')
path = "imagesTP2/simu_sky.jpg"
scipy.misc.imsave(path, xStar)

###########################################################"
#on ajoute un bruit gaussien
n = xStar.shape[0]
sigma = 500
y = xStar + np.random.normal(0, sigma, (n, n))
plt.figure()
plt.imshow(y, cmap='gray')
path = "imagesTP2/noisy_simu_sky.jpg"
scipy.misc.imsave(path, y)

###################################################"
#on crée la fonction pour calculer automatiquement le MAD
def getSigmaMAD(w0):
    lho = 1.48826
    median = np.median(w0)
    sigmaMAD = lho * np.median(np.abs(w0 - median))
    return sigmaMAD

################################################""
#on teste le MAD
#d'abord on calcule la tranformée en ondelettes
nbLevel = 4 
(c, w) = tp1.Starlet_Forward2D(x = y, J=nbLevel)
for i in range(nbLevel):
    image = w[:,:,i]
    plt.figure()
    title = "w" + str(i) 
    plt.title(title, fontsize=18)
    plt.imshow(image , cmap='gray')
    sigmaMAD = getSigmaMAD(image)
    print("sigmaMAD = " + str(sigmaMAD))


####################################################################
#########Denoising with sparsity constraint in the starlet transform
####################################################################

#############################################################
################# implémentation de soft et hard thresholding

def softThrd(x, gamma):
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)

def hardThrd(x, gamma):
    bool0 = x > -gamma
    bool1 = x < gamma
    bool2 = np.logical_and(bool0, bool1)
    res = x * np.logical_not(bool2)
    return res

############################################################
################# reconstruction à partir de l'image bruitée

def reconstruction(y, nbLevel, k, prior):
    (c, w ) = tp1.Starlet_Forward2D(x = y, J=nbLevel)
    for i in range(nbLevel):
        wi = w[:,:,i]
        gamma = k * getSigmaMAD(wi)
        if(prior == "softThrd"):
            sparseW = softThrd(wi, gamma)
        elif(prior == "hardThrd"):
            sparseW = hardThrd(wi, gamma)
        else:
            print("Error this prior is not implemented")
        w[:,:,i] = sparseW
        """plt.figure()
        plt.imshow(w, cmap='gray')
        plt.figure()
        plt.imshow(sparseW, cmap='gray')"""
    reconst = tp1.Starlet_Backward2D(c, w)
    return reconst

h = np.array([1, 4, 6, 4, 1])/16
nLevel = 4
k=3

softReconst = reconstruction(y, nLevel, k, "softThrd")
hardReconst = reconstruction(y, nLevel, k, "hardThrd")
plt.figure()
plt.title('image non bruitée', fontsize=18)
plt.imshow(xStar, cmap='gray')
plt.figure()
plt.title('image + bruit additif gaussien', fontsize=18)
plt.imshow(y, cmap='gray')
path = 'imagesTP2/simu_sky_noisy_gauss.jpg'
scipy.misc.imsave(path, y)
plt.figure()
plt.title('image debruitée par softThrd', fontsize=18)
plt.imshow(softReconst, cmap='gray')
path = 'imagesTP2/simu_sky_noisy_gauss_soft_reconst.jpg'
scipy.misc.imsave(path, softReconst)
plt.figure()
plt.title('image debruitée par hardThrd', fontsize=18)
plt.imshow(hardReconst, cmap='gray')
path = 'imagesTP2/simu_sky_noisy_gauss_hard_reconst.jpg'
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
#déconcolution

####################################################
# on charge la matrice PSF
hdul = fits.open('simu_psf.fits')
hdul.info()
H = hdul[0].data
plt.figure()
plt.imshow(H, cmap='gray')
path = 'imagesTP2/PSF_matrix.jpg'
scipy.misc.imsave(path, H)

###################################
#on va convoluer
# on crée un fonction de convoultion rapide

def convol(xStar, H):
    Hfft = np.fft.fft2(H, norm = "ortho")
    xfft = np.fft.fft2(xStar, norm = "ortho")
    Hxfft = Hfft * xfft
    Hx = np.fft.ifft2(Hxfft, norm = "ortho")
    #il y a un petit résidu complexe de l'ordre de 10**-13, on le supprime
    Hx = np.real(Hx) 
    return Hx

#####################################
# on convolue
n = xStar.shape[0]
sigma = 1
Hx = convol(xStar, H)
plt.figure()
plt.title('image non convoluée', fontsize=18)
plt.imshow(xStar, cmap='gray')
path = 'imagesTP2/simu_sky.jpg'
scipy.misc.imsave(path, xStar)
plt.figure()
plt.title('image convoluée', fontsize=18)
plt.imshow(Hx, cmap='gray')
path = 'imagesTP2/simu_sky_concoluee.jpg'
scipy.misc.imsave(path, Hx)

y = Hx + np.random.normal(0,sigma, (n,n))
plt.figure()
plt.title('image convoluée et bruitée', fontsize=18)
plt.imshow(y, cmap='gray')
path = 'imagesTP2/simu_sky_concoluee_bruitee.jpg'
scipy.misc.imsave(path, Hx)

#####################################"
#on implémente gradient on applique le Forward-Backward

def getGradConvol(c ,w , H, Htilde, y, nLevel, h): 
    x = tp1.Starlet_Backward2D(c, w)
    gradx = convol(x, H) - y
    gradx = convol(gradx , Htilde)
    (gradC, gradW) = tp1.Starlet_Forward2D(gradx, J= nLevel)
    return (gradC, gradW)


def getHtilde(H):
    n = H.shape[0]
    res = np.zeros((n,n))
    for i in range(n):
        res[:, i] = H[:, n-i-1]
    for i in range(n):
        res[i,:] = res [n-i-1,:]
    return res

def getLipConst(H, Htilde):
    nu = np.fft.fft(H) * np.fft.fft(Htilde)
    nu= np.max(np.linalg.norm(nu))
    return nu


Htilde = getHtilde(H)
nu = getLipConst(H, Htilde)
print(nu)


def forwardBackwardConvol(Niter, x0, H, y, nLevel, Lambda):
    Htilde = getHtilde(H)
    nu = getLipConst(H, Htilde)
    gamma =nu/2
    theta = 1.5
    x = cp.copy(x0)
    (c, w ) = tp1.Starlet_Forward2D(x, J = nLevel)
    
    for i in range(Niter):
        print("iteration " + str(i+1) + "/" + str(Niter) )
        (gradC, gradW) = getGradConvol(c, w, H, Htilde, y, nLevel, h)
        wHalf = w - gamma * gradW
        c = c - gamma * gradC
        w = w + theta  * (softThrd(wHalf, gamma * Lambda) -w)
        print("error = ", error(xStar, tp1.Starlet_Backward2D(c,w)))
    res = tp1.Starlet_Backward2D(c, w)
    return res


n = xStar.shape[0]
nLevel = 3
k=3
######################################"
#on fixe le paramètre Lambda avec le niveau de bruit
Lambda = getSigmaMAD(y)
print("Lambda =  ", Lambda)

x0 = reconstruction(y, nLevel, k, "softThrd")
Niter = 20
FBreconst = forwardBackwardConvol(Niter, x0, H, y, nLevel, Lambda)

################################################################
#on plotte les résultats
plt.figure()
plt.title('image non bruitée', fontsize=18)
plt.imshow(xStar, cmap='gray')
plt.figure()
plt.title('image bruitée (convolution)', fontsize=18)
plt.imshow(y, cmap='gray')
path = 'imagesTP2/convol_simu_sky.jpg'
scipy.misc.imsave(path, y)
plt.figure()
plt.title('image débruitée par softThrd brutal (convolution)', fontsize=18)
plt.imshow(x0, cmap='gray')
path = 'imagesTP2/convol_soft_reconst_simu_sky.jpg'
scipy.misc.imsave(path, softReconst)
plt.figure()
plt.title('image débruitée par FB (convolution)', fontsize=18)
plt.imshow(FBreconst, cmap='gray')
path = 'imagesTP2/convol_FB_reconst_simu_sky.jpg'
scipy.misc.imsave(path, FBreconst)
print("erreur xStar y", error(xStar, y))
print("erreur xStar softReconst", error(xStar, x0))
print("erreur xStar FB", error(xStar, FBreconst))


#########################################################""
#l'impainting

#####################################################"
#on crée une fonction pour crée un maque binaire
def getMask(p, n):
    mask = np.array([0] * p + [1] * (n**2-p))
    np.random.shuffle(mask)
    mask = np.reshape(mask, (n,n))
    return mask

#on bruite l'image
n = xStar.shape[0]
p = np.floor(n**2/3)
mask = getMask(p, n)
sigma = 50
y = mask * (xStar + np.random.normal(0, sigma, (n,n)))
plt.figure()
plt.title('image non bruitée', fontsize=18)
plt.imshow(xStar, cmap='gray')
plt.figure()
plt.title('image bruitée (inpainting)', fontsize=18)
plt.imshow(y, cmap='gray')

##########################################################""
#on implemente forward backward pour l'inpainting
def getGradInpainting(c,w, mask, y, nLevel):  
    x = tp1.Starlet_Backward2D(c, w)
    res = mask * (x - y)
    (gradC, gradW) = tp1.Starlet_Forward2D(x = res, J= nLevel)
    return (gradC, gradW)

#on fixe les paramètres
nLevel = 3
k=3
######################################"
#on fixe le paramètre Lambda avec le niveau de bruit
ligney = y.reshape(y.shape[1]**2,)
yForMAD = ligney[np.where(ligney  !=0)]
Lambda = getSigmaMAD(yForMAD)
print("Lambda =  ", Lambda)

x0 = reconstruction(y, nLevel, k, "softThrd")

def forwardBackwardInpainting(Niter, x0, mask, y, nLevel, Lambda):
    nu = 1
    gamma =nu/2
    theta = 1.5
    xOld = cp.copy(x0)
    (c, w ) = tp1.Starlet_Forward2D(x = xOld, J = nLevel)
    
    for i in range(Niter):
        print("iteration " + str(i+1) + "/" + str(Niter) )
        (gradC, gradW) = getGradInpainting(c, w, mask, y, nLevel)
        wHalf = w - gamma * gradW  
        c = c - gamma *gradC
        w = w + theta  * (softThrd(wHalf, gamma * Lambda) - w)
        print("error = ", error(xStar, tp1.Starlet_Backward2D(c,w )))
    res = tp1.Starlet_Backward2D(c, w)
    return res

Niter = 50
FBreconst = forwardBackwardInpainting(Niter, x0, mask, y, nLevel, Lambda)
#FBreconst = forwardBackwardInpainting(Niter, FBreconst, mask, y, nLevel, Lambda)

plt.figure()
plt.title('image non bruitée', fontsize=18)
plt.imshow(xStar, cmap='gray')
plt.figure()
plt.title('image bruitée (inpainting)', fontsize=18)
plt.imshow(y, cmap='gray')
path = 'imagesTP2/impainting_simu_sky.jpg'
scipy.misc.imsave(path, y)
plt.figure()
plt.title('image débruitée par softThrd brutal (inpainting)', fontsize=18)
plt.imshow(x0, cmap='gray')
path = 'imagesTP2/impainting_soft_reconst_simu_sky.jpg'
scipy.misc.imsave(path, softReconst)
plt.figure()
plt.title('image débruitée par FB (inpainting)', fontsize=18)
plt.imshow(FBreconst, cmap='gray')
path = 'imagesTP2/impainting_FB_reconst_simu_sky.jpg'
scipy.misc.imsave(path, FBreconst)
print("erreur xStar y", error(xStar, y))
print("erreur xStar softReconst", error(xStar, x0))
print("erreur xStar FB", error(xStar, FBreconst))
