import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from astropy.io import fits
import scipy.misc

def getIndices(k, l, j, tailleFiltre, tailleImage):
    listeIndiceLigne = [(k + 2**j * (i1 - tailleFiltre//2) )%tailleImage for i1 in range(tailleFiltre)]
    listeIndiceColonne = [(l + 2**j * (i2 - tailleFiltre//2) )%tailleImage for i2 in range(tailleFiltre)]
    return [listeIndiceLigne, listeIndiceColonne]

def getIndicesMirrorConditions(k, l, j, tailleFiltre, tailleImage):
    listeIndiceLigne = [(k + 2**j * (i1 - tailleFiltre//2) )%tailleImage *( (k + 2**j * (i1 - tailleFiltre//2) )//tailleImage%2 ==0 ) + (tailleImage-(k + 2**j * (i1 - tailleFiltre//2) )%tailleImage-1)*( (k + 2**j * (i1 - tailleFiltre//2) )//tailleImage%2 ==1 )   for i1 in range(tailleFiltre)]
    listeIndiceColonne = [(l + 2**j * (i2 - tailleFiltre//2) )%tailleImage *( (l + 2**j * (i2 - tailleFiltre//2) )//tailleImage%2 ==0 ) + (tailleImage-(l + 2**j * (i2 - tailleFiltre//2) )%tailleImage-1)*( (l + 2**j * (i2 - tailleFiltre//2) )//tailleImage%2 ==1 )   for i2 in range(tailleFiltre)]
    return [listeIndiceLigne, listeIndiceColonne]
    
#la fonction ci-dessous effectue la convolution 2D de l'image cOld avec le filtre h * h
# h est un filtre 1D
#ici la condition aux limites est périodique
def getNewCj(j, cOld, h):
    tailleFiltre = h.shape[0]
    tailleImage = cOld.shape[0]
    #cNew est le tableau contenant les nouveaux coefficients
    cNew = np.zeros([tailleImage, tailleImage])    
    #on parcourt l'imgae selon les lignes et les colonnes
    for k in range(tailleImage):
        for l in range(tailleImage):
            #ici on extrait la matrice à partir de laquelle on calcule la convolution
            #on prend le reste de la division euclidienne, nous appliquons donc des conditions de bords périodiques
            [listeIndiceLigne, listeIndiceColonne] = getIndices(k, l, j, tailleFiltre, tailleImage)
            #[listeIndiceLigne, listeIndiceColonne] = getIndicesMirrorConditions(k, l, j, tailleFiltre, tailleImage)
            convol = cOld[listeIndiceLigne, :]
            convol = convol[:, listeIndiceColonne]        
            #là on applique l'opération de convolution avec les filre
            convol = convol * h
            convol = (convol.T * h).T
            #on stocke le résultat dans le nouveau coefficient
            cNew[k,l] = np.sum(convol)
    return cNew

##########################################
#faire la convolition avec scipy
#grad = sig.convolve2d(data, , boundary='symm', mode='same')


#cette fonction implémente l'algorihme "A trous" foreward
#cette fonction renvoie la liste (w0, w1, ..., wn, cn)
def getCjWj(n, image, h):
    cOld = image
    res= list()
    cNew = 0
    for j in range(n):
        cNew = getNewCj(j, cOld, h)
        wNew = cOld - cNew
        cOld = cNew
        res.append(wNew)
    res.append(cNew)
    return(res)

#cette fonction reconstruit l'image à partir de la liste des w et h
def reconstruct(res):
    copyImReconst = list(res)
    imReconst = copyImReconst.pop()
    for im in copyImReconst:
        imReconst = imReconst + im
    return imReconst

"""
hdul = fits.open('ngc2997.fits')
hdul.info()
data = hdul[0].data
scipy.misc.imsave('galaxie.jpg', data)

#########################################################################################
#tests pour vérifier que tout marche
#on défini le filtre
h = np.array([1, 4, 6, 4, 1])/16

#test sur la galaxie
CijOld = data
j = 0
getCjWj(j, data, h)

n = 3
res = getCjWj(n, data, h)
i = 1
for image in res:
    plt.figure()
    plt.imshow(image, cmap='gray')
    print('moyenne = ',np.mean(image))
    path = 'galaxie_w_' + str(i) + '.jpg' 
    scipy.misc.imsave(path, image)
    i =i +1

#la reconstruction
imReconst=reconstruct(res)
plt.figure()
plt.imshow(imReconst, cmap='gray')
st=reconstruct(res)
plt.figure()
plt.imshow(imReconst, cmap='gray')

plt.figure()
plt.imshow(data, cmap='gray')
##################################################################################################""
#test sur un dirac
dirac = np.zeros((255, 255))
dirac[255//2, 255//2] = 1
n = 3
res = getCjWj(n, dirac, h)
i = 1
for image in res:
    plt.figure()
    plt.imshow(image, cmap='gray')
    print('moyenne = ',np.mean(image))
    path = 'Dirac_w_' + str(i) + '.jpg' 
    scipy.misc.imsave(path, image)
    i =i +1

#la reconstruction
imReconst=reconstruct(res)
plt.figure()
plt.imshow(imReconst, cmap='gray')
st=reconstruct(res)
plt.figure()
plt.imshow(imReconst, cmap='gray')

plt.figure()
plt.imshow(data, cmap='gray')

#######################################################################################
#bruit gaussien
gauss = np.random.normal(0, 1, (256, 256))
plt.figure()
plt.imshow(gauss, cmap='gray')
n = 3
res = getCjWj(n, gauss, h)
i = 1
for image in res:
    plt.figure()
    plt.imshow(image, cmap='gray')
    path = 'bruit_w_' + str(i) + '.jpg' 
    scipy.misc.imsave(path, image)
    print('ecart type = ', np.std(image) )
    i = i+1
    
#méthode automatique pour calculer le niveau du bruit : méthode de la médiane
"""


