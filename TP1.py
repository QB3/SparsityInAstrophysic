import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from astropy.io import fits

hdul = fits.open('ngc2997.fits')
hdul.info()
data = hdul[0].data

#la fonction ci-dessous effectue la convolution 2D de l'image cOld avec le filtre h * h
# h est un filtre 1D
def getNewCj(j, cOld, h):
    tailleFiltre = h.shape[0]
    tailleImage = CijOld.shape[0]
    #cNew est le tableau contenant les nouveaux coefficients
    cNew = np.zeros([tailleImage, tailleImage])    
    #on parcourt l'imgae selon les lignes et les colonnes
    for k in range(tailleImage):
        for l in range(tailleImage):
            #ici on extrait la matrice à partir de laquelle on calcule la convolution
            #on prend le reste de la division euclidienne, nous appliquons donc des conditions de bords périodiques
            listeIndiceLigne = [(k + 2**j * (i1 - tailleFiltre//2) )%tailleImage for i1 in range(tailleFiltre)]
            listeIndiceColonne = [(l + 2**j * (i2 - tailleFiltre//2) )%tailleImage for i2 in range(tailleFiltre)]
            convol = cOld[listeIndiceLigne, :]
            convol = convol[:, listeIndiceColonne]        
            #là on applique l'opération de convolution avec les filre
            convol = convol * h
            convol = (convol.T * h).T
            #on stocke le résultat dans le nouveau coefficient
            cNew[k,l] = np.sum(convol)
    return cNew

#cette fonction "dialte" le filtre h d'un facteur 2
def extendh(h):
    l = len(h)
    hNew = np.zeros(2*l)
    for i in range(l):
        hNew[2*i] = h[i]
    return h

#cette fonction implémente l'algorihme "A trous" foreward
#cette fonction renvoie la liste (w0, w1, ..., wn, cn)
def getCjWj(n, image, h):
    cOld = image
    hOld = h
    res= list()
    for j in range(n):
        cNew = getNewCj(j, cOld, hOld)
        wNew = cNew - cOld
        cOld = cNew
        hOld = extendh(hOld)
        res.append(wNew)
    res.append(cNew)
    return(res)

#cette fonction reconstruit l'image à partir de la liste des w et h
def reconstruct(res):
    imReconst = res.pop()
    for im in res:
        imReconst = imReconst - im
    return imReconst

#########################################################################################
#tests pour vérifier que tout marche
#
h = np.array([1/16, 1/4, 3/8, 1/4, 1/16])
CijOld = data

j = 3
getCjWj(j, data, h)

n = 3
res = getCjWj(n, data, h)

for image in res:
    plt.figure()
    plt.imshow(image)
    plt.colorbar()

imReconst=reconstruct(res)
    
plt.figure()
plt.imshow(imReconst)
plt.colorbar()

plt.figure()
plt.imshow(data)
plt.colorbar()

#il reste à faire la question avec le dirac