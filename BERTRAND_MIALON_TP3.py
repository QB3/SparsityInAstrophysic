import numpy as np
import pyBSS as bss
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


#################################################################
#whitening data
def whitening(cov):
    eigenvalues, U= np.linalg.eig(cov)
    print("eigenvalues = ", eigenvalues)
    SigmaMoins05 = np.diag(eigenvalues**(-0.5))
    PI = np.dot(SigmaMoins05, U.T)
    return PI

m = 6
t = 10
X = np.random.rand(m,t)
cov = np.dot(X, X.T)
PI = whitening(cov)
Xtilde = np.dot(PI, X)
#here we have to be carefull if the eigenvalues are too small
#there may have errors
print(np.dot(Xtilde, Xtilde.T))

###################################################################
#Best rank approximation
def getBestRankApprox(X, n):
    P, D, Q = np.linalg.svd(X, full_matrices=False)
    print("D =", D)
    Dprime = np.concatenate([np.ones(n), np.zeros(D.shape[0]-n)])
    print("n = ",n)
    Dprime = D * Dprime
    
    a = P.shape[0]
    b = Q.shape[0]
    
    middleMatrix = np.zeros((a, b))
    for i in range(D.shape[0]):
        middleMatrix[i,i] = Dprime[i]
    
    res = np.dot(P, middleMatrix)
    res = np.dot(res, Q)
    return res
    
rank =  np.linalg.matrix_rank(X)
print("rank = ", rank)
res = getBestRankApprox(X, rank-1)
print("rank(bestApprox) =", np.linalg.matrix_rank(res))

#############################################################
#apply PCA to random mixture of gaussians
#on génère et on plot les points
X, A, S = bss.GenerateMixture(n = 3, m=5)
plt.figure()
plt.scatter(X[0,:], X[1,:])

#attention à bien prendre la transposée dans la formule
pca = PCA(n_components=3)
pca.fit(X.T)
print("explained variance ratio = ", pca.explained_variance_ratio_)  
print("singulatr values", pca.singular_values_)  

#########################################################################################
##########################################################################################
# Independent component analysis

#SType = 2 to generate unifrorm variables
n = 2
m = 2
SType = 2
X, A, S = bss.GenerateMixture(n = n, m=m, SType = SType)
Aica, Sica = bss.Perform_FastICA(X, n)
print("A.shape = ", A.shape)
print("Aica.shape = ", Aica.shape)
print("A.shape = ", S.shape)
print("Sica.shape = ", Sica.shape)
#j'obtiens un warning FastICA did not converge si je ne précise pas n_component = n dans la FICA
print("ration Sica[1,:] / S[0,:] = ", Sica[1,:] / S[0,:])
print("ration Sica[0,:] / S[1,:] = ", Sica[0,:] / S[1,:])
#on observe que S et Sica sont presque identique à une permutation près et un facteur d'chelle 0.11
print("ration Aica[1,:] / A[0,:] = ", A[:,0]/Aica[:,1]) 
print("ration Aica[0,:] / A[1,:] = ", A[:,1]/Aica[:,0])
#on observe que S et Sica sont presque identique à une permutation près et un facteur d'chelle 0.11
#la fonction bss eval peut faire tout ça 


####################################################################"
#evaluate the performance whane the conditionning number of the mmixing matrix varies
listCdA = [1,2,10,50,100] 
nSample = 1000

for CdA in listCdA:
    counter = 0
    for i in range(nSample):
        X, A, S = bss.GenerateMixture(n = n, m=m, SType = SType, CdA=CdA)
        Aica, Sica = bss.Perform_FastICA(X, n)
        counter = counter + bss.Eval_BSS(A,S,Aica,Sica)
    print("CdA = ", CdA, " : mean bss.Eval", counter/nSample)
    

####################################################################
#Chandra 
#chargement des données 2
import scipy.io
mat = scipy.io.loadmat('chandra.mat')

A0 = mat['A0']
S0 = mat['S0']
X = np.dot(A0, S0)
n= 8
Aica, Sica = bss.Perform_FastICA(X, n)
bss.Eval_BSS(A0,S0,Aica,Sica)
#on obtient un warning, le nombre de composant est trop large

############################################################################################
###########################################################################################
#Sparse blind source separation

#GMCA unif sources
n = 2
m = 2
SType = 2
X, A, S = bss.GenerateMixture(n = n, m=m, SType = SType)
Agmca,Sgmca,PinvAgmca = bss.Perform_GMCA(X, n)
print("bss.Eval = ", bss.Eval_BSS(A, S, Agmca, Sgmca))
#Could you comment on the results ??

#question : pourquoi on a ça ?
fig = plt.figure()
plt.scatter(Sgmca[0], Sgmca[1])
plt.gcf().subplots_adjust(bottom=0.15)
fig.savefig('ImagesTP3/GMCAunifData.png')
#la solution de GMCA est "mauavaise" et renvoie qqchose dans la boule L1


#GMCA sparse sources
n = 2
m = 2
SType = 3
X, A, S = bss.GenerateMixture(n = n, m=m, SType = SType)
Agmca,Sgmca,PinvAgmca = bss.Perform_GMCA(X, n)
print("bss.Eval = ", bss.Eval_BSS(A, S, Agmca, Sgmca))
#Could you comment on the results ??

# on remarque que al reconstruction dans le cas sparese est bien meilleure que dans le cas uniforme

################################
#
nSample = 100
listSNR = [40,30,20,10]
for SNR in listSNR:
    counterICA = 0
    counterGMCA = 0
    for i in range(nSample):
        X, A, S = bss.GenerateMixture(n = n, m=m, SType = SType, noise_level = SNR)
        Aica, Sica = bss.Perform_FastICA(X, n)
        counterICA = counterICA + bss.Eval_BSS(A,S,Aica,Sica)
        Agmca, Sgmca, PinvA = bss.Perform_GMCA(X, n)
        counterGMCA = counterGMCA + bss.Eval_BSS(A,S,Agmca,Sgmca)        
    print("SNR = ", SNR, " : mean bss.Eval ICA", counterICA/nSample)
    print("SNR = ", SNR, " : mean bss.Eval GMCA", counterGMCA/nSample)




#####################################################################################################"
######################################################################################################


#on charge Chandra 
import scipy.io
mat = scipy.io.loadmat('chandra.mat')
Areal = mat['A0']
Sreal = mat['S0']
X = np.dot(Areal, Sreal)
n=8

#on applique GMCA à Chandra

counter = 0
nSample = 100
for i in range(nSample):
    Agmca,Sgmca,PinvAgmca = bss.Perform_GMCA(X, n)
    counter = counter + bss.Eval_BSS(Areal, Sreal, Agmca, Sgmca)
print(" mean bss.Eval, nSample = ", nSample, " : mean bss.Eval", counter/nSample)


#implémentation de PALM
import Starlet2D as tp1

def getGrad1(X, A, S):
    return np.dot(A.T, np.dot(A, S)-X) 

def getGrad2(X, A, S):
    return np.dot(np.dot(A, S)-X, S.T)

def getLip1(A):
    return np.linalg.norm(np.dot(A.T, A), ord = 'fro')

def getLip2(S):
    return np.linalg.norm(np.dot(S, S.T), ord = 'fro')


def softThrd(x, gamma):
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)

def Starlet_Forward2D_Multidim(S=0,h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3):
    (n, t)= S.shape
    tailleIm = int(np.sqrt(t))
    #on reshape les sources en image pour appliquer Starlet 2D
    newS = S.reshape((n, tailleIm, tailleIm))
    cMultidim = np.zeros((n, tailleIm, tailleIm))
    wMultidim = np.zeros((n, tailleIm, tailleIm, J))
    for i in range(n):
        cMultidim[i] , wMultidim[i,:,:,:]=  tp1.Starlet_Forward2D(newS[i, :, :], h,J,boption)
    return cMultidim, wMultidim

def Starlet_Backward2D_Multidim(cMultidim,wMultidim):
    (n, tailleIm, tailleIm, J) = wMultidim.shape
    SformatImage = np.zeros((n, tailleIm, tailleIm))
    for i in range(n):
        SformatImage[i,:,:] = tp1.Starlet_Backward2D(cMultidim[i], wMultidim[i,:,:,:])
    SformatLigne = SformatImage.reshape((n, tailleIm**2))
    return SformatLigne

def projOnBall(A):
    n, m = A.shape
    res = np.zeros((n,m))
    for i in range(m):
        Ci = A[:,i]
        norm = np.linalg.norm(Ci)
        if (norm <= 1):
            res[:,i] = Ci
        else:
            res[:,i] = Ci/norm
    return res
        
        
#attention il faut imposer des contraintes supplémentaires 
#A \in O_b : chacune des colonnes de A est normalisée en norme L2
#il faut donc rajouter un terme de contrainte : (d'où les 3 termes dans PALM)
def palm(nIter, X, A0, S0, gamma1 = 2, gamma2 = 2, Lambda = 1):
    import copy as cp    
    A = cp.copy(A0)
    S = cp.copy(S0)
    for i in range(nIter):
        ci = gamma2 * getLip2(S)
        grad2 = getGrad2(X, A, S)
        A = A - 1/ci * grad2
        A = projOnBall(A)
        di = gamma1 * getLip2(A)
        S = S - 1/di * getGrad1(X, A, S)
        cMultidim, wMultidim = Starlet_Forward2D_Multidim(S, J=2)
        wMultidim = softThrd(wMultidim, Lambda * di)
        S = Starlet_Backward2D_Multidim( cMultidim, wMultidim)
        print("evaluation bss = ", bss.Eval_BSS(Areal, Sreal, A, S))
    return A, S


dimA = Areal.shape
A0 = np.random.normal(0,1,dimA)
dimS = Sreal.shape
S0 = np.random.normal(0,1, dimS)
nIter = 1000

A, S = palm(nIter, X, A0, S0, gamma1=1, gamma2=1, Lambda= 0.01)
A0=A
S0= S

