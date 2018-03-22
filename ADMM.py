import GaussianDenoiser as gd
import numpy as np
import scipy

def ADMM(im, beta,lamb, niter, CNNprior = False):
    "computes the denoising of im (log of intensity) with ADMM"


    #INITIALIZE PARAMETERS
    x = im.astype(float)
    z  = im.astype(float) #np.zeros(im.shape)

    d = im.astype(float)#np.zeros(im.shape)
    E = []
    for k in range(niter):
        #compute the proximal of the regularisation term
        if (CNNprior):
            print("CNNprior not included yet")


        else :
            #compute it manually for TV regularisation
            v = x-d
            z, _ = gd.TVgaussianDenoiser(v,oldlamb = lamb/(beta*scipy.special.polygamma(1, 1)),niter = 10)

        d = d+z-x

        #compute the data fifelity term
        xtmp = x
        a = z+d
        #NEWTON STEPS
        for j in range(10):
            xtmp -= np.divide(beta*(xtmp-a)+np.ones(im.shape)-np.exp(im-xtmp),beta*np.ones(im.shape)+np.exp(im-xtmp))
#             print ( np.sum((xtmp-a)**2)+np.sum(xtmp)+np.sum(np.exp(im-xtmp)))
        x = xtmp
        #compute the Energy
        gradx, grady = gd.grad(x)
        e = np.sum(x)+np.sum(np.exp(im-x))+np.sum(np.sqrt(gradx**2 + grady**2))
        E.append(e)

    return x, E
