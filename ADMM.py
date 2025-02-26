import GaussianDenoiser as gd
import numpy as np
import scipy

def ADMM(im, beta,lamb, niter, L=1, CNNprior = None, verbose=False):
    "computes the denoising of im (log of intensity) with ADMM"


    #INITIALIZE PARAMETERS
    alpha=1  # ADDED : coefficient in the data term
    x = im.astype(float)
    z  = im.astype(float) #np.zeros(im.shape)

    d = im.astype(float)#np.zeros(im.shape)
    E = []
    for k in range(niter):
        #compute the proximal of the regularisation term
        if (CNNprior is not None):
            
            alpha = 1.0/lamb # it is a constant in the data term.
            
            if(verbose):
                print("iteration for the CNN: "+str(k))
            v=x-d
            shape = v.shape
            C = np.max(v)-np.min(v)
            if(C != 0):
                b = v.min()/C
                v = v/C - b
            z, _, _ = CNNprior.denoise(v.reshape(1,shape[1], shape[0],1))
            z = z.reshape(shape)
            if(C != 0):
                z = C*(z+b)


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
            # add the constant L*alpha
            xtmp -= np.divide(beta*(xtmp-a)+L*alpha*(np.ones(im.shape)-np.exp(im-xtmp)),
                              beta*np.ones(im.shape)+L*alpha*np.exp(im-xtmp))
#             print ( np.sum((xtmp-a)**2)+np.sum(xtmp)+np.sum(np.exp(im-xtmp)))
        x = xtmp
        #compute the Energy
        gradx, grady = gd.grad(x)
        e = np.sum(x)+np.sum(np.exp(im-x))+ lamb * np.sum(np.sqrt(gradx**2 + grady**2))
        E.append(e)

    return x, E
