import numpy as np
import scipy
from scipy import special


def grad(im):
    """ returns the gadient of the imput image im
    input:
    ---
        im : a RxC images (one channel)
    returns
    ---
        gradx : gradient on coordinate 0 ( RxC ndarray)
        grady : gradient on coordinate 1 ( RxC ndarray)
    """

    gradx = np.zeros(np.shape(im))
    grady = np.zeros(np.shape(im))

    imx = im[1:,:]
    gradx[:-1,:] = imx - im[:-1,:]

    imy = im[:,1:]
    grady[:,:-1] = imy - im[:,:-1]

    return gradx, grady

def div(gradx,grady):
    """ returns the divergence of an image. Takes as input
    the gradient of the image
    input:
    ---
        gradx : gradient on coordinate 0 ( RxC ndarray)
        grady : gradient on coordinate 1 ( RxC ndarray)
    returns
    ---
        div : divergence of image ( RxC ndarray)
    """
    d = np.zeros(np.shape(gradx))

    d[1:-1,:] = gradx[1:-1,:]-gradx[:-2,:]
    d[0,:] = gradx[0,:]
    d[-1,:] = - gradx[-2,:]

    d[:,1:-1] = d[:,1:-1] + grady[:,1:-1]-grady[:,:-2]
    d[:,0] = d[:,0] + grady[:,0]
    d[:,-1] = d[:,-1] - grady[:,-2]

    return d


def TVgaussianDenoiser(im,oldlamb,niter):
    """ returns the denoised image with a gaussian denoiser with
    a  TV regularization
    input:
    ---
        im : image to be denoised
        lambda : coefficient attached to the regularization
        niter : number of iteration
    returns
    ---
        u : denoised image
        E : enregy at each iteration
    """
    #INITIALIZE PARAMETERS
    nx, ny = np.shape(im)
    E = np.zeros(niter)
    ubar = im.copy()
    u = im.copy()
    lamb = oldlamb*scipy.special.polygamma(1, 1)
    tau = 0.99/(np.sqrt(8)*lamb)
    sigma = 0.99/(np.sqrt(8)*lamb)
    theta = 1
    # print ( theta )
    px = np.zeros((nx,ny))
    py = np.zeros((nx,ny))

    #COMPUTE The chambolle and Pock steps
    for k in range(niter):
        gradx, grady = grad(ubar)
        px += sigma *  lamb * gradx
        py += sigma * lamb * grady


        norm = np.multiply(px,px) + np.multiply(py,py)
        normtest = norm.copy()
        norm[norm <= 1] = 1.

        px= np.divide(px,np.sqrt(norm))
        py= np.divide(py,np.sqrt(norm))

        d = div(px,py)

        ubar = -theta * u
        u = (u+tau*lamb*d+tau*im)/(1+tau)

        # v = u+tau*lamb*d
        #
        # unew = (v-tau)*[v-im > tau]+ (v+tau)*[v-im < -tau]+ im*[abs(v-im)<=tau]
        # unew=unew[0]



        ubar = ubar+(1+theta)*u
        
        gradx, grady = grad(u)
        E[k] = 0.5*np.sum((u-im)**2) + lamb*np.sum(np.sqrt(gradx**2 + grady**2))
        # print("iteration {}".format(k),": E = {}\n".format(E[k]))
    return u, E
