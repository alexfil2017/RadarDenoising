import numpy as np
import matplotlib.pyplot as plt
from imz2mat import *




def displayRSO(im,k=3):
    """
    parameters
    ---
    im : module of the complex image
    k : threshold (in number of std)
    ---
    returns
    ---
    plot image encoded on 255 bits

    """
    m = np.mean(im)
    s = np.std(im)
    maxvalue = m+k*s
    mask = im >= maxvalue
    im[mask] = maxvalue
    # plt.figure()
    plt.imshow(im, cmap= 'gray')
    plt.show()


def addSARnoise(ima, L=1, intensite = True) :
    """
    parameters
    ---
    ima : the image
    L : number of view
    intensite : if True : work with intensity,
                else : work with amplitude
    ---
    returns
    ---
    the noised image


    """


    s = np.zeros(ima.shape)
    for k in range(L):
        s = s + np.abs(np.random.randn(*ima.shape) + 1j * np.random.randn(*ima.shape))**2 / 2

    #s image de valeurs de pdf Gamma de nb de vues L de moyenne 1
    if intensite:
        s=s/L
    else :
        s=np.sqrt(s/L)

    ima_speckle_amplitude = ima * s

    return ima_speckle_amplitude

def geometricMean(im):
    """
    compute the geometricmean of a multitemporal series
    """
    l = im.shape[2]
    res = im[:,:,0]
    for k in range(1,l):
        res = res*im[:,:,k]
    res = res**(1/l)
    return res
