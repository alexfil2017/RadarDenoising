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

def min_max_scale(image, normal=True):
    """
    image between 0 and 1 or 0 and 255
    according to normal
    """
    img = image.copy()
    out = (img-img.min())/(img.max()-img.min())
    if(normal):
        out = out * 255
        return out.astype(np.uint8)
    else:
        return out
def quant_thresh(img):
    """
    threshold the image according to the quantile
    """
    pmax = np.percentile(img, 99)
    pmin = np.percentile(img, 1)
    out = img.copy()
    out[img>pmax] = pmax
    out[img<pmin] = pmin
    return out

def robust_scale(img):
    """
    perform log + quantile threshold+ min_max_thresh
    """
    return min_max_scale(quant_thresh(np.log(img)))
