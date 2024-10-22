import numpy as np
import matplotlib.pyplot as plt
from imz2mat import *
import scipy



def displayRSO(im,k=3, is_display=True, path='test.png'):
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
    if(is_display):
        plt.imshow(im, cmap= 'gray')
    else:
        plt.imsave(path,im, format='png',cmap='gray',dpi=1200)


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


def aritmeticMean(im):
    """
    compute the aritmetic mean of a multitemporal series. Takes the amplitude of
    the multitemporal series as a parameter, computes its intensity, computes
    the mean on the intensity
    """
    l = im.shape[2]
    res = im[:,:,0]**2
    for k in range(1,l):
        res = res+im[:,:,k]**2
    res = np.sqrt(res/l)
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

def robust_scale(img, normal=True):
    """
    perform log + quantile threshold+ min_max_thresh
    """
    return min_max_scale(quant_thresh(np.log(img)), normal=normal)

def debiased(L):
    """
    a debias when we apply the homomorphic transform
    """
    return np.log(L)-scipy.special.digamma(L)