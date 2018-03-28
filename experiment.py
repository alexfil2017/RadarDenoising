# -----------------------------------------------------------------------------
# This file is for the experiment. It saves all the images in the file data/test
# it is very similar to the admm_dl notebook
# Sofiane Horache
# -----------------------------------------------------------------------------

# libraries

import scipy
import numpy as np
import tensorflow as tf
from model import denoiser # CNN
from utils_train import * # function for the CNN(compute the MSE)
import scipy.misc
import matplotlib.pyplot as plt
import imz2mat # open the sar images
import utils # other usefull function(add noise)
import ADMM # perform ADDM(with chambolle&pock or DuCNN)
import argparse
# We then run a session tf(WARNING : execute once)
# Don't forget to close
parser = argparse.ArgumentParser(description='')
parser.add_argument('--L', dest='L', type=int, default=1, help='fix the level of noise')
parser.add_argument('--path_gt', dest='path_gt',
                    default='data/PileSaintGervais_moyennearithmetique.imf',
                    help='the ground truth path')
parser.add_argument('--out_dir', dest='out_dir', default='./data/test/exp1', help="path to store the images")

args = parser.parse_args()
# open a tensor flow session to run the CNN
sess = tf.Session()
file = open('{0}/res.txt'.format(args.out_dir), 'w+')

# read the images
print('opening the images:')
img_gt = imz2mat.imz2mat(args.path_gt)[0].astype(np.float32)
img_speckle = utils.addSARnoise(img_gt, L=args.L)
img_gt_log = np.log(1+img_gt)
img_in = np.log(1+img_speckle)
res = "MSE without denoising : {}".format(cal_mse(np.exp(img_gt_log)-1, np.exp(img_in)-1))
print(res)
file.write(res+'\n')
utils.displayRSO(np.sqrt(img_gt),is_display=False, path='{0}/ground_truth.png'.format(args.out_dir))
utils.displayRSO(np.sqrt(img_speckle),is_display=False, path='{0}/noisy_image_L{1}.png'.format(args.out_dir, args.L))
utils.displayRSO(np.sqrt(img_gt[200:400,500:700]),is_display=False,
                 path='{0}/noisy_image_detail_L{1}.png'.format(args.out_dir, args.L))


#---------------------------------------------------------------------------
# experiment 1 : We denoise this image using chambolle & pock using ADMM
#---------------------------------------------------------------------------
if(True):
    print("experiment 1:")
    # TODO: Fix the problem I don't know but I have to do this
    img_gt = imz2mat.imz2mat('data/PileSaintGervais_moyennearithmetique.imf')[0].astype(np.float32)
    img_speckle = utils.addSARnoise(img_gt, L=args.L)
    img_gt_log = np.log(1+img_gt)
    img_in = np.log(1+img_speckle)

    out, E = ADMM.ADMM(img_in, beta = 0.7,lamb = 1,niter = 10)
    print(cal_mse(np.exp(img_gt_log)-1, np.exp(out)-1))
    mse_pock = cal_mse(np.exp(img_gt_log)-1, np.exp(out)-1)
    exp_out = np.exp(out)-1
    utils.displayRSO(np.sqrt(exp_out),is_display=False, path='{0}/chambolle_L{1}.png'.format(args.out_dir, args.L))
    utils.displayRSO(np.sqrt(exp_out[200:400,500:700]),is_display=False, path='{0}/chambolle_detail_L{1}.png'.format(args.out_dir, args.L))
    res="MSE using ADMM+ CHambolle & Pock: {}".format(mse_pock)
    print(res)
    file.write(res+'\n')

#---------------------------------------------------------------------------
# experiment 2 : We denoise this image using a CNN Gaussian Denoiser(train with quantile) with all sigma
# Goal see the influence of the CNN
#---------------------------------------------------------------------------
if(True):
    print("eperiment 2:")
    model = denoiser(sess, sigma=25, add_noise=False)

    list_ckp = ["./checkpoint", "./checkpoint_sar","checkpoint_sar_norm", "./checkpoint_sar_1"]
    list_name=["DuCNN25Nat", "DuCNN25Sar_quant", "DuCNN25Sar", "DuCNN13SarQuant"]

    list_beta=[1,1,1,4]
    for i in range(len(list_beta)):
    
        ckpt_dir = list_ckp[i]
        model.load(ckpt_dir)

        C = np.max(img_in)-np.min(img_in)
        b = img_in.min()/C
        img_inp = img_in/C - b
        img_input = img_inp.reshape(1, img_in.shape[1], img_in.shape[0],1)
        out2, _, _ = model.denoise(img_input)
        out2 = out2.reshape(img_in.shape)
        out2 = C*(out2+b)
        out2 = out2 + utils.debiased(args.L)
        mse_cnn = cal_mse(np.exp(img_gt_log)-1, np.exp(out2)-1)
        exp_out2 = np.exp(out2)-1
        utils.displayRSO(np.sqrt(exp_out2),is_display=False, path='{0}/{1}_L{2}.png'.format(args.out_dir,
                                                                               list_name[i],
                                                                           args.L))
        utils.displayRSO(np.sqrt(exp_out2[200:400,500:700]),is_display=False, path='{0}/{1}_detail_L{2}.png'.format(args.out_dir,
                                                                                                       list_name[i],                                                                                         
                                                                                                                    args.L))
        res = "MSE for {0}: {1}".format(list_name[i], mse_cnn)
        print(res)
        file.write(res+'\n')
#---------------------------------------------------------------------------
# experiment 3 : We use ADMM + DuCNN
#Goal : see the influence of the DuCNN
# We test with the optimal beta found automatically
#we just change the CNN
#---------------------------------------------------------------------------

print('experiment 3')
if(True):
    for i in range(len(list_beta)):
        ckpt_dir = list_ckp[i]
        model.load(ckpt_dir)
        out3, E = ADMM.ADMM(img_in, beta = list_beta[i],lamb = 1,niter = 10, CNNprior=model)
        mse_cnn = cal_mse(np.exp(img_gt_log)-1, np.exp(out3)-1)
        exp_out3 = np.exp(out3)-1
        utils.displayRSO(np.sqrt(exp_out3),is_display=False, path='{0}/{1}_ADMM_L{2}.png'.format(args.out_dir,
                                                                                    list_name[i],
                                                                                    args.L))
        utils.displayRSO(np.sqrt(exp_out3[200:400,500:700]),is_display=False,
                         path='{0}/{1}_detail_ADMM_L{2}.png'.format(args.out_dir,
                                                                    list_name[i],
                                                                    args.L))
        res="MSE for {0} ADMM: {1}".format(list_name[i], mse_cnn)
        print(res)
        file.write(res+'\n')

#---------------------------------------------------------------------------
# experiment 4 : We use ADMM + DuCNN
#Goal : see the influence of beta
# 
#we just change the CNN
# We try with just 1 dataset
#---------------------------------------------------------------------------
if(True):
    list_beta = [0.2,0.5,1,2,7,20]
    ckpt_dir = "./checkpoint_sar"
    model.load(ckpt_dir)
    #model = None
    for beta in list_beta:
        out4, E = ADMM.ADMM(img_in, beta = beta,lamb = 1,niter = 10, CNNprior=model)
        mse_cnn = cal_mse(np.exp(img_gt_log)-1, np.exp(out4)-1)
        exp_out4 = np.exp(out4)-1
        utils.displayRSO(np.sqrt(exp_out4),is_display=False, path='{0}/{1}_ADMM_L{2}.png'.format(args.out_dir,
                                                                                    beta,
                                                                                    args.L))
        utils.displayRSO(np.sqrt(exp_out4[200:400,500:700]),is_display=False,
                         path='{0}/{1}_detail_ADMM_L{2}.png'.format(args.out_dir,
                                                                    beta,
                                                                    args.L))
        res="MSE for {0}: {1}".format(beta, mse_cnn)
        print(res)
        file.write(res+'\n')
        
        
    

sess.close()
file.close()



