import argparse
import glob
import numpy as np
import scipy
import scipy.misc
import random
from imz2mat import *
from utils import *
from utils_train import data_augmentation

DATA_AUG_TIMES = 1
parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir', dest='src_dir', default='./data', help='dir of data')
parser.add_argument('--save_dir', dest='save_dir', default='./data', help='dir of patches')
parser.add_argument('--patch_size', dest='pat_size', type=int, default=40, help='patch size')
parser.add_argument('--stride', dest='stride', type=int, default=10, help='stride')
parser.add_argument('--step', dest='step', type=int, default=0, help='step')
parser.add_argument('--batch_size', dest='bat_size', type=int, default=128, help='batch size')
# check output arguments
parser.add_argument('--from_file', dest='from_file', default="./data/img_clean_pats_sar.npy", help='get pic from file')
parser.add_argument('--num_pic', dest='num_pic', type=int, default=10, help='number of pic to pick')
parser.add_argument('--is_robust', dest='is_robust', type=bool, default=True, help='apply normalization')
parser.add_argument('--name_file', dest='name_file', default='img_clean_pats_sar')
args = parser.parse_args()

def generate_patches(filepath, isDebug=False, step=0, stride=10, pat_size=40, bat_size=128, is_robust=True):
    """
    This function generates patches
    """
    global DATA_AUG_TIMES
    filepaths = glob.glob(filepath + '/*.imw')
    count = 0
    if isDebug:
        filepaths = filepaths[:10]
    print("number of training data %d" % len(filepaths))
    
    scales = [1, 0.9, 0.8, 0.7]
    
    # calculate the number of patches
    for i in range(len(filepaths)):
        img =  imz2mat(filepaths[i])[0]# convert RGB to gray
        for s in range(len(scales)):
            newsize = (int(img.shape[0] * scales[s]), int(img.shape[1] * scales[s]))
            img_s = scipy.misc.imresize(img, size=newsize, interp='bicubic')  # do not change the original img
            im_h, im_w = img_s.shape
            for x in range(0 + step, (im_h - pat_size), stride):
                for y in range(0 + step, (im_w - pat_size), stride):
                    count += 1
    origin_patch_num = count * DATA_AUG_TIMES
    
    if origin_patch_num % bat_size != 0:
        numPatches = int((origin_patch_num / bat_size + 1) * bat_size)
        
    else:
        numPatches = origin_patch_num
    print("total patches = %d , batch size = %d, total batches = %d" % 
          (numPatches, bat_size, numPatches / bat_size))
    
    # data matrix 4-D
    inputs = np.zeros((numPatches, pat_size, pat_size, 1), dtype="uint8")
    
    count = 0
    # generate patches
    for i in range(len(filepaths)):
        img = imz2mat(filepaths[i])[0]
        if(is_robust):
            img = robust_scale(img)
        else:
            img = min_max_scale(np.log(1+img))
        for s in range(len(scales)):
            newsize = (int(img.shape[0] * scales[s]), int(img.shape[1] * scales[s]))
            # print newsize
            img_s = scipy.misc.imresize(img, size=newsize, interp='bicubic')
            img_s = np.reshape(np.array(img_s, dtype="uint8"),
                               (img_s.shape[0], img_s.shape[1], 1))  # extend one dimension
            
            for j in range(DATA_AUG_TIMES):
                im_h, im_w, _ = img_s.shape
                for x in range(0 + step, im_h - pat_size, stride):
                    for y in range(0 + step, im_w - pat_size, stride):
                        inputs[count, :, :, :] = data_augmentation(img_s[x:x + pat_size, y:y + pat_size, :],
                                                                   random.randint(0, 7))
                        count += 1
    # pad the batch
    if count < numPatches:
        to_pad = numPatches - count
        inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]
        
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    np.save(os.path.join(args.save_dir, args.name_file), inputs)
    print("size of inputs tensor = " + str(inputs.shape))
    
    return inputs

if __name__ == '__main__':
    generate_patches(filepath=args.src_dir, step=args.step, stride=args.stride,
                     pat_size=args.pat_size, bat_size=args.bat_size)
