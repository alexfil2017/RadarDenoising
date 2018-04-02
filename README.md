# RadarDenoising
## MVA project : SAR image denoising 

link for the end-to-end denoiser : https://github.com/lulica/provisorio
In this project, we can find the code and the experiments done to SAR denoising. The CNN is based on this link:
https://github.com/crisb-DUT/DnCNN-tensorflow

The file experiment.py contains some experiments done on the CNN prior and the ADMM.
the notebooks explain also how we use the different denoisers.

The checkpoints folders contains the different weights of the CNN.
- ./chechpoint contain the weights of the CNN trained on natural images(sigma=25 if the image is between 0 and 255)
- ./chechpoint_sar contain the weights of the CNN trained on sar images+quantiles removed(with sigma=25)
- ./chechpoint_sar_1 contain the weights of the CNN trained on sar images+quantiles removed(with sigma=13)
- ./chechpoint_sar_norm contain the weights of the CNN trained on sar images+(with sigma=25)
