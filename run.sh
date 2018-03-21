
export PATH=/cal/softs/anaconda/anaconda3/bin:$PATH
source activate shitenv
cd /ldaphome/shorache/project/RadarDenoising/
export CUDA_VISIBLE_DEVICES=1
python2 /ldaphome/shorache/project/RadarDenoising/main.py 
