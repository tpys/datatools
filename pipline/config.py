import os,sys

CAFFE_ROOT = '/media/lm004/Data/maxiao/CoreLib'  #change to your caffe root path



SRC_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../src"
# add to env path
sys.path.append(CAFFE_ROOT)
sys.path.append(SRC_ROOT)