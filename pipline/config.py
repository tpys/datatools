import os,sys

CAFFE_ROOT = '/home/maxiao/CoreLib/caffe_windows_ms/python'  #change to your caffe root path



SRC_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../src"
# add to env path
sys.path.append(CAFFE_ROOT)
sys.path.append(SRC_ROOT)