import os,sys

CAFFE_ROOT = 'F:/CoreLib/caffe-windows/Build/x64/Release/pycaffe'  #change to your caffe root path



SRC_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../src"
# add to env path
sys.path.append(CAFFE_ROOT)
sys.path.append(SRC_ROOT)