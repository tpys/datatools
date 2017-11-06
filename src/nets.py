caffe_root = 'F:/CoreLib/caffe-windows/Build/x64/Release/pycaffe'  #change to your caffe root path
import sys
sys.path.insert(0, caffe_root)
import caffe
import numpy as np
import cv2,copy

class ResFace():
    def __init__(self, prototxt, caffemodel, gpu_id = 0, DIM=512):

        self.name = prototxt.split(".")[-2]

        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        self.faceNet = caffe.Net(prototxt, caffemodel, caffe.TEST)
        tmp = self.faceNet.forward()
        self.dim = tmp['fc5'][0].shape[0] # * 2
        print self.dim


        self.transformer = caffe.io.Transformer({'data': self.faceNet.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1)) #(112 ,96, 3) to (3, 112, 96)
        self.transformer.set_channel_swap('data', (2, 1, 0)) #RGB to BGR
    def extract(self, img):
        #self.faceNet.blobs['data'].reshape(1, 3, 112, 96)
        img_output = img
        #img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        # equalize the histogram of the Y channel
        #img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

        # convert the YUV image back to RGB format
        #img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


        img_output = (img_output - 127.5) / 128.0
        self.faceNet.blobs['data'].data[...] = self.transformer.preprocess('data',img_output)
        out1 = self.faceNet.forward()
        feature1 = copy.deepcopy(out1['fc5'][0])

        #flipimg = cv2.flip(img, 1)
        #self.faceNet.blobs['data'].data[...] = self.transformer.preprocess('data',flipimg)
        #out2 = self.faceNet.forward()
        #feature2 = copy.deepcopy(out2['fc5'][0])
        #feature = np.concatenate((feature1, feature2))
        return feature1



class PatchResFace():
    def __init__(self, prototxt, caffemodel , patch_idx=0,gpu_id = 0, DIM=512):
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        self.faceNet = caffe.Net(prototxt, caffemodel, caffe.TEST)
        tmp = self.faceNet.forward()
        self.dim = tmp['fc5'][0].shape[0] * 2
        self.patch_idx = patch_idx
        print self.dim


        self.transformer = caffe.io.Transformer({'data': self.faceNet.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1)) #(112 ,96, 3) to (3, 112, 96)
        self.transformer.set_channel_swap('data', (2, 1, 0)) #RGB to BGR

    def gen_input_examples(self, img):
        patch_img = [[],[],[],[],[]]
        '''for patch 0'''
        patch_img[0] = img[((120-70)/2):(((120-70)/2)+70), ((120-70)/2):(((120-70)/2)+70), :]
        '''for patch 1'''
        patch_img[1] = img[0:70, 0:70, :]
        '''for patch 2'''
        patch_img[2] = img[0:70, (120-70):120, :]
        '''for patch 3'''
        patch_img[3] = img[(120-70):120, (120-70):120, :]
        '''for patch 4'''
        patch_img[4] = img[(120-70):120, 0:70, :]

        cv2.imwrite("example.jpg", patch_img[self.patch_idx])
    def extract(self, img):
        #self.faceNet.blobs['data'].reshape(1, 3, 112, 96)
        patch_img = [[],[],[],[],[]]
        '''for patch 0'''
        patch_img[0] = img[((120-70)/2):(((120-70)/2)+70), ((120-70)/2):(((120-70)/2)+70), :]
        '''for patch 1'''
        patch_img[1] = img[0:70, 0:70, :]
        '''for patch 2'''
        patch_img[2] = img[0:70, (120-70):120, :]
        '''for patch 3'''
        patch_img[3] = img[(120-70):120, (120-70):120, :]
        '''for patch 4'''
        patch_img[4] = img[(120-70):120, 0:70, :]
      


        img = (patch_img[self.patch_idx] - 127.5) / 128.0
        #print img.shape
        self.faceNet.blobs['data'].data[...] = self.transformer.preprocess('data',img)
        out1 = self.faceNet.forward()
        feature1 = out1['fc5'][0]

        flipimg = cv2.flip(img, 1)
        self.faceNet.blobs['data'].data[...] = self.transformer.preprocess('data',flipimg)
        out2 = self.faceNet.forward()
        feature2 = out2['fc5'][0]
        feature = np.concatenate((feature1, feature2))
        return feature