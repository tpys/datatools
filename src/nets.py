
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
        self.shape = copy.deepcopy(self.faceNet.blobs['data'].data.shape)
        self.transformer.set_transpose('data', (2, 0, 1)) #(112 ,96, 3) to (3, 112, 96)
        
    def extract(self, batches):
        #self.faceNet.blobs['data'].reshape(1, 3, 112, 96)
        bs = len(batches)  # batch size
        
        #in_shape[0] = bs # set new batch size
        self.faceNet.blobs['data'].reshape(bs, self.shape[1], self.shape[2], self.shape[3])
        self.faceNet.reshape()
        i = 0
        features = []
        for bt in batches:
            img = bt["img"]
            img = (img - 127.5) / 128.0
            self.faceNet.blobs['data'].data[i,:,:,:] = self.transformer.preprocess('data',img)

            i += 1
        out1 = self.faceNet.forward()
        features_ = copy.deepcopy(out1['fc5'])
        features = []
        for j in range(0, bs):
            tmp = {}
            tmp["path"] = batches[j]["path"]
            tmp["feature"] = features_[j,:]
            features.append(tmp)
        return features

