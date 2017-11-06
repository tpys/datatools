from multiprocessing import Process,Queue,Pool
import multiprocessing
import loader,utils
import os
import numpy as np
import random
import nets
import alignment
import cv2

def _getDatum(url):
    #print url

    img=cv2.imread(url["img_path"])


    ldmk = np.loadtxt(url["ldmk_path"])
    face_points = ldmk.tolist()           
    alignedImg = alignment._112x96_mc40(img, face_points)

    
    return {"img": alignedImg, "path":url["img_path"]}

def _saveFeature(elem):
    feat = elem["feature"]
    
    #print path
    np.save(url["feat_path"], np.array(feat))   


class Maker:
    def __init__(self,root_path_, model_prefix):
        self.root_path = root_path_
        self.model_prefix = model_prefix
        self.ldmk_path = self.root_path + "_landmarks"
        self.feat_path = self.root_path + "_features"
        
        prototxt = model_prefix + ".prototxt"
        caffemodel = model_prefix + ".caffemodel"
        self.net = nets.ResFace(prototxt, caffemodel,6)



    def _getUrls(self, img_labels):
       
        urls = []
        #print img_labels
        N = len(img_labels)
        random_order = np.random.permutation(N).tolist()

        for eles in img_labels:
            label = eles["label"]
            imgs = eles["pathes"]
            for img in imgs:
                tmp = {}
                tmp['img_path'] = self.root_path + "/" + img
                tmp['ldmk_path'] = self.ldmk_path + "/" + img[:-4] + ".txt"
                tmp['feat_path'] = self.feat_path + "/" + img[:-4] + ".npy"
                tmp['label'] = label
                urls.append(tmp)
        random.shuffle(urls)
        #print urls
        return urls
    

    def _filteringOut(self,url):
        
        #print self.ldmk_path + "/" + filename[:-4] + ".txt"
        if os.path.exists(url['ldmk_path']) and not os.path.exists(url['feat_path']):
            return True
    
        else:
            return False


    
    
    def make(self):
        print "Loading data..."
        saved_data_name = "../tmp/" + self.root_path.split("/")[-1] + "_img_label"
        img_labels = []
        if not os.path.exists(saved_data_name + ".npy"):
            img_labels = loader.load_img_labels(self.root_path,saved_data_name)
        else:
            img_labels = np.load(saved_data_name + ".npy")
        
        data = self._getUrls(img_labels)
        print "Reading out {0} images.".format(len(data))

        utils.mkdirP(self.feat_path)

        chunk_size = 15000 # or whatever
        chunks = [data[i:i+chunk_size] for i in xrange(0,len(data),chunk_size)]
        N_chunks = len(chunks)
        pool = Pool()

        iidx = 0
        for chunk in chunks[iidx:]:
            print 'Working chunks: {0}/{1}...'.format(iidx, N_chunks)
            new_chunk = filter(self._filteringOut, chunk)
            print "Filtering out: {0}".format(len(chunk) - len(new_chunk))

            #_getDatum(new_chunk[0])

            results = pool.map(_getDatum, new_chunk)
            print "Reading out: {0}".format(len(results))
            bs = 256
            batches = [results[i:i+bs] for i in xrange(0,len(results),bs)]
            N_bs = len(batches)
            iidx_bs = 0
            for batch in batches:
                print '--->Working batches: {0}/{1}...'.format(iidx_bs, N_bs)
                feats = self.net.extract(batch)
                pool.map(_saveFeature, feats)
                iidx_bs += 1
            iidx += 1
    print "Done."