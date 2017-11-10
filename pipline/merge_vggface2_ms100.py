import config
import numpy as np
import os
from multiprocessing import Process,Queue,Pool

vgg_feat = "/media/lm004/Data/maxiao/dataset/VGGFace2_features"
ms_20k_feat = "/media/lm004/Data/maxiao/dataset/MsCeleb20k-faces_features"
ms_80k_feat = "/media/lm004/Data/maxiao/dataset/MsCelebV1-Faces-Cropped_80k_clean_with_XuMethods_features"

names = np.load("../tmp/same_names_in_VGGface2_and_ms100.npy")
#names = dict(names)

identity_meta_file = open("../tmp/identity_meta.csv","r")

vggface2_lines = identity_meta_file.readlines()

print names[1]





def work(ele):
    name = ele[0]
    vgg = ele[1]["vgg"]
    ms = ele[1]["ms"]
    needed_remove = []
    if os.path.exists(vgg_feat + "/" + vgg):
        pathes_A = os.listdir(vgg_feat + "/" + vgg)
        root = []
        pathes_B = []
        if os.path.exists(ms_20k_feat + "/" + ms):
            root = ms_20k_feat 
            pathes_B = os.listdir(ms_20k_feat + "/" + ms)
        elif os.path.exists(ms_80k_feat + "/" + ms):
            root = ms_80k_feat 
            pathes_B = os.listdir(ms_80k_feat + "/" + ms)
        if len(pathes_A)>0 and len(pathes_B)>0:
            
            for path_a in pathes_A:
                x1 = np.load(vgg_feat + "/" + vgg + "/"+ path_a)
                for path_b in pathes_B:
                    x2 = np.load(root + "/" + ms + "/" + path_b)
                    sim = np.dot(x1, x2)/(np.linalg.norm(x1) * np.linalg.norm(x2))
                    if sim > 0.98:
                        print vgg + "/"+ path_a[:-4]
                        needed_remove.append(vgg + "/"+ path_a[:-4])
            
            return needed_remove
        return None
    return None


pool = Pool()

outs = pool.map(work, names)
np.save("../tmp/needed_remove_vggface2", outs)