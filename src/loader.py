import os
import numpy as np
def load_img_labels(path, save_path):
    #f_list = open("{0}/list.txt".format(path), "w")
    datasets = []
    class_label = 0
    for parent, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            person = {}
            person["label"] = class_label
            imgs = []
            for sub_parent, sub_dirnames, sub_filenames in os.walk(path+"/"+dirname):
                for sub_filename in sub_filenames:
                    if(sub_filename.endswith("png") or sub_filename.endswith("jpg") or sub_filename.endswith("bmp")):
                        #f_list.write("{0},,{1}\r\n".format(dirname+"/"+sub_filename, str(class_label)))
                        imgs.append(dirname+"/"+sub_filename)
            person["pathes"] = imgs
            datasets.append(person)
            class_label += 1
    
    np.save(save_path, np.array(datasets))
    return np.array(datasets)