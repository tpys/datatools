import config
import loader,utils
from multiprocessing import Process,Queue,Pool
import multiprocessing
import numpy as np
import cv2,shutil
import os
root_path = "/media/lm004/Data/maxiao/dataset/MsCelebV1-Faces-Cropped_80k_clean_with_XuMethods"
out_path = "/media/lm004/Data/maxiao/dataset/MsCelebV1-Faces-Cropped_80k_clean_with_XuMethods_realskin"
ldmk_path = "/media/lm004/Data/maxiao/dataset/MsCelebV1-Faces-Cropped_80k_clean_with_XuMethods_aligned_landmark"
def copyOut(url):
    min_YCrCb = np.array([0,133,77],np.uint8)
    max_YCrCb = np.array([255,173,127],np.uint8)
    folder_name, filename = url.split("/")
    # Grab video frame, decode it and return next video frame
    sourceImage = cv2.imread(root_path + "/" + url)
    if os.path.exists(ldmk_path + "/" + folder_name + "/" + filename[:-4] + ".txt"):
        return 
    #sourceImage = sourceImage[:,:,[2,1,0]]
    # Convert image to YCrCb
    imageYCrCb = cv2.cvtColor(sourceImage,cv2.COLOR_BGR2YCR_CB)

    # Find region with skin tone in YCrCb image
    skinRegion = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)


    cv2.imwrite("z.png", skinRegion * 100)

    percent = np.count_nonzero(skinRegion) / (sourceImage.shape[0] * sourceImage.shape[1] * 1.0)
    if percent > 0.35:
        print url
       
        utils.mkdirP(out_path + "/" + folder_name)
        shutil.copy(root_path + "/" + url, out_path + "/" + url)

#persons = loader.load_img_labels(root_path, "../tmp/MsCelebV1-Faces-Cropped-aligned_182")
persons = np.load("../tmp/MsCelebV1-Faces-Cropped-aligned_182.npy")
urls = []

for p in persons:
    for z in p["pathes"]:
        urls.append(z)

utils.mkdirP(out_path)

pool = Pool()
pool.map(copyOut, urls)