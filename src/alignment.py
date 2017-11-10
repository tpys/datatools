import cv2
import sys
import numpy as np
import os,errno
import random
import shutil
import numpy as np
import math
from MtcnnDetector import FaceDetector
import align_methods
import utils






class Maker:
    def __init__(self, root_path, gpu_id = 0, save_path_suffix = '-aligned', align_method = align_methods._112x96_mc40, save_landmarks=True, save_aligned_img=False):
        self.gpu_id = gpu_id
        self.detector = []
        self.root_path = root_path
        self.save_path_suffix = save_path_suffix
        self.align_method = align_method
        self.save_landmarks = save_landmarks
        self.save_aligned_img = save_aligned_img

    def _alignment(self, func,img, face_points):
        return func(img, face_points)

    def getLines(self):
        lines = []
        root_path = self.root_path
        class_label = 0
        for parent, dirnames, filenames in os.walk(root_path):
            for dirname in dirnames:
               
                for sub_parent, sub_dirnames, sub_filenames in os.walk(root_path+"/"+dirname):
                    for sub_filename in sub_filenames:
                        if self.save_aligned_img:
                            utils.mkdirP(root_path + self.save_path_suffix +"/"+dirname)
                        if self.save_landmarks:
                            utils.mkdirP(root_path + "_landmarks/"+dirname)
                        if(sub_filename.endswith("png") or sub_filename.endswith("jpg") or sub_filename.endswith("bmp")):
                            tmp = {}
                            tmp['filename'] = dirname + "/" + sub_filename
                            tmp['class_num'] = class_label
                            lines.append(tmp)
                class_label += 1
        return lines   

    def make(self):
        self.detector = FaceDetector(minsize = 20, gpuid = self.gpu_id, fastresize = False)
        if self.save_aligned_img:
            utils.mkdirP(self.root_path + self.save_path_suffix)
        if self.save_landmarks:
            utils.mkdirP(self.root_path + "_landmarks")
        #f_list = open(list_path, "r")
        all_files = self.getLines()
        N = len(all_files)
        cnt = 0
        for line in all_files:
            filename = line['filename']
            class_num = line['class_num']
            if cnt >= 0:
                self.align_and_save_face(filename)
                print("{0}/{1}" .format(cnt + 1, N))
            cnt += 1


    def align_and_save_face(self, filename):
        detector = self.detector
        path = self.root_path
        img = cv2.imread(path + "/" + filename)
        #print img
        if img is None:
            print("Warning: input image in null: " + path + "/" + filename)
            return

        total_boxes,points,numbox = detector.detectface(img)

        default_face_idx = None
        if numbox >= 1:
            
            center = (img.shape[1] / 2, img.shape[0] / 2)
            distance_to_center = 99999.0
            for i in range(numbox):
                bb = [int(total_boxes[i][0]),int(total_boxes[i][1]),int(total_boxes[i][2]),int(total_boxes[i][3])]
                distance = abs(bb[0] + bb[2]/ 2 - center[0]) + abs(bb[1] + bb[3] / 2 - center[1])
                if (distance < distance_to_center):
                    distance_to_center = distance
                    default_bb = bb
                    default_face_idx = i
           
        elif numbox == 1:
            default_face_idx = 0
        if default_face_idx is not None:
            face_points = []
            for j in range(5):        
                tmp =  [points[j,default_face_idx], points[j + 5,default_face_idx]]
                face_points.append(tmp)
            alignedImg = []
            
            alignedImg = self._alignment( self.align_method, img, face_points)
           
                #print alignedImg
            #print path + self.save_path_suffix + "/" + filename
            if self.save_aligned_img:
                cv2.imwrite(path + self.save_path_suffix + "/" + filename, alignedImg)
            if self.save_landmarks:
                np.savetxt(path + "_landmarks/" + filename[:-4] + ".txt", np.array(face_points))

