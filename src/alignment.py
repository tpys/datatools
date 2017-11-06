import cv2
import sys
import numpy as np
import os,errno
import random
import shutil
import numpy as np
import math
from MtcnnDetector import FaceDetector

import tools


class Methods:
    
    def __init__(self):
        pass
    
    @staticmethod
    def _112x96_mc40(input_image, points, output_size = (96, 112), ec_mc_y = 40):
        eye_center = ((points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2)
        mouth_center = ((points[3][0] + points[4][0]) / 2, (points[3][1] + points[4][1]) / 2)
        angle = math.atan2(mouth_center[0] - eye_center[0], mouth_center[1] - eye_center[1]) / math.pi * -180.0
        # angle = math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0]) / math.pi * 180.0
        scale = ec_mc_y / math.sqrt((mouth_center[0] - eye_center[0])**2 + (mouth_center[1] - eye_center[1])**2)
        center = ((points[0][0] + points[1][0] + points[3][0] + points[4][0]) / 4, (points[0][1] + points[1][1] + points[3][1] + points[4][1]) / 4)
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
        rot_mat[0][2] -= (center[0] - output_size[0] / 2)
        rot_mat[1][2] -= (center[1] - output_size[1] / 2)
        warp_dst = cv2.warpAffine(input_image, rot_mat, output_size)
        return warp_dst



class Aligner:
    def __init__(self, root_path, gpu_id = 0, save_path_suffix = '-aligned', align_method = Methods._112x96_mc40):
        
        self.detector = FaceDetector(minsize = 20, gpuid = gpu_id, fastresize = False)
        self.root_path = root_path
        self.save_path_suffix = save_path_suffix
        self.align_method = align_method


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
                        tools.mkdirP(root_path + self.save_path_suffix +"/"+dirname)
                        if(sub_filename.endswith("png") or sub_filename.endswith("jpg") or sub_filename.endswith("bmp")):
                            tmp = {}
                            tmp['filename'] = dirname + "/" + sub_filename
                            tmp['class_num'] = class_label
                            lines.append(tmp)
                class_label += 1
        return lines   

    def align(self):
        tools.mkdirP(self.root_path + self.save_path_suffix)
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


    def AlignWuXiang(self, input_image, points, output_size = (96, 112), ec_mc_y = 40):
        eye_center = ((points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2)
        mouth_center = ((points[3][0] + points[4][0]) / 2, (points[3][1] + points[4][1]) / 2)
        angle = math.atan2(mouth_center[0] - eye_center[0], mouth_center[1] - eye_center[1]) / math.pi * -180.0
        # angle = math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0]) / math.pi * 180.0
        scale = ec_mc_y / math.sqrt((mouth_center[0] - eye_center[0])**2 + (mouth_center[1] - eye_center[1])**2)
        center = ((points[0][0] + points[1][0] + points[3][0] + points[4][0]) / 4, (points[0][1] + points[1][1] + points[3][1] + points[4][1]) / 4)
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
        rot_mat[0][2] -= (center[0] - output_size[0] / 2)
        rot_mat[1][2] -= (center[1] - output_size[1] / 2)
        warp_dst = cv2.warpAffine(input_image, rot_mat, output_size)
        return warp_dst

    

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
            #similarTransformation = cv2.estimateRigidTransform(np.array(default_face[2]), dst_points, fullAffine=False)
            face_points = []
            for j in range(5):        
                tmp =  [points[j,default_face_idx], points[j + 5,default_face_idx]]
                face_points.append(tmp)
            alignedImg = []
            
            alignedImg = self._alignment( self.align_method, img, face_points)
           
                #print alignedImg
            #print path + self.save_path_suffix + "/" + filename
            cv2.imwrite(path + self.save_path_suffix + "/" + filename, alignedImg)

