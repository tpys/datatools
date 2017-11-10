from multiprocessing import Process,Queue,Pool
import multiprocessing
import lmdb
import numpy as np
import cv2
import os,errno
import sys
import struct
sys.path.insert(0, '/home/maxiao/CoreLib/caffe_patchData/build/install/python')
import caffe
import math
import random
img_path = 'MsCeleb20k-faces'
glasses_path = 'MsCeleb20k-faces_glasses'
ldmk_path = 'MsCeleb20k-faces_aligned_landmark'
img_80k_path = "MsCelebV1-Faces-Cropped_80k_clean_with_XuMethods"
ldmk_80k_path = "MsCelebV1-Faces-Cropped_80k_clean_with_XuMethods_aligned_landmark"
out_path = 'LMDB/MsCeleb100K-faces-aligned_112x96_lmdb/'

tmp_path = "tmp"

def mkdirP(path):
    """
    Create a directory and don't error if the path already exists.
    If the directory already exists, don't do anything.
    :param path: The directory to create.
    :type path: str
    """
    assert path is not None

    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def create_list(path, ldmk_path, glass_path, list_name, create_aligned_folder=False):
    f_list = open(list_name, "w")
    class_label = 0
    
    # 20k
    for parent, dirnames, filenames in os.walk(ldmk_path):
        for dirname in dirnames:
            is_small_num_faces = False
            for sub_parent, sub_dirnames, sub_filenames in os.walk(ldmk_path+"/"+dirname):
                if create_aligned_folder:
                    #print path + "-aligned"+"/"+dirname
                    mkdirP(img_path + "-aligned"+"/"+dirname)
                #if len(sub_filenames) < 25:
                #    is_small_num_faces = True
                #    break
                for sub_filename in sub_filenames:
                    

                    if(sub_filename.endswith("txt")):
                        
                        f_list.write( "{0},{1},{2},0\r\n".format(img_path + "/" + dirname + "/" + sub_filename[:-3] + "jpg", ldmk_path + "/" + dirname + "/" + sub_filename[:-3] + "txt", str(class_label)))
            if not is_small_num_faces:
                class_label += 1


        

    # 80k:
    for parent, dirnames, filenames in os.walk(ldmk_80k_path):
        for dirname in dirnames:
            is_small_num_faces = False
            for sub_parent, sub_dirnames, sub_filenames in os.walk(ldmk_80k_path+"/"+dirname):
                if create_aligned_folder:
                    #print path + "-aligned"+"/"+dirname
                    mkdirP(img_80k_path + "-aligned"+"/"+dirname)
                if len(sub_filenames) < 25:
                    is_small_num_faces = True
                    break
                for sub_filename in sub_filenames:
                    

                    if(sub_filename.endswith("txt")):
                        
                        f_list.write( "{0},{1},{2},0\r\n".format(img_80k_path + "/" + dirname + "/" + sub_filename[:-3] + "png", ldmk_80k_path + "/" + dirname + "/" + sub_filename[:-3] + "txt", str(class_label)))
            if not is_small_num_faces:
                class_label += 1
    print class_label

    f_list.close()



def getUrls(img_path, ldmk_path,glasses_path):
    list_name = "list_100k_test"
    #data_folder = 'test_images'
    #create_list(img_path, ldmk_path, glasses_path, list_name, True)

    f_list = open(list_name, "r")
    lines = f_list.readlines()
    key_ctr = 0
    writting_count = 0
    label_ctr = 0

    N = len(lines)
    
    
    urls = []
    random_order = np.random.permutation(N).tolist()

    for count in range(N):
        tmp = {}
        idx = random_order[count]
        line = lines[idx]
        filename, ldmk_name, class_num, is_glasses_str = line.split(",")
        tmp['path'] = filename
        tmp['label'] = int(class_num)
        tmp['ldmk_path'] = ldmk_name
        tmp['is_glasses'] = int(is_glasses_str)
        urls.append(tmp) 


 
    return urls

def AlignWuXiang(input_image, points, output_size = (96, 112), ec_mc_y = 40):
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


def getDatum(url):
    path = url['path'] 
    label = url['label']
    
    img=cv2.imread(path, 1)
    if(path.endswith("png")) and len(img.shape) == 3:
        #print path
        img = img[:,:,[2,1,0]]
    ldmk_txt_path = url['ldmk_path']
    is_glasses = url["is_glasses"]
    if os.path.exists(ldmk_txt_path):
        ldmk = np.loadtxt(ldmk_txt_path)
        #if len(img.shape) == 3:
        #    img = img[:,:,[2,1,0]]
        
        face_points = ldmk.tolist()

        alignedImg = AlignWuXiang(img, face_points)
        
        #img_path = 'MsCeleb20k-faces_test'
        #if is_glasses == 0:
        #    cv2.imwrite(img_path + '-aligned/' + tmp_splits[1] + "/" +tmp_splits[2], alignedImg)
        #elif is_glasses == 1:
        #    cv2.imwrite(glasses_path + '-aligned/' + tmp_splits[1] + "/" +tmp_splits[2], alignedImg)
        
        mkdirP(tmp_path + "/" + path.split("/")[-2])
        cv2.imwrite(tmp_path + "/" + path.split("/")[-2] + "/" + path.split("/")[-1], alignedImg)
        data = alignedImg.transpose(2,0,1)
        
        
        #landmarks = (ldmk)
        return data, label


def float2bytes(floats):
	if type(floats) is float:
		floats = [floats]
	return struct.pack('%sf' % len(floats), *floats)


def getDatumString(url):
    img, label = getDatum(url)

    datum = caffe.io.array_to_datum(img, label=label)
   
    return datum.SerializeToString()

def filterOut(url):
   

    if os.path.exists(url["ldmk_path"]):
        return True
    else:
        print "filtering out: " + url["ldmk_path"]
        return False 
 
def make_lmdb():
    #mkdirP(ldmk_path+"-aligned")
    urls = getUrls(img_path, ldmk_path,glasses_path)
    N = len(urls)
    print N
    data = urls
    chunk_size = 15000 # or whatever
    chunks = [data[i:i+chunk_size] for i in xrange(0,len(data),chunk_size)]
    
    pool = Pool()
    writting_count = 0
    
    lmdb_env = lmdb.open(out_path, map_size=int(1e12))
    lmdb_txn = lmdb_env.begin(write=True)
    for chunk in chunks:
         print 'Working...'
         new_chunk = filter(filterOut, chunk)

         results = pool.map(getDatumString, new_chunk)
         for datum in results:
             
             key = '%07d' % writting_count
             lmdb_txn.put(key, datum)
             if(writting_count % 1000 == 0):
            
                lmdb_txn.commit()
                lmdb_txn = lmdb_env.begin(write=True)
             writting_count = writting_count + 1
             print('{}/{}'.format(writting_count, N))
    lmdb_txn.commit()
    lmdb_env.close()
    print "Done."
      


def create_test_list(list_name):
    list_name = "list_100k"
    #data_folder = 'test_images'
    #create_list(img_path, ldmk_path, glasses_path, list_name, True)
    test_list_name = "list_100k_test"
    f_list = open(list_name, "r")
    f_list_test = open(test_list_name, "w")
    lines = f_list.readlines()

    N = len(lines)
    
    print N
    urls = []
    random_order = np.random.permutation(N).tolist()

    for count in range(1000):
        idx = random_order[count]
        line = lines[idx]
        f_list_test.write(line)

    f_list_test.close()
    f_list.close()

def main():
    #list_name = "list_100k"
    #data_folder = 'test_images'
    create_list(img_path, ldmk_path, glasses_path, list_name)
    #create_test_list(list_name)

    #make_lmdb()
    #list_name = "list_100k"
    #data_folder = 'test_images'
    #create_list(img_path, ldmk_path, glasses_path, list_name)
    #create_test_list(list_name)

if __name__ == '__main__':
    main()
    
