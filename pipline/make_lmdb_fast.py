from multiprocessing import Process,Queue,Pool
import multiprocessing
import lmdb
import numpy as np
import cv2
import os,errno
import sys
import struct

sys.path.insert(0, '/home/zhouyn/caffe/python')
import caffe
import math
img_path = 'weibo_face_clustered_aligned_112x96'
#ldmk_path = 'MsCelebV1-Faces-Cropped-aligned_182_aligned_landmark'
out_path = './LMDB/weibo_face_clustered_aligned_112x96_lmdb/'
outliers_path = ""

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


def create_list(path, list_name, create_aligned_folder=False):
    f_list = open(list_name, "w")
    class_label = 64521
    zz = 0
    for dirname in os.listdir(path):
        files = os.listdir(path+"/"+dirname)
        #if len(files) < 30:
        #    continue
        #count = 0
        lines = []
        for sub_filename in files:
            if(sub_filename.endswith("png") or sub_filename.endswith("jpg")):
                #if not os.path.exists(outliers_path + "/" + dirname + "/" + sub_filename):
                    #f_list.write( "{0} {1}\r\n".format(dirname + "/" + sub_filename[:-4], str(class_label)))
                    lines.append("{0}_,,,_{1}\r\n".format(dirname + "/" + sub_filename, str(class_label)))
                    
                #else:
                #    zz += 1
                #    print "{0}, {1}".format(outliers_path + "/" + dirname + "/" + sub_filename, zz)
        if len(lines) > 0:
            for line in lines:
                #print line
                f_list.write(line)
            class_label += 1
    f_list.close()



def getUrls(img_path, ldmk_path):
    #list_path = "/data1/gaoxiang/FaceData/list"
    list_path = "list"
    create_list(img_path,list_path)
    
    f_list = open(list_path,"rU")
    urls = []

    lines = f_list.readlines()
    N = len(lines)
    random_order = np.random.permutation(N).tolist()
    
    for count in range(N):
        idx = random_order[count]
        line = lines[idx]
        line = line[:-1]
        filename,class_num = line.split('_,,,_')
        tmp = {}
        tmp['path'] = filename
        tmp['label'] = int(class_num)
        urls.append(tmp)
       
    return urls


def getDatum(url):
    path = url['path'] 
    label = url['label']
    img=cv2.imread(img_path + '/' + path)

    #cv2.imwrite(ldmk_path + '-aligned/' + path + '.png', alignedImg) 
    data = img.transpose(2,0,1)

    return data, label


def getDatumString(url):
    img, label = getDatum(url)

    datum = caffe.io.array_to_datum(img, label=label)
   
    return datum.SerializeToString()
    
 
def make_lmdb():
    #mkdirP(ldmk_path+"-aligned")
    urls = getUrls(img_path, "")
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
         results = pool.map(getDatumString, chunk)
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
      
    
def main():
    make_lmdb()

if __name__ == '__main__':
    main()
    
