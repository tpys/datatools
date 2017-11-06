import numpy as np
import argparse
import lfw
import os
import sys
import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from scipy import misc
from sklearn.model_selection import KFold
import cv2,itertools
from scipy import spatial

import random,shutil

import tools
#reload(sys)
#sys.setdefaultencoding("utf-8")





def filtering_out_simple_pairs(lfw_dir, pairs_path, lfw_file_ext, net):
    feats = {}
    if not os.path.exists("./feats.npy"):
        #np.save("feats", np.array(feats))
        print "feature extraction..."
        for parent, dirnames, filenames in os.walk(lfw_dir):
            for dirname in dirnames:
                
                for sub_parent, sub_dirnames, sub_filenames in os.walk(lfw_dir+"/"+dirname):
                    #sub_filename = sub_filenames[0]
                    for sub_filename in sub_filenames:
                        if(sub_filename.endswith("png") or sub_filename.endswith("jpg") or sub_filename.endswith("bmp")):
                            idx = int(sub_filename[-8:-4])
                            img_path = lfw_dir + "/" + dirname + "/" + sub_filename
                            img = misc.imread(img_path)            
                            feature = net.extract(img)
                            #tmp = {}
                            feats[unicode(img_path,'gbk')] = feature
                            print img_path
                            #feats.append(tmp)

        np.save("feats", np.array(feats))
    else:
        feats = dict(np.load("feats.npy").tolist())
    lfw_pairs = pairs_path
    pairs = lfw.read_pairs(os.path.expanduser(lfw_pairs))
    #print len(pairs)
    similarities = []
    #print feats
    print "calculate similarities..."
    new_pairs = []
    for pair in pairs:
        
        if len(pair) == 3:
            
            path0 = (lfw_dir + "/" + pair[0] + "/" + pair[0] + '_' + '%04d' % int(pair[1])+'.'+lfw_file_ext)
            path1 = (lfw_dir + "/" + pair[0] + "/" + pair[0] + '_' + '%04d' % int(pair[2])+'.'+lfw_file_ext)
            
            issame = True
        elif len(pair) == 4:
            path0 = (lfw_dir + "/" + pair[0] + "/" + pair[0] + '_' + '%04d' % int(pair[1])+'.'+lfw_file_ext)
            path1 = (lfw_dir + "/" + pair[2] + "/" + pair[2] + '_' + '%04d' % int(pair[3])+'.'+lfw_file_ext)
            issame = False
     
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            #path_list += (path0,path1)
            #print path0 + ",  " + path1

            f1 = feats[unicode(path0, 'gbk')]
            f2 = feats[unicode(path1,'gbk')]
            sim = 1 - spatial.distance.cosine(f1, f2)
            similarities.append(sim)
            if issame:
                if sim < 0.75:
                    print "added: " + path0 + ",  " + path1
                    new_pairs.append(pair)
            else:
                if sim > 0.5:
                    print "added: " + path0 + ",  " + path1
                    new_pairs.append(pair)        

    pairs_file = open(pairs_path, "w")
    new_pairs_random  = random.sample(new_pairs, len(new_pairs))  
    for pair in new_pairs_random:
        if len(pair) == 3:
            pairs_file.write("{0} {1} {2}\r\n".format(pair[0], pair[1], pair[2]))
        if len(pair) == 4:
            pairs_file.write("{0} {1} {2} {3}\r\n".format(pair[0], pair[1], pair[2], pair[3]))
           


def get_pairs_images(lfw_dir, pairs_path, lfw_file_ext):
    lfw_pairs = pairs_path
    pairs = lfw.read_pairs(os.path.expanduser(lfw_pairs))
    pathes = {}
    for pair in pairs:
    
        if len(pair) == 3:
            
            path0 = (lfw_dir + "/" + pair[0] + "/" + pair[0] + '_' + '%04d' % int(pair[1])+'.'+lfw_file_ext)
            path1 = (lfw_dir + "/" + pair[0] + "/" + pair[0] + '_' + '%04d' % int(pair[2])+'.'+lfw_file_ext)
            
            issame = True
        elif len(pair) == 4:
            path0 = (lfw_dir + "/" + pair[0] + "/" + pair[0] + '_' + '%04d' % int(pair[1])+'.'+lfw_file_ext)
            path1 = (lfw_dir + "/" + pair[2] + "/" + pair[2] + '_' + '%04d' % int(pair[3])+'.'+lfw_file_ext)
            issame = False
        
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            #path_list += (path0,path1)
            #print path0 + ",  " + path1
            pathes[path0] = 1
            pathes[path1] = 1

    print(len(pathes))
    count = 0
    for path in pathes:
        dirs_ = path.split("/")
        tools.mkdirP("./tmp/" + dirs_[3])
        shutil.copyfile(path,"./tmp/" + dirs_[3] + "/" + dirs_[4])  
        #count = count + 1


def gen_pairs(lfw_dir, pairs_path, lfw_file_ext):
    pairs_file = open(pairs_path, "w")
    
    

    pairs = []
    count_p = 0
    # A1 TO A2
    for parent, dirnames, filenames in os.walk(lfw_dir):
        for dirname in dirnames:
            
            for sub_parent, sub_dirnames, sub_filenames in os.walk(lfw_dir+"/"+dirname):
                #sub_filename = sub_filenames[0]
                if len(sub_filenames) <2:
                    continue
                pairs_lite = []
                #if len(sub_filenames) > 8:
                #     sub_filenames = random.sample(sub_filenames, 8)
                for sub_filename in sub_filenames:
                    if(sub_filename.endswith("png") or sub_filename.endswith("jpg") or sub_filename.endswith("bmp")):
                        #print sub_filename[-8:-4]
                        idx = int(sub_filename[-8:-4])
                        #print idx
                        pairs_lite.append(idx)
                pairs_lite_ = list(itertools.combinations(pairs_lite, 2))
                #print pairs_lite_
                for pair in pairs_lite_:
                    count_p += 1
                    pairs_file.write("{0} {1} {2}\r\n".format(dirname, pair[0], pair[1]))   
   
    print count_p
    
    # A TO B
    pairs = []
    for parent, dirnames, filenames in os.walk(lfw_dir):
        for dirname in dirnames:
            
            for sub_parent, sub_dirnames, sub_filenames in os.walk(lfw_dir+"/"+dirname):
                #print lfw_dir+"/"+dirname
                for sub_filename in sub_filenames:
                    
                    #for sub_filename in sub_filenames:
                    if(sub_filename.endswith("png") or sub_filename.endswith("jpg") or sub_filename.endswith("bmp")):
                        idx = int(sub_filename[-8:-4])
                        #print idx
                        pairs.append(dirname + " " + str(idx))  
    pairs_ = list(itertools.combinations(pairs, 2))
    #print pairs_
    
    print ("writting:")
    #pairs_ = random.sample(pairs_, 1000000)

    for pair in pairs_:
         pairs_file.write("{0} {1}\r\n".format(pair[0], pair[1]))
    
    pairs_file.close()




def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(threshold, dist) #dist > threshold return true
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    #acc = float(tp) / dist.size
    return tpr, fpr, acc

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    # diff = np.subtract(embeddings1, embeddings2)
    # dist = np.sum(np.square(diff), 1)

    dist = np.zeros(embeddings1.shape[0])
    for i in range(embeddings1.shape[0]):
        dist[i] = np.dot(embeddings1[i], embeddings2[i])/(np.linalg.norm(embeddings1[i]) * np.linalg.norm(embeddings2[i]))

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
	tpr_train = np.zeros((nrof_thresholds))
	fpr_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            tpr_train[len(thresholds)-threshold_idx-1], fpr_train[threshold_idx], acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
	best_threshold_index2 = len(thresholds)-np.argmax(tpr_train)-1
	best_threshold_index3 = np.argmin(fpr_train)
	print(thresholds[best_threshold_index])
	print(thresholds[best_threshold_index2])
	print(thresholds[best_threshold_index3])
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])
        #_, _, accuracy[fold_idx] = calculate_accuracy(0.9, dist[test_set],
        #                                              actual_issame[test_set])

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy

def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(threshold, dist)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    dist = np.zeros(embeddings1.shape[0])
    for i in range(embeddings1.shape[0]):
        dist[i] = np.dot(embeddings1[i], embeddings2[i])/(np.linalg.norm(embeddings1[i]) * np.linalg.norm(embeddings2[i]))
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
	
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
            #threshold = 0.0
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def evaluate(embeddings, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(-1, 1, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds)
    thresholds = np.arange(0, 1, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far


def saveResults(dataset, net, name, obj):
    RESULTS_path = "../results"
    tools.mkdirP(RESULTS_path)
    tools.mkdirP(RESULTS_path + "/" + dataset)
    tools.mkdirP(RESULTS_path + "/" + dataset + "/" + net)

    save_path = RESULTS_path + "/" + dataset + "/" + net + "/" + name

    np.save(save_path, obj)

def val(net,lfw_dir, pairs_path, lfw_file_ext):
    
    dataset = lfw_dir.split("/") [-1]
    
    lfw_pairs = pairs_path
    pairs = lfw.read_pairs(os.path.expanduser(lfw_pairs))

    #pairs = random.sample(pairs, 100)


    #print len(pairs)
    paths, actual_issame = lfw.get_paths(os.path.expanduser(lfw_dir), pairs, lfw_file_ext)


    
    batch_size = 100
    nrof_images = len(paths)
    print(nrof_images)

    nrof_batches = int(math.ceil(1.0 * nrof_images / batch_size))
    print(nrof_batches)
    



    FEATS = {}

    for i in range(nrof_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]
        #images = load_data(paths_batch, 96, 112)
        #print()
        print('process %s batch image, %s.' % (i, len(paths_batch)))
        for j in range(len(paths_batch)):
            if not FEATS.has_key(paths_batch[j]):
        
                img = misc.imread(paths_batch[j])
                
                feature = net.extract(img)

                FEATS[paths_batch[j]] = feature

          #print (images)

    A_pathes = paths[0::2]
    B_pathes = paths[1::2]

    assert (len(A_pathes) == len(B_pathes))

    N_p = len(A_pathes)

    similarites = np.zeros(N_p)
    for i in range(0, N_p):
        x1 = FEATS[A_pathes[i]]
        x2 = FEATS[B_pathes[i]]
        similarites[i] = np.dot(x1, x2)/(np.linalg.norm(x1) * np.linalg.norm(x2))


    val, far = calculate_val_far(0.6, similarites, actual_issame)
    print "VAL = {0}, FAR = {1}".format(val, far)
    
    saveResults(dataset, net.name, "similarites", similarites)
    #np.save(net.name + "similarites",similarites)
    thresholds = np.arange(0, 1, 0.01)
    vals = []
    fars = []
    for threshold in thresholds:
        
        val, far = calculate_val_far(threshold, similarites, actual_issame)
        vals.append(val)
        fars.append(far)
    vals = np.array(vals)
    fars = np.array(fars)
    saveResults(dataset, net.name, "vals", vals)
    saveResults(dataset, net.name, "fars", fars)
    #np.save(net.name + "vals",vals)
    #np.save(net.name + "fars",fars)
    target_fars = np.array([0.01, 0.001, 0.0001, 0.00001])
    for target in target_fars:
        
        idx = np.argmin(np.abs(fars - target))

        print "{2}: VAL = {0} @ FAR = {1}".format(vals[idx], fars[idx], thresholds[idx])

 