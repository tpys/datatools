import config

import numpy as np

from multiprocessing import Process,Queue,Pool

import multiprocessing

import cv2
import os,errno
import sys
import struct,shutil

from functools import partial
import loader,utils,random,pickle

ms20k_path = "/media/lm004/Data/maxiao/dataset/MsCeleb20k-faces"
ms80k_path = "/media/lm004/Data/maxiao/dataset/MsCelebV1-Faces-Cropped_80k_clean_with_XuMethods"



Z = []

def face_distance(x1es, x2):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    global Z
    if len(x1es) == 0:
        return np.empty((0))

    #return 1/np.linalg.norm(face_encodings - face_to_compare, axis=1)
    sims = []
    for x1_ in x1es:
        x1 = x1_[1]
        #print Z[x1][x2]
        sims.append(Z[x1][x2])
    
    return sims


def _chinese_whispers(encoding_list, iterations=20):
    """ Chinese Whispers Algorithm
    Modified from Alex Loveless' implementation,
    http://alexloveless.co.uk/data/chinese-whispers-graph-clustering-in-python/
    Inputs:
        encoding_list: a list of facial encodings from face_recognition
        threshold: facial match threshold,default 0.6
        iterations: since chinese whispers is an iterative algorithm, number of times to iterate
    Outputs:
        sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
            sorted by largest cluster to smallest
    """
    threshold=0.8
    #from face_recognition.api import _face_distance
    from random import shuffle
    import networkx as nx
    # Create graph
    nodes = []
    edges = []
    #print encoding_list
    #image_paths, encodings = zip(*encoding_list)
    encodings = encoding_list
    if len(encodings) <= 1:
        print ("No enough encodings to cluster!")
        return []
    idx = 0
    for Z in encodings:
        # Adding node of facial encoding
        image_path = Z[0]
        face_encoding_to_check = Z[1]
        node_id = idx+1

        # Initialize 'cluster' to unique value (cluster of itself)
        node = (node_id, {'cluster': image_path, 'path': image_path})
        nodes.append(node)

        # Facial encodings to compare
        if (idx+1) >= len(encodings):
            # Node is last element, don't create edge
            break
        #print encodings
        compare_encodings = encodings[idx+1:]
        distances = face_distance(compare_encodings, face_encoding_to_check)
        encoding_edges = []
        for i, distance in enumerate(distances):
            if distance == True:
                #print "zz"
                # Add edge if facial match
                edge_id = idx+i+2
                encoding_edges.append((node_id, edge_id, {'weight': 1}))

        edges = edges + encoding_edges

        idx += 1

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Iterate
    for _ in range(0, iterations):
        cluster_nodes = G.nodes()
        shuffle(cluster_nodes)
        for node in cluster_nodes:
            neighbors = G[node]
            clusters = {}

            for ne in neighbors:
                if isinstance(ne, int):
                    if G.node[ne]['cluster'] in clusters:
                      
                        clusters[G.node[ne]['cluster']] += G[node][ne]['weight']
                    else:
                        clusters[G.node[ne]['cluster']] = G[node][ne]['weight']

            
            edge_weight_sum = 0
            max_cluster = 0
            
            for cluster in clusters:
                if clusters[cluster] > edge_weight_sum:
                    edge_weight_sum = clusters[cluster]
                    max_cluster = cluster

            # set the class of target node to the winning local class
            G.node[node]['cluster'] = max_cluster

    clusters = {}

    # Prepare cluster output
    for (_, data) in G.node.items():
        cluster = data['cluster']
        path = data['path']

        if cluster:
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(path)

    # Sort cluster output
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

    return sorted_clusters

def find_same_persons(facial_encodings):
    """ Cluster facial encodings
        Intended to be an optional switch for different clustering algorithms, as of right now
        only chinese whispers is available.
        Input:
            facial_encodings: (image_path, facial_encoding) dictionary of facial encodings
        Output:
            sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
                sorted by largest cluster to smallest
    """

    if len(facial_encodings) <= 1:
        print ("Number of facial encodings must be greater than one, can't cluster")
        return []

    # Only use the chinese whispers algorithm for now
    sorted_clusters = _chinese_whispers(facial_encodings)
    return sorted_clusters





def computer_center(path_feat, path_outerliers, path_class_center, folder):
    foldername = folder["foldername"]
    pathes = folder["pathes"]
    
    center = []
    if len(pathes) > 0:
        if len(pathes) > 10:
            pathes = random.sample(pathes, 10)
        center = np.load(path_feat + "/" + pathes[0])
        center =  center / np.linalg.norm(center)
        for path in pathes[1:]:
        
            feat = np.load(path_feat + "/" + path)
            feat =  feat / np.linalg.norm(feat)
        
            center += feat
        
        center = center / np.linalg.norm(center)
        np.save(path_class_center + "/" + foldername, center)
        print foldername + " saved."
    else:
        print "empty"

def is_outlier(path_outerliers, path):
    if os.path.exists(path_outerliers + "/" + path[:-4] + ".png"):
        print "outlier: " + path_outerliers + "/" + path[:-4] + ".png"
        return False
    else:
        return True

def gen_class_centers(root_path):
    path_feat = root_path + "_features"
 
    path_outerliers = root_path + "_outliers"
    path_class_center = root_path + "_cls_centers"

    
    print "Loading features..."
    files = loader.load_prefix(path_feat, "../tmp/" + path_feat.split("/")[-1], "npy", partial(is_outlier, path_outerliers))


    utils.mkdirP(path_class_center)

    func = partial(computer_center, path_feat,path_outerliers,path_class_center)
    #for feat in feature_ms20k_files'
    pool = Pool()

    pool.map(func, files)

def merge(root_path):
    path_class_center = root_path + "_cls_centers"
    save_path = root_path + "_merged_foldernames"
    files = os.listdir(path_class_center)
    print "merge>>>Loading features..."
    X = np.zeros([len(files), 2048])
    folder_names = []
    for i, f in enumerate(files):
        x = np.load(path_class_center + "/" + f)
        X[i][:] = x
        folder_names.append(f[:-4])
    if not os.path.exists("../tmp/" + save_path.split("/")[-1]+".npy"):
        global Z

        print "merge>>>calculate similarity matrix..."
        S = np.dot(X, X.transpose())
        threshold = 0.9
        Z = S > threshold
    
        flag = np.argwhere(S > threshold)
        outs = {}
        indexes = []
        for ff in flag:
            if ff[0] != ff[1]:
                name = folder_names[ff[0]]
                indexes.append([name,ff[0]])
        print "merge>>>figure out same persons..."      
        outs = find_same_persons(indexes)
        print len(outs)

        np.save("../tmp/" + save_path.split("/")[-1], outs)
    else:
        
        outs = np.load("../tmp/" + save_path.split("/")[-1] + ".npy")
        for i,out in enumerate(outs):
            for z in out:
                shutil.copytree(root_path + "/" + z, save_path + "/"+ str(i) + "/" + z)
                


#gen_class_centers(ms20_path)
merge(ms20k_path)
#gen_class_centers(ms80k_path)
