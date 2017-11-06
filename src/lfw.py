"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import codecs 

def get_paths(lfw_dir, pairs, file_ext):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    name1 = ""
    name2 = ""
    for pair in pairs:
        #print(pair)
        if len(pair) == 3:
            
            path0 =(lfw_dir + "/" +  pair[0] + "/" +  pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = (lfw_dir + "/" +  pair[0] + "/" +  pair[0] + '_' + '%04d' % int(pair[2])+'.'+file_ext)
            #print(path0)
            #print(path1)
            name1 = pair[0]
            name2 = pair[0]
        elif len(pair) == 4:
            path0 = (lfw_dir + "/" +  pair[0] + "/" + pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = (lfw_dir + "/" + pair[2] + "/" +  pair[2] + '_' + '%04d' % int(pair[3])+'.'+file_ext)
            name1 = pair[0]
            name2 = pair[2]
            
        #print(path0)
        #print(path1)
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame = (name1 == name2)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    #print path_list
    return path_list, issame_list

def read_pairs(pairs_filename):
    pairs = []
    with codecs.open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)



