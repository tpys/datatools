import os,errno

import shutil
import re

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



def gen_topplus_dataset(path):
    #f_list = open(list_name, "w")
    #zhPattern = re.compile(ur'[\u4e00-\u9fa5]+')
    #test = PinYin()
    #test.load_word()
    class_label = 0
    count = 0
    for parent, dirnames, filenames in os.walk(path):
        for filename in filenames:
            is_small_num_faces = False
            print filename
            #match = zhPattern.search(filename.decode('utf8'))
            #print match
            #pinyin_words = test.hanzi2pinyin_split(string=match.group(0), split="_")
            new_filename = filename.replace(" ", "_")
            folder_path = path +"/"+new_filename[:-4]# + "_" + pinyin_words

            mkdirP(folder_path)
            shutil.copyfile(path + "/" + filename,folder_path + "/"+ new_filename[:-4] + "_0001" + filename[-4:])
            count = count + 1
def change_filenames_for_weibofaces(path):
    for parent, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            for sub_parent, sub_dirnames, sub_filenames in os.walk(path+"/"+dirname):
                count = 1
                for sub_filename in sub_filenames:
                    if(sub_filename.endswith("png") or sub_filename.endswith("jpg") or sub_filename.endswith("bmp")):
                        new_filename = dirname + "_%04d" % (count) + sub_filename[-4:]
                        count = count + 1
                        print path + "/" + dirname+"/"+new_filename
                        os.rename(path + "/" + dirname+"/"+sub_filename, path + "/" + dirname+"/"+ new_filename)
                        # f_list.write("{0} {1}\r\n".format(dirname+"/"+sub_filename, str(class_label)))


def create_list(path, list_path, create_aligned_folder = False, folder_suffix = "-aligned"):
    f_list = open("{0}/list.txt".format(list_path), "w")
    class_label = 0
    for parent, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            if create_aligned_folder:
                mkdirP(path + folder_suffix +"/"+dirname)
            for sub_parent, sub_dirnames, sub_filenames in os.walk(path+"/"+dirname):
                for sub_filename in sub_filenames:
                    if(sub_filename.endswith("png") or sub_filename.endswith("jpg") or sub_filename.endswith("bmp")):
                        f_list.write("{0} {1}\r\n".format(dirname+"/"+sub_filename, str(class_label)))
            class_label += 1
    f_list.close()





if __name__ == '__main__':
    # gen_topplus_dataset("D:/dataset/Face/topplus/people")
    change_filenames_for_weibofaces("../data/weibo_face-aligned")

    



