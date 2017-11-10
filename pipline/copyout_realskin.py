import config
import loader,utils,shutil

root_path = "/media/lm004/Data/maxiao/dataset/MsCelebV1-Faces-Cropped_80k_clean_with_XuMethods_realskin_"
out_path = "/media/lm004/Data/maxiao/dataset/MsCelebV1-Faces-Cropped_80k_clean_with_XuMethods_realskin"
img_path = "/media/lm004/Data/maxiao/dataset/MsCelebV1-Faces-Cropped_80k_clean_with_XuMethods"

persons = loader.load_img_labels(root_path, "../tmp/MsCelebV1-Faces-Cropped_80k_clean_with_XuMethods_realskin")

utils.mkdirP(out_path)


for person in persons:
    if len(person["pathes"]) >= 2:
        foldername,filename= person["pathes"][0].split("/")
        #for path in person["pathes"]:
        shutil.copytree(img_path + "/" + foldername, out_path + "/" + foldername)

