import config
import loader,utils,shutil
import numpy as np

identity_meta_file = open("../tmp/identity_meta.csv","r")

vggface2_lines = identity_meta_file.readlines()

vggface2_names = []
#vggface2_names_ =[]
print "Loading VGGFace2 names..."
for line in vggface2_lines[1:]:
    zz = line.split(",")
    name = zz[1][2:-1]
   
    name = name.replace("_", " ")
    #print name
    tmp = {}
    tmp["folder"] = zz[0]
    tmp["name"] = name
    
    vggface2_names.append(tmp)
    #vggface2_names_.append(name)

print "loaded {0} names.".format(len(vggface2_names))

MsCelebV1_face_names_file = open("../tmp/MsCelebV1_face_names.txt", "r")

MsCelebV1_lines = MsCelebV1_face_names_file.readlines()

MsCelebV1_names = []
#MsCelebV1_names_ = []
print "Loading MsCelebV1 names..."
for line in MsCelebV1_lines:
    zz = line.split(" ")
    #name_ = zz[1]
    t = " ".join(zz[1:])
    name_ = t.split("@")[0]
    name = name_[1:-1]
    tmp  = {}
    tmp["folder"] = zz[0]
    tmp["name"] = name
    #name.replace("_", " ")
    #print name
    #MsCelebV1_names_.append(tmp["name"])
    MsCelebV1_names.append(tmp)
print "loaded {0} names.".format(len(MsCelebV1_names))


#z = filter(lambda x:x["name"].split(",,,___,,,") in MsCelebV1_names_[:]["name"], MsCelebV1_names)

z = {}
for MsCelebV1_name in MsCelebV1_names:
    for vggface2_name in vggface2_names:
        if vggface2_name["name"] == MsCelebV1_name["name"]:
            
            name = vggface2_name["name"]
            print name
            vgg = vggface2_name["folder"]
            ms = MsCelebV1_name["folder"]
            if not z.has_key(name):
                tmp = {}
                tmp["vgg"] = vgg
                tmp["ms"] = ms
                z[name] = tmp

np.save("../tmp/same_names_in_VGGface2_and_ms100", np.array(z.items()))
print len(z)