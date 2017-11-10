import config

import align_methods
import alignment

aliger = alignment.Maker("../data/lfw",0,"-aligned-112x96",align_methods._112x96_mc40)
aliger.make()


import mk_features

feat_makers = mk_features.Maker("../data/lfw", "../tmp/MS100KALL_ResFace_28_BN_112x96_2048_without_poooling",7)
feat_makers.make()