import config


from alignment import *

aliger = Aligner("../data/lfw",0,"-aligned-112x96",Methods._112x96_mc40)
aliger.align()
