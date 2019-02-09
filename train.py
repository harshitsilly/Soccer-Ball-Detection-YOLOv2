import sys,os
import numpy as np
from PIL import Image
import io
from darkflow.net.build import TFNet
import cv2

if len(sys.argv)>=3:
    if not os.listdir("ckpt"):
        os.mkdir("ckpt")
    options= sys.argv[2]
    tfnet = TFNet(options)
    tfnet.train()
else:
    print("provide valid arguments")