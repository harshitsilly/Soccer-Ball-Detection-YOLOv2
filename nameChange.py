import os
import sys
import xml.etree.ElementTree as ET


c ="dataset/abcd1/img"
imageFiles = os.listdir(c)
d ="dataset/abcd1/ann"
annotationFiles = os.listdir(d)
i=0
for imageFile in imageFiles:
    try:
        os.rename(c + "/" + imageFile,c + "/" + str(i) + ".jpg")
    
    except Exception as e:
        print(e)
    
    i = i +1

i = 0
for annotationFile in annotationFiles:
    annotation = open(d+"/" + annotationFile,"r+")
    e = annotation.read()
    e =  e.split("filename")[0] + "filename>" + str(i) + ".jpg</filename" + e.split("filename")[2]
    e = e.split("path")[0] + "path>" +  c + "/" + str(i)+ ".jpg</path" + e.split("path")[2]
    annotation.write(e)
    try:
        os.rename(c + "/" + imageFile,c + "/" + str(i) + ".xml")
    
    except Exception as e:
        print(e)
    
    i = i +1

