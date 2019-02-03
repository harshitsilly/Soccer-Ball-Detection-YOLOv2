import xml.etree.ElementTree as ET

import os
import json
import sys
# create the file structure


afiles=[]
if len(sys.argv)>=3:
    annotationFilePath = sys.argv[1]
    imageFilePath = sys.argv[2]
    darkFlowannotation = sys.argv[3]
    try:
        os.listdir(darkFlowannotation)
    except Exception as e:
        os.mkdir(darkFlowannotation)
        print(darkFlowannotation + " directory created succesfully")
else:
    print("Provide arguments in correct order of annotationDirectory, imageDirectory,darFlowAnnotation directory")
annotationFiles = os.listdir(annotationFilePath)
for annotationFile in annotationFiles:
    # data={"tags": ["train", "train"], "description": imageFile.split('.')[0], "objects": [], "size": {"height": 34, "width": 152}}
    bRemove = False
    with open(annotationFilePath + annotationFile, 'r') as fp:
        # json.dump(data, fp)
        data1 = json.load(fp)
        if(data1["objects"]):
            with open(darkFlowannotation + annotationFile.split(".")[0] + ".xml",'wb') as fp:
                print(data1)
        
                data = ET.Element('annotation')  
                folder = ET.SubElement(data, 'folder') 
                filename = ET.SubElement(data, 'filename') 
                path = ET.SubElement(data, 'path')  
                source = ET.SubElement(data, 'source') 
                database = ET.SubElement(source, 'database') 
                size = ET.SubElement(data, 'size') 
                width = ET.SubElement(size, 'width') 
                height = ET.SubElement(size, 'height') 
                depth = ET.SubElement(size, 'depth') 
                segmented = ET.SubElement(data, 'segmented') 
                width.text = str(data1["size"]["width"])
                height.text = str(data1["size"]["height"])
                depth.text = "3"
                segmented.text = "0"
                for element  in data1["objects"]:
                    object = ET.SubElement(data, 'object') 
                    name = ET.SubElement(object, 'name') 
                    pose = ET.SubElement(object, 'pose') 
                    truncated = ET.SubElement(object, 'truncated') 
                    difficult = ET.SubElement(object, 'difficult') 
                    bndbox = ET.SubElement(object, 'bndbox') 
                    xmin = ET.SubElement(bndbox, 'xmin')
                    ymin = ET.SubElement(bndbox, 'ymin')
                    xmax = ET.SubElement(bndbox, 'xmax')
                    ymax = ET.SubElement(bndbox, 'ymax') 

                    
                    difficult.text  = "0"
                    truncated.text = "1"
                    pose.text = "Unspecified"
                    name.text = element["classTitle"]
                    database.text = "Unknown"
                    folder.text="images"
                    path.text = imageFilePath + annotationFile.split(".")[0] + ".jpg"
                    filename.text = annotationFile.split(".")[0] + ".jpg"
                    # width.text = width
                    # height.text = height
                    xmin.text = str(element["points"]["exterior"][0][0])
                    ymin.text = str(element["points"]["exterior"][0][1])
                    xmax.text = str(element["points"]["exterior"][1][0])
                    ymax.text = str(element["points"]["exterior"][1][1])
                    


                # create a new XML file with the results
                mydata = ET.tostring(data)  
                print(data)
                fp.write(mydata) 
            