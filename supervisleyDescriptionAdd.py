
import os
import json
import sys
# create the file structure


afiles=[]
if len(sys.argv)>=2:
    imageFilePath = sys.argv[1]
    annotationFilePath = sys.argv[2]
    try:
        os.listdir(imageFilePath)
    except Exception as e:
       
        print(imageFilePath + " not found")
else:
    print("Provide arguments in correct order of annotationDirectory, imageDirectory,darFlowAnnotation directory")
imageFiles = os.listdir(imageFilePath)

for imageFile in imageFiles:
    with open(annotationFilePath + imageFile, 'r+') as fp:
        data = json.load(fp)
        data["description"] = annotationFile.split(".json")[0]
        fp.seek(0)  # rewind
        json.dump(data, fp)
        fp.truncate()
