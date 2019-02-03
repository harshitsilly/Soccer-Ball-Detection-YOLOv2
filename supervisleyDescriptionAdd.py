
import os
import json
import sys
# create the file structure


afiles=[]
if len(sys.argv)>=2:
    annotationFilePath = sys.argv[1]
    try:
        os.listdir(annotationFilePath)
    except Exception as e:
       
        print(annotationFilePath + " not found")
else:
    print("Provide arguments in correct order of annotationDirectory, imageDirectory,darFlowAnnotation directory")
annotationFiles = os.listdir(annotationFilePath)

for annotationFile in annotationFiles:
    with open(annotationFilePath + annotationFile, 'r+') as fp:
        data = json.load(fp)
        data["description"] = annotationFile.split(".json")[0]
        fp.seek(0)  # rewind
        json.dump(data, fp)
        fp.truncate()
