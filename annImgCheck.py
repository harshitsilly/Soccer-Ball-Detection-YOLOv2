import os
from PIL import Image
import cv2

imageFiles = os.listdir("custom_dataset1/images/img")
filesToBeDeleted = []
annotationFileExist = {}
for imageFile in imageFiles:
    
    index = imageFile.index('.')
    fileExtension = imageFile[index:len(imageFile)]
    annotationFile = imageFile[:index]
    
    try :
        print(imageFile)
        img = cv2.imread("custom_dataset1/images/img/" +imageFile)
        h, w, c = img.shape 
        # if fileExtension not in [".jpg",".JPG"]:
            
        #     im = Image.open("custom_dataset/images/img" + "/" + imageFile)
        #     rgb_im = im.convert('RGB')
        #     rgb_im.save("custom_dataset/images/img" + "/" + annotationFile+ ".jpg")
        #     filesToBeDeleted.append("custom_dataset/images/img/" + imageFile)
        
        
        if(os.path.exists("custom_dataset1/images/ann/" + annotationFile+".xml")):
            print(imageFile)
            annotationFileExist[annotationFile+".xml"] = True

        else:
            filesToBeDeleted.append("custom_dataset1/images/img/" + imageFile)
    
    except Exception as e :
        filesToBeDeleted.append("custom_dataset1/images/img/" + imageFile)
        if(os.path.exists("custom_dataset1/images/ann/" + annotationFile+".xml")):
            print(imageFile)
            filesToBeDeleted.append("custom_dataset1/images/ann/" + annotationFile +".xml")
        

annotationFiles = os.listdir("custom_dataset/images/ann")
print("AnnotatedFilesExist:" + str(len(annotationFileExist)))
print(filesToBeDeleted)
print("imagefiles to be deleted:" + str(len(filesToBeDeleted)))
# for deleteFile in filesToBeDeleted:
#     print("remove" +imageFile)
#     os.remove(deleteFile)

print("annotaionfiles to be deleted:" + str((len(annotationFiles) - len(annotationFileExist))))
# for annotationFile in annotationFiles:
#     if annotationFile in annotationFileExist:
#         print("FileExist" +annotationFile)
#     else:
#         os.remove("custom_dataset/images/ann/" + annotationFile)
       

    

