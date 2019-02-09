#check whether the iou is greater than the previous models. 
#for this validation set should have practical examples with proper annotaion in pascal_voc_format.(right now only done for single result)
#visualization added to check the new model performance.

import sys,os
import numpy as np
from PIL import Image
import io
from darkflow.net.build import TFNet
import cv2
from annImageShapeCheck import pascal_voc_clean_xml
import random
import numpy as np
import matplotlib.pyplot as plt
from math import ceil





def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou


def boxing(original_img, bbox,predictedbbox,iou):
    global iolessthan50Path
    newImage = np.copy(original_img)
    i = 0
    for result in bbox:
        # if confidence > 0.1:
        label =  str(iou)
        newImage = cv2.rectangle(newImage, (result[1], result[2]), (result[3], result[4]), (255,0,0), 3)
        newImage = cv2.rectangle(newImage, (predictedbbox[i][0], predictedbbox[i][1]), (predictedbbox[i][2], predictedbbox[i][3]), (255,255,0), 3)
        newImage = cv2.putText(newImage, label, (result[1], result[2]-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
        i = i +1
        if iou < 0.7:
            cv2.imwrite('{}/result/{}.jpg'.format(iolessthan50Path,len(os.listdir(iolessthan50Path + "/result"))),newImage)
            cv2.imwrite('{}/original/{}.jpg'.format(iolessthan50Path,len(os.listdir(iolessthan50Path + "/original"))),original_img)
    return newImage


def plot(iou):
    global iRange
    # ncols=2
    # fig, ax = plt.subplots(nrows= int(iRange/ncols), ncols=2, figsize=(50, 50))
    i = 0
    for imageObject in iou:
        original_img = cv2.imread(imageObject["imgpath"])
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        boxing(original_img, imageObject["bbox"],imageObject["predictedbbox"],imageObject["iou"])
        # fig.colorbar(ax[ceil(i/ncols)-1, i%ncols].imshow()
        i= i +1
    
    # plt.show()


def calculateIou(tfnet):
    global annotaionHashMap,imgFolderPath
    iou = []    
    for value in annotaionHashMap.items():
        iouObject = {}
        imgpath = "{}/{}".format(imgFolderPath,value[1][0])
        img1 = Image.open(imgpath)
        frame = np.asarray(img1)
        try:
            results = tfnet.return_predict(frame)
            iouObject["imgpath"] = imgpath
            iouObject["bbox"] = []
            iouObject["predictedbbox"] = []
            iouObject["confidence"] = 0
            try:
                for bbox in value[1][1][2]:
                    resulbbox = (results[0]["topleft"]["x"],results[0]["topleft"]["y"],results[0]["bottomright"]["x"],results[0]["bottomright"]["y"])
                    intIou = bb_intersection_over_union((bbox[1],bbox[2],bbox[3],bbox[4]),resulbbox)
                    iouObject["bbox"].append(bbox)
                    iouObject["predictedbbox"].append(resulbbox)
                    iouObject["iou"] = intIou
                    iouObject["confidence"] = results[0]['confidence']
                iou.append(iouObject)
            except Exception as e:
                iouObject["imgpath"] = imgpath
                iouObject["bbox"] = []
                iouObject["predictedbbox"] = [None]
                iouObject["confidence"] = 0
                iouObject["iou"] = 0
                for bbox in value[1][1][2]:
                    iouObject["bbox"].append(bbox)
                iou.append(iouObject)
                print("No boundary box detected")
        except Exception as e:
            iouObject["imgpath"] = imgpath
            iouObject["bbox"] = []
            iouObject["predictedbbox"] = [None]
            iouObject["confidence"] = 0
            iouObject["iou"] = 0
            for bbox in value[1][1][2]:
                iouObject["bbox"].append(bbox)
            iou.append(iouObject)
            print(e)
            
    return iou


def calculateParameters(ModelIou):
    tp = 0
    fp = 0
    fn = 0
    for i in range(0,len(newModelIou) - 1):
        predictedbbox = ModelIou[i]["predictedbbox"]
        bbox = ModelIou[i]["bbox"]
        iou = ModelIou[i]["iou"]
        confidence = ModelIou[i]["confidence"]
        if predictedbbox[0] and bbox[0]:
            if iou > 0.4 and confidence > 0.2:
                tp = tp + 1
            else:
                fn = fn + 1
        if not predictedbbox[0] and bbox[0]:
            fn = fn +1
        if predictedbbox[0] and not bbox[0]:
            fp = fp + 1
    try:
        recall = (tp/(tp + fn))
    except Exception as e:
        recall = None
        print(e)
    try:
        precision = (tp/(tp + fp))
    except Exception as e:
        precision = None
        print(e)
    
    return recall,precision
    

def checkIouPercentage(newModelIou,oldModelIou):
    changeinIou = []
    newRecall,newPrecision = calculateParameters(newModelIou)
    oldRecall,oldPrecision = calculateParameters(oldModelIou)
    for i in range(0,len(newModelIou)):
        newIou = newModelIou[i]["iou"]
        oldIou = oldModelIou[i]["iou"]
        try:
            changeinIou.append(((newIou - oldIou)/oldIou) * 100)
        except Exception as e:
            if not oldIou or oldIou == 0:
                changeinIou.append(100)
            else:
                print(e)

    
    changeiniouNumpy = np.array(changeinIou) 
    histogramBin = [-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100]
    hist,bins = np.histogram(changeiniouNumpy,bins = histogramBin ) 
    plt.hist(changeiniouNumpy, bins = histogramBin) 
    plt.title("histogram") 
    plt.show()
    print(newRecall,newPrecision)
    print(oldRecall,oldPrecision)


if len(sys.argv)>=1:
    if not os.listdir("ckpt"):
        os.mkdir("ckpt")
    iRange = 30
    iolessthan50Path = "Build/Iou50/"
    validationDataset = "Build/Validation/"
    #confidenceThresold = 0.3
    #iouThresold = 0.5
    try:
        os.listdir(iolessthan50Path)
    except Exception as e:
        os.mkdir(iolessthan50Path)
    try:
        os.listdir(validationDataset)
    except Exception as e:
        os.mkdir(validationDataset)
    
    ann = os.path.join(os.getcwd() + "/{}ann".format(validationDataset))
    className = ["licensePlate"]
    annData  = pascal_voc_clean_xml(ann,className)
    imgFolderPath = os.path.join(os.getcwd() + "/{}img".format(validationDataset))
    annotaionHashMap = {}
    
    annotationFiles = os.listdir(ann)
    
    for i in range (0,iRange):
        def checkHasMap(index) :
            if index not in annotaionHashMap.keys():
                annotaionHashMap[index] = annData[index]
            else:
                index = random.randrange(0, len(annotationFiles) -1)
                checkHasMap(index)
        index = random.randrange(0, len(annotationFiles) -1)
        checkHasMap(index)
    # for time consideration range is set to default 30. 
    

    options = {"model": "cfg/yolo_custom.cfg",
               "pbLoad" : "built_graph/yolo_custom12750.pb"  ,
               "metaLoad": "built_graph/yolo_custom12750.meta" ,
             "gpu": 0}
    tfnet = TFNet(options)
    oldModelIou = calculateIou(tfnet)
    options = {"model": "cfg/yolo_custom.cfg",
               "pbLoad" : "built_graph/yolo_custom.pb"  ,
               "metaLoad": "built_graph/yolo_custom.meta" ,
             "gpu": 0}
    tfnet = TFNet(options)
    newModelIou = calculateIou(tfnet)
    print(newModelIou)
    # plot(newModelIou)
    checkIouPercentage(newModelIou,oldModelIou)
    # checkIouPercentage() - iou improvement over the previous ones
    #lp detected with confidence - images to be saved with confidence less thane .5
    # saveImagesWithIOUlessthen50() - that is already saved in the iou folder
    # statistics plot for visualization
    #   like - dataset involved, lp detected, confidence , iou, recall,precision
    # add false negative and false positive terms  - for this first the fix should have to be done on lp predicted result
    # if the model good than use this model to run the server
else:
    print("provide valid arguments")


