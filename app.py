from flask import Flask, jsonify, request ,Response ,send_file
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
from darkflow.net.build import TFNet
import cv2
import os,sys
from zipfile import ZipFile
import zipfile as zf
from io import BytesIO
# app logging
from subprocess import Popen, PIPE
import logging
import time




import pytesseract

app = Flask(__name__)
img1 = None
# this function is used to create boundary box across the detected object
def boxing(original_img , predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))
        
        if confidence > 0.1:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
        
    return newImage



# post call for detecting licenseplate number
@app.route('/processLicensePlateImage', methods=['POST'])
def processLicensePlateImage():
    global tfnet2
    imageFile = request.files['file']
    img1 = Image.open(imageFile)
    frame = np.asarray(img1)
    results = tfnet2.return_predict(frame)
  
    # boxing(img1, results)
   
    newImage = np.copy(img1)
    for result in results:
        
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))

        if confidence > 0.1:
            newImage = newImage[top_y:btm_y, top_x:btm_x]
            # newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
    text = getPredictedText(newImage)  
    return jsonify(text)

# post call for detecting licenseplate number
@app.route('/getLPBBImage', methods=['POST'])
def getLPBBImage():
    global tfnet2,img1
    imageFile = request.files['file']
    img1 = Image.open(imageFile)
    frame = np.asarray(img1)
    results = tfnet2.return_predict(frame)
    newImage = boxing(img1, results)
    im = Image.fromarray(newImage)
    img_io = BytesIO()
    im.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io,mimetype='image/jpeg')

@app.route('/rejectLpImage',methods=['GET'])
def rejectLpImage():
    global img1
    rejectedimagesFolder = "Rejected Images"
    rejectImagePath = os.path.join("{}/{}.jpg".format(rejectedimagesFolder,len(os.listdir(rejectedimagesFolder))))
    img1.save(rejectImagePath,"JPEG")
    app.logger.info("image successfully saved")
    return "image successfully saved"

@app.route('/getPredictedText', methods=['POST'])
def getLPPredictedText():
    imageFile = request.files['file']
    img1 = Image.open(imageFile)
    newImage = np.copy(img1)
    text = getPredictedText(newImage)  
    return jsonify(text)


# post call to get object boundary box 
@app.route('/getObjectBoundaryBox', methods=['POST'])
def getObjectBoundaryBox():
    global tfnet2
    imageFile = request.files['file']
    img1 = Image.open(imageFile)
    frame = np.asarray(img1)
    results = tfnet2.return_predict(frame)
  
    return jsonify(results)

# convert rgb to gray
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


# to get text from the image keras model used
def getPredictedText(img):
    ocrModel,ocrSession = getOCRModel()
    net_inp = ocrModel.get_layer(name='the_input').input
    net_out = ocrModel.get_layer(name='softmax').output
    # img = rgb2gray(img)
    img = np.resize(img,(64,128))
    img = np.expand_dims(img, 0)
    img = img.T
    # X_data = [img]
    X_data = np.ones([1, 128, 64,1])
    X_data[0] = img
    net_out_value = ocrSession.run(net_out, feed_dict={net_inp:X_data})
    pred_texts = decode_batch(net_out_value)
    return pred_texts

# save cropped object image
def saveLicensePlateImage(ifile,fileName,folderPath,isCrop):
    global model,session
    
    img1 = Image.open(ifile)
    frame = np.asarray(img1)
   
    try:
        predictions = tfnet2.return_predict(frame)
        if(isCrop):
            if len(predictions) > 0:
            
                newImage = np.copy(img1)
                for result in predictions:
                    
                    top_x = result['topleft']['x']
                    top_y = result['topleft']['y']

                    btm_x = result['bottomright']['x']
                    btm_y = result['bottomright']['y']

                    confidence = result['confidence']
                    label = result['label'] + " " + str(round(confidence, 3))

                    if confidence > 0.1:
                        newImage = newImage[top_y:btm_y, top_x:btm_x]
                       
        else:
            newImage = boxing(img1, results)

        im = Image.fromarray(newImage)
        im.save("scImages\\" + folderPath +"\\" + fileName)  
    except Exception as e:
        print("error in making new image")

    
# post call to extract data from zip and save all the objects detected image
@app.route('/processLicensePlateZip', methods=['POST'])
def processLicensePlateZip():
    zipfile = request.files['file']
    folderPath = request.form['folderPath']
    isCrop = request.form['isCrop']
    try:
        os.getcwd("\\scImages\\" + folderPath)
    except Exception as e:
        os.mkdir(os.path.join(os.getcwd() + "\\scImages\\" + folderPath))
    
    with ZipFile(zipfile) as archive:
        for entry in archive.infolist():
            with archive.open(entry) as file:
                fileName = file.name
                saveLicensePlateImage(file,fileName,folderPath,isCrop)
              
    return "Successfully created data for ocr"

# get tfjs model.json
@app.route('/getTFjsModel', methods=['Get'])
def getTFjsModel():
    return send_file(os.path.join(os.getcwd() + "/built_graph/tfjs/kerasYoloV2/model.json"))

@app.route('/getProcessingtime',methods=['Get'])
def getProcessingtime():
    global tfnet2
    original_img = cv2.imread(os.path.join(os.getcwd() +"/scImages/new dataset/Hyundai-elite-i20-vs-i20-Active-Comparison-test-e1465889837605.jpg"))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    t0 = time.time()
    results = tfnet2.return_predict(original_img)
    t1 = time.time()
    total = t1-t0
    return str(total)

    
@app.route('/<shard>', methods=['Get'])
def getshardFile(shard):
    return send_file(os.path.join(os.getcwd() + "/built_graph/tfjs/YoloTiny/" + shard))


 # with open("ckpt/checkpoint",'w') as fp:
        #     fp.writelines(['model_checkpoint_path: "yolo_tiny-25750"\n','all_model_checkpoint_paths: "yolo_tiny-25750"'])

        # options = {"model": "cfg/yolo_tiny.cfg",
        #        "load": -1,

        #        "gpu": 0}
@app.route("/setYoloType",methods=['Post'])
def setYoloType():
    global tfnet2,tfnet,tfnet1
    isYoloTiny= request.form['yoloTiny']
    if isYoloTiny=='X':
       
        tfnet2 = tfnet1
        model = "yoloTinyV2"       
        
    else:
        
        tfnet2 = tfnet
        model = "yoloV2"   

    tfnet = TFNet(options)
   
    tfnet2 = tfnet
    app.logger.info('Yolo {} sucessfully setted'.format(model))
    return "Yolo {} sucessfully setted".format(model)

#Datset upload functionality
#upload Dataset zip file
@app.route("/uploadDataset",methods=['Post'])
def uploadDataset():
    zipfile = request.files['file']
    directory_to_extract_to = "Build/Dataset/{}.zip".format(len(os.listdir("Build/Dataset")))
    zipfile.save(directory_to_extract_to)
    app.logger.info('%s ZipFile successfully uploaded')
    return "ZipFile successfully uploaded"



#Dataset Checker -  annImageShapeCheck.py
#utility checket - annImgCheck.py
@app.route("/getLogs",methods=['Get'])
def getLogs():
    with open("error.log",'r') as fp:
        text = fp.read()
    return text

@app.route("/getRejectedImageZip",methods=['Get'])
def getRejectedImageZip():
    filename = os.path.join(os.getcwd() + "/" + 'RejectImages.zip')
    # zipf = zipfile(filename, 'w', zipfile.ZIP_DEFLATED)
    rejectedFolderlist  =  os.listdir(os.path.join(os.getcwd() + "/Rejected Images"))
    zipData = BytesIO()
    zipf = ZipFile(filename, 'w', zf.ZIP_DEFLATED)
    for file in rejectedFolderlist:
        if file != "" :
            fileName = file
            zipf.write(os.path.join(os.getcwd() + "/Rejected Images/")  + file,fileName)
    zipf.close()
    return send_file(filename)
                # zipf.seek(0)
    # file.write("")
    # thread_a = Compute('RejectImages.zip')
    # thread_a.start()
    

# model build functionality 
# if images zip not given the model will use the files placed in Voc_pascal Dataset
@app.route("/trainYoloModel",methods=['Post'])
def trainYoloModel():
    config = request.form["config"] if "config" in request.form.keys() else None
    zipFile = request.files['file'] if "file" in request.form.keys() else None
    Zipfileused ="Last Uploaded {}.zip".format(len(os.listdir("Build/Dataset")))
    options = {"model": "cfg/yolo_custom.cfg", 
           "load": "yolo.weights",
           "batch": config["batch"] if config  else 8,
           "epoch": config["epoch"] if config  else 100,
           "gpu": config["gpu"] if config  else 0,
           "train": True,
           "annotation": "Build/Dataset/{}/ann".format(config[Dataset] if config else len(os.listdir("Build/Dataset")) ),
           "dataset": "Build/Dataset/{}/img".format(config[Dataset] if config else len(os.listdir("Build/Dataset")))}
    if zipFile:
        # default directory
        Zipfileused = "User"
        directory_to_extract_to = "Build/Dataset/{}/".format(len(os.listdir("Build/Dataset")))
        zip_ref = ZipFile.zipFile(zipFile, 'r')
        zip_ref.extractall(directory_to_extract_to)
        zip_ref.close()
        options["annotation"] = "{}/ann".format(directory_to_extract_to)
        options["dataset"] = "{}/img".format(directory_to_extract_to)
    #tf model has to be trained on a separate python file
    tfnet = TFNet(options)
    tfnet.train()
    app.logger.info('%s Model successfully sttarted. {} zipfile is uded for training'.format(Zipfileused))
    return "Model successfully sttarted. {} zipfile is used for training".format(Zipfileused)
    
    


#server initialization
if __name__ == "__main__":
    print("* Starting web server... please wait until server has fully started")
    port = int(os.getenv("PORT", 5000)) 
    # app.run(host="0.0.0.0",port=port) #run app in debug mode on port 5000
    # app.run(port=port)
    with open("ckpt/checkpoint",'w') as fp:
            fp.writelines(['model_checkpoint_path: "yolo_custom-18750"\n','all_model_checkpoint_paths: "yolo_custom-18750"'])
    # options = {"model": "cfg/yolo_custom.cfg",
    #         "load": -1,
    #         "gpu": 0}



    optionsYoloTiny = {"model": "cfg/yolo_tiny.cfg",
               "pbLoad" : "built_graph/yolo_tiny.pb"  ,
               "metaLoad": "built_graph/yolo_tiny.meta" ,

               "gpu": 0}
    options = {"model": "cfg/yolo_custom.cfg",
               "pbLoad" : "built_graph/yolo_custom.pb"  ,
               "metaLoad": "built_graph/yolo_custom.meta" ,
             "gpu": 0}
   

    if len(sys.argv)>=2:
        port = sys.argv[1]
    logging.basicConfig(filename='error.log',level=logging.DEBUG)
    # try:
    tfnet = TFNet(options)
    tfnet1 = TFNet(optionsYoloTiny)
    tfnet2 = tfnet
    from ocr_model import decode_batch,getOCRModel,getImageData
    app.run(debug=False,host='localhost', port=port,threaded=True)
        
        

    # except Exception as e:
    #     print(e)
    
   