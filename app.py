from flask import Flask, jsonify, request ,Response ,send_file
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
from darkflow.net.build import TFNet
import cv2
import os,sys
from zipfile import ZipFile
from io import BytesIO

import pytesseract

app = Flask(__name__)

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
    global tfnet2
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

@app.route('/<shard>', methods=['Get'])
def getshardFile(shard):
    return send_file(os.path.join(os.getcwd() + "/built_graph/tfjs/kerasYoloV2/" + shard))

@app.route("/setYoloType",methods=['Post'])
def setYoloType():
    global tfnet2
    isYoloTiny= request.form['yoloTiny']
    if isYoloTiny=='X':
        with open("ckpt/checkpoint",'w') as fp:
            fp.writelines(['model_checkpoint_path: "yolo_tiny-25750"\n','all_model_checkpoint_paths: "yolo_tiny-25750"'])

        options = {"model": "cfg/yolo_tiny.cfg",
               "load": -1,

               "gpu": 0}
        model = "yoloV2"       
        
    else:
        with open("ckpt/checkpoint",'w') as fp:
            fp.writelines(['model_checkpoint_path: "yolo_custom-18750"\n','all_model_checkpoint_paths: "yolo_custom-18750"'])
        options = {"model": "cfg/yolo_custom.cfg",
               "load": -1,

               "gpu": 0}
        model = "yoloTinyV2"   

    tfnet = TFNet(options)
   
    tfnet2 = tfnet
    return "Yolo {} sucessfully setted".format(model)



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



    options = {"model": "cfg/yolo_custom.cfg",
               "load": -1,

               "gpu": 0}

   

    if len(sys.argv)>=2:
        port = sys.argv[1]
       
    # try:
    tfnet = TFNet(options)
    # tfnet1 = TFNet(options1)
    tfnet2 = tfnet
    from ocr_model import decode_batch,getOCRModel,getImageData
    app.run(host='localhost', port=port)
        
        

    # except Exception as e:
    #     print(e)
    
   