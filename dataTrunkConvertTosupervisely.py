import os
import json


def getBoundaryBox(bbx_label, bbx_data, width, height):
    if len(bbx_data['points']) == 4:
        #Regular BBX has 4 points of the rectangle.
        xmin = width*min(bbx_data['points'][0][0], bbx_data['points'][1][0], bbx_data['points'][2][0], bbx_data['points'][3][0])
        ymin = height * min(bbx_data['points'][0][1], bbx_data['points'][1][1], bbx_data['points'][2][1],
                           bbx_data['points'][3][1])

        xmax = width * max(bbx_data['points'][0][0], bbx_data['points'][1][0], bbx_data['points'][2][0],
                           bbx_data['points'][3][0])
        ymax = height * max(bbx_data['points'][0][1], bbx_data['points'][1][1], bbx_data['points'][2][1],
                           bbx_data['points'][3][1])

    else:
        #OCR BBX format has 'x','y' in one point.
        # We store the left top and right bottom as point '0' and point '1'
        xmin = int(bbx_data['points'][0]['x']*width)
        ymin = int(bbx_data['points'][0]['y']*height)
        xmax = int(bbx_data['points'][1]['x']*width)
        ymax = int(bbx_data['points'][1]['y']*height)
    null = None    
    data= {
            "description": "",
            "tags": [],
            "bitmap": null,
            "classTitle": "licensePlate",
            "points": {
                "exterior": [
                [
                    xmin,
                    ymin
                ],
                [
                    xmax,
                    ymax
                ]
                ],
                "interior": []
            }
            }

    return data

# "custom_dataset/Car_License_Plate_Detection.json"
dataturks_JSON_FilePath = "custom_dataset/Indian_Number_plates.json"
lines = []
with open(dataturks_JSON_FilePath, 'r') as f:
    lines = f.readlines()

for line in lines:
    try:
        data = json.loads(line)
        

        width = data['annotation'][0]['imageWidth']
        height = data['annotation'][0]['imageHeight']
        image_url = data['content']
        fileName = image_url.split("/")[-1]
        index_of_dot = fileName.index('.')
        file_name_without_extension = fileName[:index_of_dot]
        for bbx in data['annotation']:
                if not bbx:
                    continue
                #Pascal VOC only supports rectangles.
                if "shape" in bbx and bbx["shape"] != "rectangle":
                    continue

                bbx_labels = bbx['label']
                bbx_labelsFinal =[]
                #handle both list of labels or a single label.
                if not isinstance(bbx_labels, list):
                    bbx_labelsFinal.append(getBoundaryBox(bbx_label, bbx, width, height))

                for bbx_label in bbx_labels:
                    bbx_labelsFinal.append(getBoundaryBox(bbx_label, bbx, width, height))

        finalJson = {"tags": ["train"], "description": "", "objects": bbx_labelsFinal, "size": {"width": width, "height": height}}
        with open("custom_dataset\\images\\ann\\" + file_name_without_extension + ".json", 'w') as fp:
            json.dump(finalJson, fp)

    except Exception as e:
        print("Unable to process item " + file_name_without_extension + "\n" + "error = "  + str(e))
        
