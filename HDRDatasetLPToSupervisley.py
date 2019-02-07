import os
import json
import csv
from PIL import Image


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
csv_FilePath = "alphaNumericDataset/2017-IWT4S-HDR_LP-dataset/trainVal.csv"
lines = []
with open(csv_FilePath, 'r') as f:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            fp = row["image_path"].split("./")[1]
            image = Image.open(fp)
            width = image.size[0]
            height = image.size[1]
            image_url = fp
            fileName = image_url.split("/")[-1]
            index_of_dot = fileName.index('.')
            file_name_without_extension = fileName[:index_of_dot]
            bbx_labelsFinal = {}
            finalJson = {"tags": ["train"], "description": file_name_without_extension, "objects": bbx_labelsFinal, "size": {"width": width, "height": height}}
            with open("alphaNumericDataset/2017-IWT4S-HDR_LP-dataset/ann/" + file_name_without_extension + ".json", 'w') as fp:
                json.dump(finalJson, fp)

        except Exception as e:
            print("Unable to process item " + file_name_without_extension + "\n" + "error = "  + str(e))
        
