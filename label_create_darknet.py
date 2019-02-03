import os
import json
# convert supervisley to darknet format
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

annotationFiles = os.listdir("images\\images\\ann")
for annotationFile in annotationFiles:
    # data={"tags": ["train", "train"], "description": imageFile.split('.')[0], "objects": [], "size": {"height": 34, "width": 152}}
    with open("images\\images\\ann\\" + annotationFile, 'r') as fp:
        # json.dump(data, fp)
        data = json.load(fp)
    with open("images\\images\\labels\\" + annotationFile.split(".")[0] + ".txt",'w') as fp:
        print(data)
        if(data["objects"]):
            xmin = data["objects"][0]["points"]["exterior"][0][0]
            ymin = data["objects"][0]["points"]["exterior"][0][1]
            xmax = data["objects"][0]["points"]["exterior"][1][0]
            ymax = data["objects"][0]["points"]["exterior"][1][1]
            width = data["size"]["width"]
            height = data["size"]["height"]
            bb = convert([width,height],[xmin,xmax,ymin,ymax])
            data = "0" + " " + " ".join([str(a) for a in bb]) + "\n"
            fp.write(data)

