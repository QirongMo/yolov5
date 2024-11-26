

import xml.etree.ElementTree as ET  
import os  
  
def convert(size, box):  
    dw = 1. / size[0]  
    dh = 1. / size[1]  
    x = (box[0] + box[1]) / 2.0 - 1  
    y = (box[2] + box[3]) / 2.0 - 1  
    w = box[1] - box[0]  
    h = box[3] - box[2]  
    x = x * dw  
    w = w * dw  
    y = y * dh  
    h = h * dh  
    return [x, y, w, h]  
  
def xml2yolo(xml_file, class_names):  
    tree = ET.parse(xml_file)  
    root = tree.getroot()  
    size = root.find('size')  
    w = int(size.find('width').text)  
    h = int(size.find('height').text)  
    boxes = []
    for obj in root.iter('object'):  
        difficult = obj.find('difficult').text  
        cls = obj.find('name').text  
        if cls not in class_names:  
            print(cls)
            continue  
        cls_id = class_names[cls] if isinstance(class_names, dict) else class_names.index(cls)
        xmlbox = obj.find('bndbox')  
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),  
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))  
        bb = convert((w, h), b)  
        boxes.append([cls_id]+bb)
    return boxes

if __name__ == "__main__":
    class_names = [  
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',  
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',  
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',  
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',  
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',  
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',  
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',  
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',  
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',  
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',  
        'toothbrush'  
    ]
    
    # 调用函数，转换一个XML文件  
    xml_file = '/home/mqr/Desktop/datasets/coco128/xml/train2017/000000000009.xml'  
    xml2yolo(xml_file, class_names)
