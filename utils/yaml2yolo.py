

import yaml
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
  
def yaml2yolo(yaml_path, class_names):  
    with open(yaml_path, 'r', encoding='gb18030') as f:
        yaml_data = yaml.safe_load(f)
    img_h, img_w = yaml_data["img_h"], yaml_data["img_w"]
    boxes = []
    for anno in yaml_data["annotations"]:  
        class_name = anno["class_name"] 
        if class_name not in class_names:  
            continue  
        cls_id = class_names.index(class_name)
        b = (anno['xmin'], anno['xmax'],  
             anno['ymin'], anno['ymax'])  
        bb = convert((img_w, img_h), b)  
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
    yaml_file = '/home/mqr/Desktop/datasets/coco128/xml/train2017/000000000009.yaml'  
    yaml2yolo(yaml_file, class_names)
