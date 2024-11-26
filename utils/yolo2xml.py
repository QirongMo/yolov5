
import xml.etree.ElementTree as ET 
from xml.dom import minidom 
  
def create_xml(img_name, img_w, img_h, txt_path, class_names, xml_path):  
    """  
    将YOLO格式的标签转换为PASCAL VOC的XML格式。  
    """  
    # 创建根元素  
    annotation = ET.Element("annotation")  
      
    # 添加文件夹、文件名和来源  
    folder = ET.SubElement(annotation, "folder").text = "images"  
    filename = ET.SubElement(annotation, "filename").text = img_name
    source = ET.SubElement(annotation, "source")  
    ET.SubElement(source, "database").text = "Unknown"  
      
    # 图像大小  
    size = ET.SubElement(annotation, "size")  
    ET.SubElement(size, "width").text = str(img_w)  
    ET.SubElement(size, "height").text = str(img_h)  
    ET.SubElement(size, "depth").text = "3"  
    
    # 读取txt并转化目标框信息
    num_classes = len(class_names)
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            if len(line) != 5:
                continue
            cls_id, x_center, y_center, width, height = line  
            cls_id = int(cls_id)
            if cls_id > num_classes-1:
                continue
            class_name = class_names[cls_id]
            # 转换为PASCAL VOC的边界框格式（x_min, y_min, x_max, y_max）  
            x_min = int((float(x_center) - float(width) / 2) * img_w)  
            y_min = int((float(y_center) - float(height) / 2) * img_h)  
            x_max = int((float(x_center) + float(width) / 2) * img_w)  
            y_max = int((float(y_center) + float(height) / 2) * img_h)  
            
            obj = ET.SubElement(annotation, "object")  
            ET.SubElement(obj, "name").text = class_name  
            ET.SubElement(obj, "pose").text = "Unspecified"  
            ET.SubElement(obj, "truncated").text = "0"  
            ET.SubElement(obj, "difficult").text = "0"  
            bndbox = ET.SubElement(obj, "bndbox")  
            ET.SubElement(bndbox, "xmin").text = str(x_min)  
            ET.SubElement(bndbox, "ymin").text = str(y_min)  
            ET.SubElement(bndbox, "xmax").text = str(x_max)  
            ET.SubElement(bndbox, "ymax").text = str(y_max)  
      
    # 写入文件  
    # tree = ET.ElementTree(annotation)  
    # tree.write(xml_path, encoding='utf-8', xml_declaration=True)  
    xml_str = ET.tostring(annotation, 'utf-8').decode('utf-8')  
    pretty_xml_str = minidom.parseString(xml_str).toprettyxml(indent="  ")  
    # 写入文件  
    with open(xml_path, 'w', encoding='utf-8') as f:  
        f.write(pretty_xml_str)  

if __name__ == "__main__":
    import cv2
    import numpy as np
    import os
    from pathlib import Path

    root_dir = Path(r"/home/mqr/Desktop/datasets/coco128")
    img_root = root_dir/"images"
    anno_root = root_dir/"labels"
    xml_root = root_dir/"xml"
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
    
    for root, dirs, files in os.walk(str(img_root)):
        root = Path(root)
        for file in files:
            filepath = root/file
            suffix = filepath.suffix.lower()
            stem = filepath.stem
            if suffix not in ['.jpg', '.jpeg', '.png']:
                continue
            img_path = str(filepath)
            relative_path = img_path.split(str(img_root)+os.sep)[-1]
            anno_path = (anno_root/relative_path).with_name(stem+'.txt')
            if not anno_path.exists():
                continue
            img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
            img_h, img_w = img.shape[:2]
            xml_path = (xml_root/relative_path).with_name(stem+'.xml')
            xml_path.parent.mkdir(exist_ok=True, parents=True)

            create_xml(file, img_w, img_h, anno_path, class_names, xml_path)