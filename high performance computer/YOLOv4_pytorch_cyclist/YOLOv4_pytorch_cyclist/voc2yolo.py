import os
import xml.etree.ElementTree as ET


classes = ["cyclist", "person"]


def convert_annotation(rootpath,xmlname,rootImg):
    xmlpath = rootpath
    xmlfile = os.path.join(xmlpath,xmlname)
    with open(xmlfile, "r", encoding='UTF-8') as in_file:
        tree=ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        content = rootImg + xmlname[:-4]+".jpg "
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymin = float(xmlbox.find('ymin').text)
            ymax = float(xmlbox.find('ymax').text)
            bb = [str(xmin), str(ymin),str(xmax), str(ymax), str(cls_id)]
            bb_str = ",".join(bb)+ " "
            content = content + bb_str

        return content

if __name__ == '__main__':
    root_path = r'F:\tfg/train_annotations'  #需要转换成Yolo格式的VOC xml路径
    xmls = os.listdir(root_path)
    rootImg = r'F:\tfg/train_images/' #图片所在路径 最后一个/不要忘记加
    txtPath = 'train_annotation.txt'
    with open(txtPath,'w') as f:
        for name in xmls:
            print(name)
            content = convert_annotation(root_path, name, rootImg)
            f.write(content[:-1]+"\n")

