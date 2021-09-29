import os
import numpy as np
import shutil










if __name__ == '__main__':
    root_annotation = r'F:\tfg\SSD_cyclist\SSD_cyclist\datasets\VOC/Annotations' #原始标注文件路径
    root_imgs = r'F:\tfg\SSD_cyclist\SSD_cyclist\datasets\VOC/JPEGImages'  #原始图片路径
    train_annotation = r'F:\tfg/train_annotations' #划分的训练集标注路径
    train_imgs = r'F:\tfg/train_images' #划分的训练集图片路径
    val_annotation = r'F:\tfg/val_annotations' #划分的验证集标注路径
    val_imgs = r'F:\tfg/val_images' #划分的验证集图片路径
    val_txt = r'F:\tfg/val.txt'
    os.makedirs(train_imgs,exist_ok=True)
    os.makedirs(train_annotation, exist_ok=True)
    os.makedirs(val_imgs, exist_ok=True)
    os.makedirs(val_annotation, exist_ok=True)
    np.random.seed(8)
    xmls = os.listdir(root_annotation)
    np.random.shuffle(xmls)
    valNum = int(len(xmls)*0.2)  #8:2划分训练集和验证集
    for i in range(0,len(xmls)-valNum):
        img_name = xmls[i][:-4]+".jpg"
        img_path = os.path.join(train_imgs,img_name)
        xml_path = os.path.join(train_annotation, xmls[i])
        shutil.copy(os.path.join(root_annotation,xmls[i]),xml_path)
        shutil.copy(os.path.join(root_imgs,img_name),img_path)
        print(i,img_name)
    with open(val_txt,'w') as f:
        for i in range(len(xmls) - valNum, len(xmls)):
            img_name = xmls[i][:-4] + ".jpg"
            img_path = os.path.join(val_imgs, img_name)
            xml_path = os.path.join(val_annotation, xmls[i])
            shutil.copy(os.path.join(root_annotation, xmls[i]), xml_path)
            shutil.copy(os.path.join(root_imgs, img_name), img_path)
            print(i, img_name)
            f.write(xmls[i][:-4]+"\n")





