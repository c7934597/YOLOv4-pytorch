import sys

sys.path.append("..")
import glob
import os
import pickle
import xml.etree.ElementTree as ET
from os import listdir
from os.path import join
import config.yolov4_config as cfg

# Convert PascalVOC Annotations to YOLO 需先把圖片跟XML檔放在Images資料夾
# 產生檔案路徑檔、座標檔，可是還不能用pytorch yolo訓練，因為要檔案路徑+座標的合併檔
# https://gist.github.com/Amir22010/a99f18ca19112bc7db0872a36a03a1ec

'''檔案路徑參數設定'''

# (String)訓練圖片資料路徑
TRAIN_DATA_PATH = cfg.TRAIN_DATA_PATH

# (String)測試圖片資料路徑
TEST_DATA_PATH = cfg.TEST_DATA_PATH

# (String List)VOC轉YOLO路徑列表
CONVERT_DIRS = [TRAIN_DATA_PATH, TEST_DATA_PATH]


'''資料參數設定'''

# (String)圖片副檔名
IMAGE_NAME_EXTENSION = '.png'

# (String List)分類項目
CLASSES_LIST = ['phone']


def getImagesInDir(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*'+IMAGE_NAME_EXTENSION):
        image_list.append(filename)

    return image_list

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(dir_path, output_path, image_path):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]

    in_file = open(dir_path + '/' + basename_no_ext + '.xml')
    out_file = open(output_path + basename_no_ext + '.txt', 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in CLASSES_LIST or int(difficult)==1:
            continue
        cls_id = CLASSES_LIST.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        
if __name__ == "__main__":
    for dir_path in CONVERT_DIRS:
        output_path = dir_path +'/yolo/'

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        image_paths = getImagesInDir(dir_path)
        list_file = open(dir_path + '.txt', 'w')

        for image_path in image_paths:
            image_path = image_path.replace('.png','')
            list_file.write(image_path + '\n')
            convert_annotation(dir_path, output_path, image_path)
        list_file.close()

        print("Finished processing: " + dir_path)