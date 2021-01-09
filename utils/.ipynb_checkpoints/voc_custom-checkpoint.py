import sys

sys.path.append("..")
import xml.etree.ElementTree as ET
import config.yolov4_config as cfg
import os
from tqdm import tqdm


def parse_voc_annotation(
    data_path, file_type, anno_path, use_difficult_bbox=False
):
    """
    phase pascal voc annotation, eg:[image_global_path xmin,ymin,xmax,ymax,cls_id]
    :param data_path: eg: VOC\VOCtrainval-2007\VOCdevkit\VOC2007
    :param file_type: eg: 'trainval''train''val'
    :param anno_path: path to ann file
    :param use_difficult_bbox: whither use different sample
    :return: batch size of data set
    """
    if cfg.TRAIN["DATA_TYPE"] == "VOC":
        classes = cfg.VOC_DATA["CLASSES"]
    elif cfg.TRAIN["DATA_TYPE"] == "COCO":
        classes = cfg.COCO_DATA["CLASSES"]
    else:
        classes = cfg.Customer_DATA["CLASSES"]
    img_inds_file = os.path.join(
        data_path, "ImageSets", "Main", file_type + ".txt"
    )
    with open(img_inds_file, "r") as f:
        lines = f.readlines()
        image_ids = [line.strip() for line in lines]

    with open(anno_path, "a") as f:
        for image_id in tqdm(image_ids):
            new_str = ''
            image_path = os.path.join(
                data_path, "JPEGImages", image_id + cfg.IMAGE_NAME_EXTENSION
            )
            annotation = image_path
            label_path = os.path.join(
                data_path, "Annotations", image_id + ".xml"
            )
            root = ET.parse(label_path).getroot()
            objects = root.findall("object")
            for obj in objects:
                difficult = obj.find("difficult").text.strip()
                if (not use_difficult_bbox) and (
                    int(difficult) == 1
                ):  # difficult 表示是否容易识别，0表示容易，1表示困难
                    continue
                bbox = obj.find("bndbox")
                class_id = classes.index(obj.find("name").text.lower().strip())
                xmin = bbox.find("xmin").text.strip()
                ymin = bbox.find("ymin").text.strip()
                xmax = bbox.find("xmax").text.strip()
                ymax = bbox.find("ymax").text.strip()
                new_str += " " + ",".join(
                    [xmin, ymin, xmax, ymax, str(class_id)]
                )
            if new_str == '':
                continue
            annotation += new_str
            annotation += "\n"
            # print(annotation)
            f.write(annotation)
    return len(image_ids)


if __name__ == "__main__":
    # train_set : 
    train_data_path = os.path.join(
        cfg.DATA_PATH, 'Train'
    )
    train_annotation_path = os.path.join("../data", "train_annotation.txt")
    if os.path.exists(train_annotation_path):
        os.remove(train_annotation_path)

    # val_set   : 
    test_data_path = os.path.join(
        cfg.DATA_PATH, 'Test'
    )
    test_annotation_path = os.path.join("../data", "test_annotation.txt")
    if os.path.exists(test_annotation_path):
        os.remove(test_annotation_path)

    len_train = parse_voc_annotation(
        train_data_path,
        "trainval",
        train_annotation_path,
        use_difficult_bbox=False,
    )
    len_test = parse_voc_annotation(
        test_data_path,
        "test",
        test_annotation_path,
        use_difficult_bbox=False,
    )

    print(
        "The number of images for train and test are :train : {0} | test : {1}".format(
            len_train, len_test
        )
    )