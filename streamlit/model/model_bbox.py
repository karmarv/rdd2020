# install dependencies: 
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.6")

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
import argparse, glob, tqdm, time
from xml.etree import ElementTree
from xml.dom import minidom
import pandas as pd

# import some common detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image

DETECTRON2_DATASETS = "/media/rahul/Karmic/data"
#DETECTRON2_DATASETS = "/home/jovyan/ws-data/data"
ROADDAMAGE_DATASET  = DETECTRON2_DATASETS+"/rdd2020"
DATASET_BASE_PATH = ROADDAMAGE_DATASET

_PREDEFINED_SPLITS_GRC_MD = {}
_PREDEFINED_SPLITS_GRC_MD["rdd2020"] = {
    "rdd2020_demo": ( "train_short/Czech", 
                      "train_short/India", 
                      "train_short/Japan")                       
}

RDD_DAMAGE_CATEGORIES=[
        {"id": 1, "name": "D00", "color": [220, 20, 60] , "submission_superid": 1, "description": "Longitudinal Crack"}, 
        {"id": 2, "name": "D01", "color": [165, 42, 42] , "submission_superid": 1, "description": "Longitudinal Crack"}, 
        {"id": 3, "name": "D10", "color": [0, 0, 142]   , "submission_superid": 2, "description": "Transverse Crack"}, 
        {"id": 4, "name": "D11", "color": [0, 0, 70]    , "submission_superid": 2, "description": "Transverse Crack"}, 
        {"id": 5, "name": "D20", "color": [0, 60, 100]  , "submission_superid": 3, "description": "Aligator Crack"}, 
        {"id": 6, "name": "D40", "color": [0, 80, 100]  , "submission_superid": 4, "description": "Pothole"}, 
        {"id": 7, "name": "D43", "color": [0, 0, 230]   , "submission_superid": 4, "description": "Crosswalk blur"}, 
        {"id": 8, "name": "D44", "color": [119, 11, 32] , "submission_superid": 4, "description": "Whiteline blur"}, 
        {"id": 9, "name": "D50", "color": [128, 64, 128], "submission_superid": 4, "description": "Manhole lid/plate"},
        {"id": 10,"name": "D0w0","color": [96, 96, 96]  , "submission_superid": 4, "description": "Unknown"}
    ]

RDD_DAMAGE_LABEL_COLORS = { k["name"] : k["color"] for k in RDD_DAMAGE_CATEGORIES }

damage_names = [k["name"] for k in RDD_DAMAGE_CATEGORIES]

def cv2_imshow(im):
    plt.figure(figsize=(8,8))
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

def set_configuration():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS          = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.OUTPUT_DIR             = "./model"
    cfg.MODEL.DEVICE           = "cuda"
    cfg.MODEL.WEIGHTS          = os.path.join(cfg.OUTPUT_DIR, "model_bbox_e10k_class10_19Aug-faster_rcnn_R_50_FPN_3x.pth")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES       = len(RDD_DAMAGE_CATEGORIES)  # only few class
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold for this model
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg 

cfg = set_configuration()

def format_detections(predictions):
    boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
    scores = predictions.scores.numpy() if predictions.has("scores") else None
    classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
    xmin, xmax, ymin, ymax, labels, scrs = [], [], [], [], [], []
    if classes is not None:
        for i, (clas, scr, bbx) in enumerate(zip(classes, scores, boxes)):
            (x_min, y_min, x_max, y_max) = bbx
            # class_submission_id, x_min, y_min, x_max, y_max
            # out_str = "{0} {1} {2} {3} {4} {5}".format(int(cls), int(scr), int(x_min), int(y_min), int(x_max), int(y_max))
            xmin.append(int(x_min))
            ymin.append(int(y_min))
            xmax.append(int(x_max))
            ymax.append(int(y_max))
            labels.append(damage_names[int(clas)])
            scrs.append(scr)
    boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "labels": labels, "scores": scrs})
    return boxes[["xmin", "ymin", "xmax", "ymax", "labels", "scores"]]

"""
 Predict result from image
"""
def predict_rdd(img, confidence_threshold = 0.7):
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    predictor = DefaultPredictor(cfg)
    outputs = predictor(img)
    return format_detections(outputs["instances"].to("cpu"))




def load_images_ann_dicts(basepath, splits_per_dataset):
    dataset_df = pd.DataFrame(columns=['frame', 'xmin', 'ymin', 'xmax', 'ymax', 'label'])
    
    image_id_count = 0
    for idx, regions_data in enumerate(splits_per_dataset):
        # Assume pre-defined datasets live in `./datasets`.
        ann_path = os.path.join(basepath, regions_data, "annotations/xmls")
        img_path = os.path.join(basepath, regions_data, "images")
        
        # list annotations/xml dir and for each annotation load the data
        img_file_list = [filename for filename in os.listdir(img_path) if filename.endswith('.jpg')]
        print(idx,"\tLoading ", len(img_file_list),  " images from path = ", img_path)
        for img_id, img_filename in enumerate(img_file_list):
            image_id_count = image_id_count + 1
            record = {}
            annos = []
            ann_file = img_filename.split(".")[0] + ".xml"
            if os.path.isfile(os.path.join(ann_path, ann_file)):
                infile_xml = open(os.path.join(ann_path, ann_file))
                tree = ElementTree.parse(infile_xml)
                root = tree.getroot()
                for obj in root.iter('object'):
                    cls_name, xmlbox = obj.find('name').text, obj.find('bndbox')
                    xmin, xmax = np.float(xmlbox.find('xmin').text), np.float(xmlbox.find('xmax').text)
                    ymin, ymax = np.float(xmlbox.find('ymin').text), np.float(xmlbox.find('ymax').text)
                    dataset_df = dataset_df.append({'frame': img_filename, 'xmin': int(xmin), 'ymin': int(ymin), 
                                                    'xmax': int(xmax), 'ymax': int(ymax), 'label': cls_name, 
                                                    'full_file': os.path.join(img_path, img_filename)}, ignore_index=True)
            else:
                dataset_df = dataset_df.append({'frame': img_filename, 'xmin': 0, 'ymin': 0, 
                                'xmax': 0, 'ymax': 0, 'label': None, 'full_file': os.path.join(img_path, img_filename)}, ignore_index=True)
    return dataset_df

""" 
    python model_bbox.py --input 

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RDD2020 Test")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    args = parser.parse_args()
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            start_time = time.time()
            image_fullPath = path
            img = read_image(image_fullPath, format="BGR")
            predictions = predict_rdd(img)
            print(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions)),
                    time.time() - start_time,
                )
            )