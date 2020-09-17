# import some common libraries
import numpy as np
import os, json, cv2, random
from pathlib import Path
import argparse, glob, tqdm, time
from detectron2.structures import BoxMode
from xml.etree import ElementTree
from xml.dom import minidom

DETECTRON2_DATASETS = "/media/rahul/Karmic/data"
#DETECTRON2_DATASETS = "/home/jovyan/ws-data/data"
ROADDAMAGE_DATASET  = DETECTRON2_DATASETS+"/rdd2020"
DATASET_BASE_PATH = ROADDAMAGE_DATASET

_PREDEFINED_SPLITS_GRC_MD = {}
_PREDEFINED_SPLITS_GRC_MD["rdd2020_source"] = {
    "rdd2020_train": ( "train/Czech", 
                       "train/India", 
                       "train/Japan")
}

_PREDEFINED_SPLITS_GRC_MD["rdd2020"] ={
    "rdd2020_test"  : ( 
                       "lval/Czech", 
                       "lval/India", 
                       "lval/Japan"
                     ),
    "rdd2020_val" : ( 
                       "ltest/Czech", 
                       "ltest/India", 
                       "ltest/Japan"
                     ),
    "rdd2020_train": ( 
                       "lvaltrain/Czech", 
                       "lvaltrain/India", 
                       "lvaltrain/Japan"
                     )
}

_PREDEFINED_SPLITS_GRC_MD["rdd2020_test"] = {
    "rdd2020_test1": ( "test1/Czech", 
                       "test1/India", 
                       "test1/Japan")
}

RDD_DAMAGE_CATEGORIES_SUPER=[
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

# Submission specific class identifiers
RDD_DAMAGE_CATEGORIES=[
        {"id": 1, "name": "D00", "color": [220, 20, 60] , "submission_superid": 1, "description": "Longitudinal Crack"}, 
        {"id": 2, "name": "D10", "color": [0, 0, 142]   , "submission_superid": 2, "description": "Transverse Crack"}, 
        {"id": 3, "name": "D20", "color": [0, 60, 100]  , "submission_superid": 3, "description": "Aligator Crack"}, 
        {"id": 4, "name": "D40", "color": [0, 80, 100]  , "submission_superid": 4, "description": "Pothole"}
    ]

import random, os
random.seed(0) 
# Setup target directory structure
def prepare_target_directories(basepath, dataset_name, splits_per_dataset):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_GRC_MD["rdd2020"].items():
        for idx, regions_data in enumerate(splits_per_dataset):
            print(idx+1, ".) Creating Target Dirs--> ",basepath,"/",regions_data)
            os.makedirs(os.path.join(basepath, regions_data, "annotations/xmls"), exist_ok=True)
            os.makedirs(os.path.join(basepath, regions_data, "images"), exist_ok=True)
            os.makedirs(os.path.join(basepath, regions_data, "images_segm"), exist_ok=True)

# Prepare dataset split as Test:Train:Val => (80:15:5)
def prepare_dataset_split(basepath, dataset_name, splits_per_dataset, train_ratio=80, val_ratio=15, test_ratio=5):
    print("Prepare data split (80:15:5)", basepath, " {", dataset_name,"}  \t", splits_per_dataset)
    image_id_count = 0
    for idx, regions_data in enumerate(splits_per_dataset): # Region specific
        # Assume pre-defined datasets live in `./datasets`.
        ann_path = os.path.join(basepath, regions_data, "annotations/xmls")
        img_path = os.path.join(basepath, regions_data, "images")
        seg_path = os.path.join(basepath, regions_data, "images_segm")
        if os.path.isdir(img_path): # list annotations/xml dir and for each annotation load the data
            img_file_list = [filename for filename in os.listdir(img_path) if filename.endswith('.jpg')]
            img_file_idxs = list(range(len(img_file_list)))
            random.seed(0)
            random.shuffle(img_file_idxs)
            train_limit, val_limit = int((train_ratio/100) * len(img_file_list)), int(((train_ratio + val_ratio)/100) * len(img_file_list))
            print(idx,"\tLoading Total:", len(img_file_list),", Train Images:", train_limit, ". Read images from path = ", img_path)
            for run_id, img_file_idx in enumerate(img_file_idxs):
                image_id_count = image_id_count + 1
                img_filename = img_file_list[img_file_idx]
                ann_filename = img_filename.split(".")[0] + ".xml"
                ann_fullfile = os.path.join(ann_path, ann_filename)                
                img_fullfile = os.path.join(img_path, img_filename)
                seg_fullfile = os.path.join(seg_path, img_filename)
                if os.path.isfile(os.path.join(ann_fullfile)):
                    if run_id < train_limit: 
                        target_basepath = _PREDEFINED_SPLITS_GRC_MD["rdd2020"]["rdd2020_train"][idx] 
                    elif run_id < val_limit:
                        target_basepath = _PREDEFINED_SPLITS_GRC_MD["rdd2020"]["rdd2020_val"][idx]                     
                    else:
                        target_basepath = _PREDEFINED_SPLITS_GRC_MD["rdd2020"]["rdd2020_test"][idx] 
                    #print(run_id+1, "\t",target_basepath," : ", img_fullfile, " - ", ann_fullfile)
                    # Read images & annotation from source. Target is where a softlink file will be placed
                    ann_fullfile_target = os.path.join(basepath, target_basepath, "annotations/xmls", ann_filename)                                             
                    img_fullfile_target = os.path.join(basepath, target_basepath, "images", img_filename)
                    img_segmfile_target = os.path.join(basepath, target_basepath, "images_segm", img_filename)
                    # Soft link images & Annotation from source to target
                    if os.path.isfile(ann_fullfile_target): os.unlink(ann_fullfile_target)
                    if os.path.isfile(img_fullfile_target): os.unlink(img_fullfile_target)
                    if os.path.isfile(img_segmfile_target): os.unlink(img_segmfile_target)
                    os.symlink(ann_fullfile, ann_fullfile_target)
                    os.symlink(img_fullfile, img_fullfile_target)
                    os.symlink(seg_fullfile, img_segmfile_target)


# -------------- Detectron Loaders ------------------ #

def get_rdd_coco_instances_meta():
    thing_ids = [k["id"] for k in RDD_DAMAGE_CATEGORIES]
    thing_names = [k["name"] for k in RDD_DAMAGE_CATEGORIES]
    thing_colors = [k["color"] for k in RDD_DAMAGE_CATEGORIES]
    assert len(thing_ids) == 4, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in RDD_DAMAGE_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
        "thing_names" : thing_names,
        "stuff_classes" : []
    }
    return ret

def load_images_ann_dicts(basepath, splits_per_dataset):
    dataset_dicts = []
    thing_ids   = [k["id"] for k in RDD_DAMAGE_CATEGORIES]
    thing_names = [k["name"] for k in RDD_DAMAGE_CATEGORIES]
    image_id_count = 0
    for idx, regions_data in enumerate(splits_per_dataset):
        # Assume pre-defined datasets live in `./datasets`.
        ann_path = os.path.join(basepath, regions_data, "annotations/xmls")
        # TODO: Pre-processed segmented images in 'images_segm'
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
                img_height = int(root.find('size').find('height').text)
                img_width  = int(root.find('size').find('width').text)
                for obj in root.iter('object'):
                    cls_name, xmlbox = obj.find('name').text, obj.find('bndbox')
                    if cls_name in thing_names: # if class is interesting
                        xmin, xmax = np.float(xmlbox.find('xmin').text), np.float(xmlbox.find('xmax').text)
                        ymin, ymax = np.float(xmlbox.find('ymin').text), np.float(xmlbox.find('ymax').text)
                        bbox = [xmin, ymin, xmax, ymax]       # (x0, y0, x1, y1)  -> (x0, y0, w, h) #bbox = BoxMode.convert(bbox, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS) 
                        anno = {
                        "category_id"   : thing_names.index(cls_name),
                        "category_name" : cls_name,
                        "bbox_mode"     : BoxMode.XYXY_ABS,  # (x0, y0, w, h) 
                        "bbox"          : bbox,
                        "iscrowd"       : 0
                        }
                        annos.append(anno)
            else:
                im = cv2.imread(os.path.join(img_path, img_filename))
                img_height = im.shape[0]
                img_width  = im.shape[1]
            record["image_id"] = image_id_count
            record["image_name"] = img_filename
            record["file_name"] = os.path.join(img_path, img_filename)
            record["height"] = img_height
            record["width"] = img_width
            record["annotations"] = annos
            record["supercategory"] = regions_data
            dataset_dicts.append(record)
    return dataset_dicts

"""
Prepare data split (80:15:5) /media/rahul/Karmic/data/rdd2020  { rdd2020_train }  	 ('train/Czech', 'train/India', 'train/Japan')
    0 	Loading Total: 2829 , Train Images: 2263 . Read images from path =  /media/rahul/Karmic/data/rdd2020/train/Czech/images
    1 	Loading Total: 7706 , Train Images: 6164 . Read images from path =  /media/rahul/Karmic/data/rdd2020/train/India/images
    2 	Loading Total: 10506 , Train Images: 8404 . Read images from path =  /media/rahul/Karmic/data/rdd2020/train/Japan/images
"""
if __name__ == "__main__":
    print("\n-----------", DATASET_BASE_PATH, "-----------\n")
    prepare_target_directories(DATASET_BASE_PATH, "rdd2020_source", _PREDEFINED_SPLITS_GRC_MD["rdd2020_source"]["rdd2020_train"])
    # Setup dataset for test/train/val split
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_GRC_MD["rdd2020_source"].items():
        prepare_dataset_split(DATASET_BASE_PATH, dataset_name, splits_per_dataset)


