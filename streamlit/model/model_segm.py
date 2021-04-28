import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from pathlib import Path
import argparse, glob, tqdm, time
import matplotlib.pyplot as plt
from copy import deepcopy

from xml.etree import ElementTree
from xml.dom import minidom

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from deeplab import add_deeplab_config, build_lr_scheduler


# Environment variable setup
DETECTRON2_DATASETS = "/media/rahul/Karmic/data/"
ROADDAMAGE_DATASET  = os.path.join(DETECTRON2_DATASETS, "rdd2020/")
DATASET_BASE_PATH   = ROADDAMAGE_DATASET
WINDOW_NAME         = "Semantic Road Segmentations"

_PREDEFINED_SPLITS_GRC_MD = {}
_PREDEFINED_SPLITS_GRC_MD["rdd2020_source"] = {
    "rdd2020_train": ( "train/Czech", 
                       "train/India", 
                       "train/Japan")
}

_PREDEFINED_SPLITS_GRC_MD["rdd2020"] ={
    "rdd2020_val"  : ( "lval/Czech", 
                       "lval/India", 
                       "lval/Japan"),
    "rdd2020_test" : ( "ltest/Czech", 
                       "ltest/India", 
                       "ltest/Japan"),
    "rdd2020_train": ( "ltrain/Czech", 
                       "ltrain/India", 
                       "ltrain/Japan")
}

_PREDEFINED_SPLITS_GRC_MD["rdd2020_test"] = {
    "rdd2020_test1": ( "test1/Czech", 
                       "test1/India", 
                       "test1/Japan")
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

# Configuration setup
def set_configuration():
    cfg = get_cfg()
    # We retry random cropping until no single category in semantic segmentation GT occupies more than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    cfg.SOLVER.POLY_LR_POWER = 0.9                          # Used for `poly` learning rate schedule.
    cfg.SOLVER.POLY_LR_CONSTANT_ENDING = 0.0
    cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE = "hard_pixel_mining"  # Loss type, choose from `cross_entropy`, `hard_pixel_mining`.
    cfg.MODEL.SEM_SEG_HEAD.PROJECT_FEATURES = ["res2"]      # DeepLab settings
    cfg.MODEL.SEM_SEG_HEAD.PROJECT_CHANNELS = [48]
    cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS = 256
    cfg.MODEL.SEM_SEG_HEAD.ASPP_DILATIONS = [6, 12, 18]
    cfg.MODEL.SEM_SEG_HEAD.ASPP_DROPOUT = 0.1
    cfg.MODEL.RESNETS.RES4_DILATION = 1         # Backbone new configs
    cfg.MODEL.RESNETS.RES5_MULTI_GRID = [1, 2, 4]
    cfg.MODEL.RESNETS.STEM_TYPE = "deeplab"     # ResNet stem type from: `basic`, `deeplab`
    cfg.merge_from_file("./model/configs/Cityscapes-SemanticSegmentation/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml")
    cfg.OUTPUT_DIR            = "./model"
    cfg.MODEL.WEIGHTS         = os.path.join(cfg.OUTPUT_DIR, "model_segm_e90k_class19_31Aug-deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.pth")
    cfg.MODEL.DEVICE          = "cuda"
    cfg.DATASETS.TEST         = ("cityscapes_fine_sem_seg_test", )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold for this model
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg 

cfg = set_configuration()
metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
cpu_device = torch.device("cpu")
predictor = DefaultPredictor(cfg)

def cv2_imshow(im, time_out=50000):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 1024)
    cv2.imshow(WINDOW_NAME, im)
    if cv2.waitKey(time_out) == 27:
        cv2.destroyAllWindows() # esc to quit
        print("Closing the view")


def predict_visualize(full_image_path):
    im_input = cv2.imread(full_image_path)
    #print("Image:  ", full_image_path, ", shape: ", im_input.shape)
    predictions = predictor(im_input)
    # Convert image from OpenCV BGR format to Matplotlib RGB format.
    image = im_input[:, :, ::-1]
    visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)
    if "sem_seg" in predictions:
        vis_output = visualizer.draw_sem_seg(
            predictions["sem_seg"].argmax(dim=0).to(cpu_device)
        )
    if "instances" in predictions:
        instances = predictions["instances"].to(cpu_device)
        vis_output = visualizer.draw_instance_predictions(predictions=instances)
    return predictions, im_input, vis_output

""" Process segmentation mask """
def process_pred_masks(predictions):
    # flatten the output to a single mask layer
    sem_seg = predictions["sem_seg"].argmax(dim=0).to(cpu_device).numpy()
    label_text = "road" 
    label_id = metadata.stuff_classes.index(label_text)
    labels, areas = np.unique(sem_seg, return_counts=True)
    #print("Road  : ", sem_seg.shape)
    #print("Labels: ", labels, areas)
    binary_mask = (sem_seg == label_id).astype(np.uint8)
    (im_w, im_h) = binary_mask.shape
    #print("Mask  : ", im_w, im_h)
    contours, hierarchy = cv2.findContours(binary_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # Take the N biggest road contours and merge them
    cntsSorted = sorted(contours, key=lambda x: -cv2.contourArea(x)) 
    selected_contours, bound_rect = None, (0, 0, im_w, im_h)
    for n, contour in enumerate(cntsSorted):
        if cv2.contourArea(contour) > 500:
            #print(n,") Contours Sorted: ", cv2.contourArea(contour))
            if selected_contours is None:
                selected_contours = contour
            else:
                np.append(selected_contours, contour, axis=0)
    if selected_contours is not None:
        #print("Min XY (",np.min(selected_contours[:, :, 0]), ", ", np.min(selected_contours[:, :, 1]), ")")
        #print("Max XY (",np.max(selected_contours[:, :, 0]), ", ", np.max(selected_contours[:, :, 1]), ")")
        y_crop = int(np.min(selected_contours[:, :, 1], axis=0)) # adjusting the bbox slightly
        (x,y,w,h)  = cv2.boundingRect(selected_contours)
        bound_rect = (x, y_crop, w, h - (y-y_crop))
        binary_mask[y_crop:, x:x+w] = 1  # Open up the rect view on the road part of the image
    return binary_mask, selected_contours, bound_rect

""" Load & Write on image if annotation exists """
def check_annotations(image_infile, ann_image):
    filename = os.path.basename(image_infile)
    dirname = os.path.dirname(image_infile) 
    ann_path = os.path.join(dirname, "../annotations/xmls/")
    annos = []
    ann_file = filename.split(".")[0] + ".xml"
    record = {}    
    if os.path.isfile(os.path.join(ann_path, ann_file)):
        print("\tAnn: ",os.path.join(ann_path, ann_file))
        thing_names = [k["name"] for k in RDD_DAMAGE_CATEGORIES]
        infile_xml = open(os.path.join(ann_path, ann_file))
        tree = ElementTree.parse(infile_xml)
        root = tree.getroot()
        img_height = int(root.find('size').find('height').text)
        img_width  = int(root.find('size').find('width').text)
        for obj in root.iter('object'):
            cls_name, xmlbox = obj.find('name').text, obj.find('bndbox')
            xmin, xmax = np.float(xmlbox.find('xmin').text), np.float(xmlbox.find('xmax').text)
            ymin, ymax = np.float(xmlbox.find('ymin').text), np.float(xmlbox.find('ymax').text)
            bbox = [xmin, ymin, xmax, ymax]       # (x0, y0, x1, y1)  -> (x0, y0, w, h) #bbox = BoxMode.convert(bbox, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS) 
            anno = {
                "category_id"   : thing_names.index(cls_name),
                "category_name" : cls_name,
                "bbox"          : bbox,
                "iscrowd"       : 0
            }
            annos.append(anno)
            cv2.rectangle(ann_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
            cv2.putText(ann_image, text=cls_name, org=(int(xmin), int(ymin-5)), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        record["image_name"] = filename
        record["file_name"] = image_infile
        record["height"] = img_height
        record["width"] = img_width
        record["annotations"] = annos
        print("\tAnnotation: ",record)
    return record, ann_image


""" Process image and preserve only the label==road segment in image """
def process_segment_images(image_infile, image_outpath, save_out=False):
    filename = os.path.basename(image_infile)
    predictions, im_input, vis_output = predict_visualize(image_infile)
    if isinstance(predictions["sem_seg"], torch.Tensor):
        binary_mask, contours, bound_box = process_pred_masks(predictions)
        #output_vis_image = vis_output.get_image()[:, :, ::-1]

        if contours is not None:
            masked_image = binary_mask.reshape(im_input.shape[0], im_input.shape[1], 1) * im_input
            cv2.drawContours(masked_image, [contours], -1, (0, 255, 0), 3)
            (x,y,w,h) = bound_box
            cv2.rectangle(masked_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            check_annotations(image_infile, masked_image)
            if save_out:
                cv2.imwrite(os.path.join(image_outpath, filename),  masked_image)
            else:
                cv2_imshow(masked_image)
        else:
            if save_out:
                cv2.imwrite(os.path.join(image_outpath, filename),  im_input)
            else:
                cv2_imshow(im_input)
            check_annotations(image_infile, im_input)
        print("\tSaving to ", os.path.join(image_outpath, filename))
    return predictions, vis_output


""" 
    python model_segm.py --input 
"""
if __name__ == "__main__":
    print("Metadata List: ", MetadataCatalog.list())
    #input_folders = _PREDEFINED_SPLITS_GRC_MD["rdd2020_source"]["rdd2020_train"]
    input_folders = ["train_short/Japan"]
    try:
        for folder in input_folders:
            print("\n----------- ", folder, "------------\n")
            image_filepath = os.path.join(ROADDAMAGE_DATASET, "train/Japan/images", "Japan_000000.jpg")    # single image
            #image_filepath = os.path.join(ROADDAMAGE_DATASET, folder, "images")          # in  directory
            image_outpath  = os.path.join(image_filepath, "../images_segm")              # out directory
            #os.makedirs(image_outpath, exist_ok=True)
            if os.path.isdir(image_filepath):
                for id, imfile in enumerate(sorted(glob.glob(os.path.join(image_filepath, '*.jpg')))): # assuming jpg images
                    start_time = time.time()
                    print("\n")
                    print("{}.)\tLoading Images: {}".format(id, imfile))
                    predictions, im_input, vis_output = process_segment_images(imfile, image_outpath, save_out=False)
                    print("    \t{}: {} in {:.2f}s".format(imfile, "detected {} instances".format(len(predictions["sem_seg"])), time.time() - start_time))
                    #cv2_imshow(vis_output.get_image()[:, :, ::-1])
            else:
                start_time = time.time()
                predictions, im_input, vis_output = process_segment_images(image_filepath, image_outpath, save_out=False)
                print("{}.)\t{}: {} in {:.2f}s".format(id, image_filepath, "detected {} instances".format(len(predictions["sem_seg"])), time.time() - start_time))
                cv2_imshow(vis_output.get_image()[:, :, ::-1])
    except Exception as e:
        raise Exception('Error loading data from %s: %s\n' % (input_folders, e))
