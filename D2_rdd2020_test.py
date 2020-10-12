# install dependencies: 
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
# install detectron2: (colab has CUDA 10.1 + torch 1.6)
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
assert torch.__version__.startswith("1.6")

# Some basic setup, Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators

import data_rdd


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
#cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
#cfg.OUTPUT_DIR            = "../rdd2020_model_repository/det2-fasterrcnn-fpn/run_d2_frcnn-fpn-combovt_b640_v3/"
#cfg.OUTPUT_DIR            = "./output/run_d2_frcnn-fpn-combovt_b640_v0/"
cfg.OUTPUT_DIR            = "./output/run_d2_frcnn-fpn-combovt_b640_v0_extd/"

cfg.MODEL.DEVICE          = "cuda"
cfg.MODEL.ROI_HEADS.NUM_CLASSES           = len(data_rdd.RDD_DAMAGE_CATEGORIES)  # only has one class (ballon)
cfg.MODEL.RETINANET.NUM_CLASSES           = len(data_rdd.RDD_DAMAGE_CATEGORIES)
# ## Inference & evaluation using the trained model
# Now, let's run inference with the trained model on the balloon validation dataset. First, let's create a predictor using the model we just trained:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold for this model
print("Evaluating Model:", cfg.MODEL.WEIGHTS)
predictor = DefaultPredictor(cfg)

# Generate RDD2020 Submission dataset
splits_per_submission_dataset = ( "test1/India", "test1/Japan", "test1/Czech")
#splits_per_submission_dataset = ( "test2/India", "test2/Japan", "test2/Czech")
dataset_test_submission_dicts = data_rdd.load_images_ann_dicts(data_rdd.ROADDAMAGE_DATASET, splits_per_submission_dataset)


# **Submission**: https://rdd2020.sekilab.global/submissions/
# 
# 
# For each image in the test dataset, your algorithm needs to predict a list of labels, and the corresponding bounding boxes.
# 
# The output is expected to contain the following two columns:
# - ImageId: the id of the test image, for example, Adachi_test_00000001
# - PredictionString: the prediction string should be a space-delimited of 5 integers. For example, 2 240 170 260 240 means it's label 2, with a bounding box of coordinates (x_min, y_min, x_max, y_max). We accept up to 5 predictions. For example, if you submit 3 42 24 170 186 1 292 28 430 198 4 168 24 292 190 5 299 238 443 374 2 160 195 294 357 6 3 214 135 356 which contains 6 bounding boxes, we will only take the first 5 into consideration.

# ----
# 
# > Damage Types = ["D00", "D01", "D10", "D11", "D20", "D40", "D43", "D44", "D50", "D0w0"]
# - Classes of interest {D00: Longitudinal Crack, D10: Transverse Crack, D20: Aligator Crack, D40: Pothole}
# 
# 

map_classes_superids = []
for k in data_rdd.RDD_DAMAGE_CATEGORIES:
    print(k["id"], "\t", k["name"], "\t", k["submission_superid"], "\t", k["description"] )
    map_classes_superids.append(k["submission_superid"])


# ## TODO
# - [ ] Take the best 5 scoring/confidence results for submission
# 
# ## Assumed !!
# - Skipping results which are more than 5 in count
# - Mapping the 10 classes to the 4 submission required classes !!


# Generate submission format result for RDD2020
def format_submission_result(image_meta, predictions):
    boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
    scores = predictions.scores.numpy() if predictions.has("scores") else None
    classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
    formatted_result = ["{}".format(image_meta["image_name"])]
    if classes is not None:
        score_dict = {}
        for i, (clss, scr, bbx) in enumerate(zip(classes, scores, boxes)):
            (x_min, y_min, x_max, y_max) = bbx
            # class_submission_id, x_min, y_min, x_max, y_max
            out_str = "{0} {1} {2} {3} {4} ".format(map_classes_superids[int(clss)], int(x_min), int(y_min), int(x_max), int(y_max))
            score_dict[scr] = out_str
        result_item = ""
        for key in sorted(score_dict.keys(), reverse=True):
            result_item += score_dict[key]
        formatted_result.append(result_item)
    return formatted_result


def generate_results():
    results = []
    for idx, d in tqdm(enumerate(dataset_test_submission_dicts)):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        formatted_result = format_submission_result(d, outputs["instances"].to("cpu"))
        if formatted_result is not None:
            results.append("{},{}".format(formatted_result[0], formatted_result[1]))
    return results

def write_results_to_file():
    with open(os.path.join(cfg.OUTPUT_DIR, 'hal_submission_rdd2020_t1exp_exttt10k.txt'), 'w') as f:
      f.writelines("%s\n" % line for line in results)


print("------------------- Test -------------------")
results = generate_results()
tqdm._instances.clear()
print("------------------- Write -------------------")
write_results_to_file()

