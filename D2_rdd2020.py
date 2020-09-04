#!/usr/bin/env python
# coding: utf-8

# # Detectron2 for RDD2020
# # Install detectron2


# install dependencies: 
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
# install detectron2: (colab has CUDA 10.1 + torch 1.6)
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
assert torch.__version__.startswith("1.6")


# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
from copy import deepcopy

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import data_rdd

def cv2_imshow(im):
    plt.figure(figsize=(8,8))
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

# Setup COCO style dataset
DatasetCatalog.clear()
for dataset_name, splits_per_dataset in data_rdd._PREDEFINED_SPLITS_GRC_MD["rdd2020"].items():
    inst_key = f"{dataset_name}"
    d = dataset_name.split("_")[1]
    print("[",d,"]\t",dataset_name, "\t", splits_per_dataset)
    DatasetCatalog.register(inst_key, lambda path=data_rdd.ROADDAMAGE_DATASET, d=deepcopy(splits_per_dataset) : data_rdd.load_images_ann_dicts(path, d))
    meta = data_rdd.get_rdd_coco_instances_meta()
    MetadataCatalog.get(inst_key).set(evaluator_type="coco", basepath=data_rdd.ROADDAMAGE_DATASET, splits_per_dataset=deepcopy(splits_per_dataset), **meta) 

rdd2020_metadata = MetadataCatalog.get("rdd2020_val")

# ## Train!
# 
# Now, let's fine-tune a COCO-pretrained R50-FPN Mask R-CNN model on the RDD dataset.
# 

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators

class RDDTrainer(DefaultTrainer):
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
      os.makedirs(os.path.join(cfg.OUTPUT_DIR,"coco_eval"), exist_ok=True)
      output_folder=os.path.join(cfg.OUTPUT_DIR,"coco_eval")
    return COCOEvaluator(dataset_name, cfg, False, output_folder)


# Configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS         = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.DATASETS.TRAIN        = ("rdd2020_train",)
cfg.DATASETS.TEST         = ("rdd2020_val", )
cfg.OUTPUT_DIR            = "./output/run_rdd/"
cfg.MODEL.DEVICE          = "cuda"
cfg.DATALOADER.NUM_WORKERS= 8
cfg.SOLVER.IMS_PER_BATCH  = 8
cfg.SOLVER.BASE_LR        = 0.001      # Pick a good LR
cfg.SOLVER.WARMUP_ITERS   = 100 
cfg.SOLVER.MAX_ITER       = 27000       # You may need to train longer for a practical dataset
cfg.SOLVER.STEPS          = (220000, 250000)
cfg.SOLVER.GAMMA          = 0.05
cfg.TEST.EVAL_PERIOD      = 1000

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE  = 64   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES           = len(data_rdd.RDD_DAMAGE_CATEGORIES)  # only has one class (ballon)
cfg.SOLVER.CHECKPOINT_PERIOD              = 500
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Train
trainer = RDDTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

# Look at training curves in tensorboard:
# %load_ext tensorboard
# %tensorboard --logdir output/run_rdd/

# ## Inference & evaluation using the trained model
# Now, let's run inference with the trained model on the balloon validation dataset. First, let's create a predictor using the model we just trained:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold for this model
cfg.DATASETS.TEST = ("rdd2020_val",)
predictor = DefaultPredictor(cfg)


# Then, we randomly select several samples to visualize the prediction results.
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, inference_on_dataset
from detectron2.data import build_detection_test_loader

evaluator = COCOEvaluator("rdd2020_val", cfg, False, "coco_eval")
val_loader = build_detection_test_loader(cfg, "rdd2020_val")
eval_results = inference_on_dataset(trainer.model, val_loader, DatasetEvaluators([evaluator]))
# another equivalent way is to use trainer.test
print(eval_results)

# Empty the GPU Memory 
torch.cuda.empty_cache()

