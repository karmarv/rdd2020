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

# import some common libraries
import numpy as np
import os, json, cv2, random, time
import matplotlib.pyplot as plt
from copy import deepcopy

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

## Train detectron2
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data import transforms as T
from detectron2.modeling import build_model
from detectron2.data import detection_utils
from detectron2.data.build import (
    build_detection_test_loader,
    build_detection_train_loader,
)

import data_rdd

#def cv2_imshow(im):
#    plt.figure(figsize=(8,8))
#    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

def cv2_imshow(im, time_out=30000):
    WINDOW_NAME = "RDD"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 1024)
    cv2.imshow(WINDOW_NAME, im)
    if cv2.waitKey(time_out) == 27:
        cv2.destroyAllWindows() # esc to quit
        print("Closing the view")

# Setup COCO style dataset
DatasetCatalog.clear()
for dataset_name, splits_per_dataset in data_rdd._PREDEFINED_SPLITS_GRC_MD["rdd2020"].items():
    inst_key = f"{dataset_name}"
    d = dataset_name.split("_")[1]
    print("[",d,"]\t",dataset_name, "\t", splits_per_dataset)
    DatasetCatalog.register(inst_key, lambda path=data_rdd.ROADDAMAGE_DATASET, d=deepcopy(splits_per_dataset) : data_rdd.load_images_ann_dicts(path, d))
    meta = data_rdd.get_rdd_coco_instances_meta()
    MetadataCatalog.get(inst_key).set(evaluator_type="coco", basepath=data_rdd.ROADDAMAGE_DATASET, splits_per_dataset=deepcopy(splits_per_dataset), **meta) 

class MyColorAugmentation(T.Augmentation):
    def get_transform(self, image):
        r = np.random.rand(2)
        return T.ColorTransform(lambda x: x * r[0] + r[1] * 10)

class MyCustomResize(T.Augmentation):
    def get_transform(self, image):
        old_h, old_w = image.shape[:2]
        new_h, new_w = int(old_h * np.random.rand()), int(old_w * 1.5)
        return T.ResizeTransform(old_h, old_w, new_h, new_w)



# Trainer Class
class RDDTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs(os.path.join(cfg.OUTPUT_DIR,"coco_eval"), exist_ok=True)
            output_folder=os.path.join(cfg.OUTPUT_DIR,"coco_eval")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:
        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        return model

    @classmethod
    def build_mapper(cls, cfg, is_train=True):
        augs = detection_utils.build_augmentation(cfg, is_train)
        #if cfg.INPUT.CROP.ENABLED and is_train:
        #    augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))

        # Define a sequence of augmentations: TODO
        augs.append(T.RandomBrightness(0.9, 1.1))
        augs.append(MyColorAugmentation())
        # augs.append(T.RandomRotation([5,10,15,20,25,30], expand=True, center=None))
        # T.RandomCrop("absolute", (640, 640)),
        # MyCustomResize()
        # type: T.Augmentation
        return DatasetMapper(cfg, is_train=is_train, augmentations=augs)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """        
        # TODO : Augmentation is not working
        mapr = cls.build_mapper(cfg, is_train=True) 
        return build_detection_train_loader(cfg, mapper=None)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        mapr = cls.build_mapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=None)

# Configuration
cfg = get_cfg()
# Faster RCNN
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS         = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo

#cfg.MODEL.WEIGHTS         = "./output/run_d2_frcnn-fpn-combovt_b640_v0_extd/model_final_10k.pth"
# RetinaNet
#cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
#cfg.MODEL.WEIGHTS         = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")

cfg.DATASETS.TRAIN        = ("rdd2020_train",)
cfg.DATASETS.TEST         = ("rdd2020_val", )
#cfg.OUTPUT_DIR            = "./output/run_d2_frcnn-fpn-combovt_b640_v0/"
cfg.OUTPUT_DIR            = "../rdd2020_model_repository/det2-fasterrcnn-fpn/run_d2_frcnn-fpn-combovt_b640_v0/"
cfg.MODEL.DEVICE          = "cuda"
cfg.DATALOADER.NUM_WORKERS= 8
cfg.SOLVER.IMS_PER_BATCH  = 8
cfg.SOLVER.BASE_LR        = 0.01        # Pick a good LR
cfg.SOLVER.WARMUP_ITERS   = 1000 
cfg.SOLVER.MAX_ITER       = 30000       # You may need to train longer for a practical dataset
cfg.SOLVER.STEPS          = (23000, 25000, 26000)
cfg.SOLVER.GAMMA          = 0.05
cfg.TEST.EVAL_PERIOD      = 1000

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE  = 640   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES           = len(data_rdd.RDD_DAMAGE_CATEGORIES)  # only has few damage classes
cfg.MODEL.RETINANET.NUM_CLASSES           = len(data_rdd.RDD_DAMAGE_CATEGORIES)
cfg.SOLVER.CHECKPOINT_PERIOD              = 1000
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
setup_logger(output=cfg.OUTPUT_DIR)



# Variables for processing 
rdd2020_metadata = MetadataCatalog.get("rdd2020_val")
print("\nRDD2020 Metadata: ", rdd2020_metadata,"\n")
trainer = RDDTrainer(cfg)

def visualize_results(visualize_flag=True):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold for this model
    cfg.DATASETS.TEST = ("rdd2020_val",)
    #splits_per_submission_dataset = ( "test1/India", "test1/Japan", "test1/Czech")
    #splits_per_submission_dataset = ( "test2/India", "test2/Japan", "test2/Czech")
    splits_per_submission_dataset = ( "ltest/India", "ltest/Japan", "ltest/Czech")
    dataset_test_submission_dicts = data_rdd.load_images_ann_dicts(data_rdd.ROADDAMAGE_DATASET, splits_per_submission_dataset)
    predictor = DefaultPredictor(cfg)
    #rdd2020_metadata = MetadataCatalog.get("rdd2020_val")
    for idx, d in enumerate(dataset_test_submission_dicts):
        print("Visualize this result: ", d["file_name"])
        im = cv2.imread(d["file_name"])
        start = time.time()
        outputs = predictor(im)
        end = time.time()
        print(idx, ".) ", outputs["instances"].pred_classes)
        print("     ", outputs["instances"].scores, ", Time(sec): ", (end - start))
        v = Visualizer(im[:, :, ::-1],
                    metadata=rdd2020_metadata, 
                    scale=0.5
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2_imshow(out.get_image()[:, :, ::-1])
    return        

def evaluate_results():
    # Inference & evaluation using the trained model
    # Now, let's run inference with the trained model on the validation dataset. 
    # First, let's create a predictor using the model we just trained:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold for this model
    cfg.DATASETS.TEST = ("rdd2020_val",)
    predictor = DefaultPredictor(cfg)
    trainer.resume_or_load(resume=False)

    # Then, we randomly select several samples to visualize the prediction results.
    from detectron2.utils.visualizer import ColorMode
    from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, inference_on_dataset
    from detectron2.data import build_detection_test_loader

    evaluator = COCOEvaluator("rdd2020_val", cfg, False, "coco_eval")
    val_loader = build_detection_test_loader(cfg, "rdd2020_val")
    eval_results = inference_on_dataset(trainer.model, val_loader, DatasetEvaluators([evaluator]))
    # another equivalent way is to use trainer.test
    print(eval_results)

#JUST_EVALUATE = False     # False means Train
JUST_EVALUATE = True

# Decision to Train or just evaluate
if not JUST_EVALUATE:
    # Train
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Look at training curves in tensorboard:
    # %load_ext tensorboard
    # %tensorboard --logdir output/run_rdd/
else:
    #evaluate_results()
    visualize_results()

# Empty the GPU Memory 
torch.cuda.empty_cache()
