# RDD 2020
Road Damage Detection Challenge (IEEE Big Data Cup 2020)


##### 1.) [Detectron2 Notebook Sample](D2_rdd2020.ipynb)

--------------------------------------------------------------------------------

#### A.) Codebase
- [x] D2_rdd2020_test.py : Runs the test on dataset test1/* or test2/* when provided with config and model weights in the codebase. It also writes the submittable output file.
- [x] D2_rdd2020.py : Training codebase for Detectron2 models
- [x] data_rdd.py  : Data utility for loading and transforming the Road damage data to COCO format and create train-val-test split
- [x] requirements : use this file to install packages in a python virtual environment. Use command "pip install -r requirements"

--------------------------------------------------------------------------------

#### B.) Plan of attack
- [x] 1.) Create a basic toolbox using 
    - [x] [Detectron2](https://detectron2.readthedocs.io/) 
- [x] 2.) Refer to old dataset/model for augmentation/init at https://github.com/sekilab/RoadDamageDetector
- [x] Resnet 50, 101 and various Hyper parameters in Faster-R-CNN model
- [x] RetinaNet and YoloV5
- [x] Mix train & test dataset and fine tune
- [x] Mix train & val dataset and fine tune 

--------------------------------------------------------------------------------

#### C.) Steps to reproduce the results
- [x] 1.) Setup Data split for train/val/test using [data_rdd.py]
    - Add the train/ data in a folder called rdd2020/
    - Set the variable in data_rdd.py(line 10) based on the path of your dataset > DETECTRON2_DATASETS = "/media/rahul/Karmic/data"
    - Expect a soft link of train files in folder rdd2020/lval, rdd2020/ltrain and rdd2020/ltest with annotations
    - Manually Merge rdd2020/lval and rdd2020/ltrain to create rdd2020/lvaltrain dataset for training 
    - Current dataset in code is configured for training on 'rdd2020/lvaltrain' data

- [x] 2.) Configure the training codebase using [D2_rdd2020.py]. Two strategies are shown below,
    - Config For best Test2 score training
        - Update relevant line:130-154 with the following configurations
          ``` 
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.WEIGHTS         = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
            cfg.SOLVER.BASE_LR        = 0.01                
            cfg.SOLVER.MAX_ITER       = 30000               
            cfg.SOLVER.STEPS          = (23000, 25000, 26000)
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE  = 640
          ```
    - Config For best Test1 score training
        - Update relevant line:147-154 with the following configurations
          ``` 
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            cfg.MODEL.WEIGHTS         = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
            cfg.SOLVER.BASE_LR        = 0.015               
            cfg.SOLVER.MAX_ITER       = 30000               
            cfg.SOLVER.STEPS          = (25000, 28000)
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE  = 4096
          ```

- [x] 3. Train by running command 'python D2_rdd2020.py'
    - upon completing the max iterations a 'model_final.pth' will be dumped in respective output folder.

- [x] 4. **Test the dumped model_final.pth weights by using [D2_rdd2020_test.py]**
    - **Pre-trained model weights** download at [https://github.com/vishwakarmarhl/rdd2020/releases/tag/b0.1](https://github.com/vishwakarmarhl/rdd2020/releases/tag/b0.1)
    - Configure the line:32 for the model configuration based on the training strategy used in Step 2
    - Comment or uncomment line:49 (test1) or line:50 (test2) based on the test dataset you want to generate the submission file for.
    - run the test command 'python D2_rdd2020_test.py' to find a txt dump file that evaluates the test images using the trained model
    - use this txt dump file for submission or eval purposes
