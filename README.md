# RDD 2020
Road Damage Detection Challenge (IEEE Big Data Cup 2020)

*Google Docs*: [Task/Plan Sheet](https://docs.google.com/document/d/1bOCSYn8rKUJLYNTRSNDRTYRlqz1KzSykh-hOJX33hwQ/edit?usp=sharing)

##### 1.) [Detectron2 Notebook](D2_rdd2020.ipynb)
- [ ] 

##### 2.) [MMDetection Notebook](MMDet_rdd2020.ipynb)
- [ ] 

--------------------------------------------------------------------------------


#### A.) Codebase [TODO]
- [ ] prep_data.py : --input rdd_data_folder_path --output rdd_split_traindata
- [ ] train_net.py : --input rdd_split_traindata  --output rdd_model&eval
- [ ] test_net.py  : --input rdd_testdata rdd_model --output eval_submission.tsv
- [ ] streamlit.py : --input rdd_testdata rdd_model --output visualize_predictions


--------------------------------------------------------------------------------


#### B.) Plan of attack [TODO]
- [ ] 1.) Create a basic toolbox using 
    - [ ] [Detectron2](https://detectron2.readthedocs.io/) 
    - [ ] [MMDetection](https://github.com/open-mmlab/mmdetection)
    - [ ] [YoloV5](https://github.com/ultralytics/yolov5)
- [ ] 2.) Refer to old dataset/model for augmentation/init at https://github.com/sekilab/RoadDamageDetector
- [ ] 3.) List out strategies for fine-tuning from old dataset to latest -> 2020
- [ ] 4.) Previous challengers
    - [ ] a. Open Images 2019 - Object Detection (https://github.com/Sense-X/TSD)
- [ ] 5.) Feature Engineering
    - [ ] a. Attend to the road bounds for all detection purposes
    - [ ] b. Class Balance the dataset (RD2020 has a huge data imbalance problem)
    - [ ] c. Ensemble models 
    - [ ] d. Ensemble boxed for detection https://www.kaggle.com/c/open-images-2019-object-detection/discussion/115086
    - [ ] e. GAN and VAE for feature engineering
- [ ] 6.) Categorically train models based on geography (czech, Japan and India). Produce inference on those three individually and submit the combined CSV
- [ ] 7.) Run AutoML to identify the best Hyperparam/config
- [ ] 8.) Use VoVNet backend. an upgrade from ResNet. https://github.com/youngwanLEE/vovnet-detectron2
- [ ] 9.) Object Detection with Transformers https://github.com/facebookresearch/detr



--------------------------------------------------------------------------------