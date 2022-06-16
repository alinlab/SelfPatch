## Evaluating object detection and instance segmentation on the COCO dataset
Step 1. Prepare COCO dataset

The dataset can be downloaded at `https://cocodataset.org/#download`

Step 2. Install mmdetection

`git clone https://github.com/open-mmlab/mmdetection.git`

Step 3. Fine-tune on the COCO dataset 

`tools/dist_train.sh configs/selfpatch/mask_rcnn_vit_small_12_p16_1x_coco.py [number of gpu] --work-dir /path/to/saving_dir --seed 0 --deterministic --options model.pretrained=/path/to/model_dir`

## Acknowledgement
This code is built using the <a href=https://github.com/open-mmlab/mmdetection>mmdetection</a> libray.
