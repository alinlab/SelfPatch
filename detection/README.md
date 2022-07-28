## Evaluating object detection and instance segmentation on the COCO dataset
Step 1. Prepare COCO dataset

The dataset can be downloaded at `https://cocodataset.org/#download`

Step 2. Install mmdetection

```
git clone https://github.com/open-mmlab/mmdetection.git
```

Step 3. Fine-tune on the COCO dataset 

```
tools/dist_train.sh configs/selfpatch/mask_rcnn_vit_small_12_p16_1x_coco.py [number of gpu] --work-dir /path/to/saving_dir --seed 0 --deterministic --options model.pretrained=/path/to/model_dir
```

## Pretrained weights on MS-COCO
You can download the weights of the fine-tuned models on object detection and instance segmentation tasks. All models are fine-tuned with `Mask R-CNN`. 

| backbone | arch | bbox mAP | mask mAP | checkpoint |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| DINO | ViT-S/16 + Mask R-CNN | 40.8 | 37.3 | <a href="https://drive.google.com/file/d/1yFTycxtmWEQEQyLT6l_SYV0YQ-5Z1BvN/view?usp=sharing">download</a> |
| DINO + SelfPatch | ViT-S/16 + Mask R-CNN | 42.1 | 38.5 | <a href="https://drive.google.com/file/d/1q45LplADkDdiCqzhWOkgyUXivHJrI7eT/view?usp=sharing">download</a> |

## Acknowledgement
This code is built using the <a href=https://github.com/open-mmlab/mmdetection>mmdetection</a> libray.
