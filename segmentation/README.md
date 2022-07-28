## Evaluating semantic segmentation on the ADE20K dataset

Step 1. Prepare ADE20K dataset

The dataset can be downloaded at 
`http://groups.csail.mit.edu/vision/datasets/ADE20K/toolkit/index_ade20k.pkl`

or following instruction of `https://github.com/CSAILVision/ADE20K`

Step 2. Install mmsegmentation

```
git clone https://github.com/open-mmlab/mmsegmentation.git
```

Step 3. Convert your model

```
python tools/model_converters/vit2mmseg.py /path/to/model_dir /path/to/saving_dir
```

Step 4. Fine-tune on the ADE20K dataset

```
tools/dist_train.sh configs/selfpatch/semfpn_vit-s16_512x512_40k_ade20k.py [number of gpu] --work-dir /path/to/saving_dir --seed 0 --deterministic --options model.pretrained=/path/to/model_dir
```

## Pretrained weights on ADE20K
You can download the weights of the fine-tuned models on semantic segmentation task. We provide fine-tuned models with `Semantic FPN` (40k iterations) and `UperNet` (160k iterations). 

| backbone | arch | iterations | mIoU | checkpoint |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| DINO | ViT-S/16 + Semantic FPN | 40k | 38.3 | <a href="https://drive.google.com/file/d/1SdbNT5d5d5JQ8IYodpMvuF6JKtp_ovm6/view?usp=sharing">download</a> |
| DINO + SelfPatch | ViT-S/16 + Semantic FPN | 40k | 41.2 | <a href="https://drive.google.com/file/d/1il-K4ual9VRW-yDC92eNyZrtjc6KmZK6/view?usp=sharing">download</a> |
| DINO | ViT-S/16 + UperNet | 160k | 42.3 | <a href="https://drive.google.com/file/d/1vl2dhglveKK_1rmMi8XQEr8z3_zlecfO/view?usp=sharing">download</a> |
| DINO + SelfPatch | ViT-S/16 + UperNet | 160k | 43.2 | <a href="https://drive.google.com/file/d/1JoXtIsJh2RxEOrqqiNSsQ_oc8uicgOIS/view?usp=sharing">download</a> |

## Acknowledgement
This code is built using the <a href=https://github.com/open-mmlab/mmsegmentation>mmsegmentation</a> libray. The optimization hyperarameters are adopted from <a href=https://github.com/facebookresearch/xcit>XCiT</a>.
