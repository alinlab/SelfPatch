# Patch-level Representation Learning for Self-supervised Vision Transformers (SelfPatch)

PyTorch implementation for <a href=https://arxiv.org/abs/2206.07990>"Patch-level Representation Learning for Self-supervised Vision Transformers"</a> (accepted Oral presentation in CVPR 2022)

<p align="center">
<img width="782" alt="thumbnail" src="https://user-images.githubusercontent.com/4075389/174249727-e1d4433f-93ad-4675-ac58-2b6740dc7ae4.png">
</p>

## Requirements
- `torch==1.7.0`
- `torchvision==0.8.1`

## Pretraining on ImageNet
```
python -m torch.distributed.launch --nproc_per_node=8 main_selfpatch.py --arch vit_small --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir --local_crops_number 8 --patch_size 16 --batch_size_per_gpu 128 --out_dim_selfpatch 4096 --k_num 4
```

## Pretrained weights on ImageNet
You can download the weights of the pretrained models on ImageNet. All models are trained on `ViT-S/16`. For detection and segmentation downstream tasks, please check <a href="https://github.com/alinlab/SelfPatch/tree/main/detection">SelfPatch/detection</a>, <a href="https://github.com/alinlab/SelfPatch/tree/main/segmentation">SelfPatch/segmentation</a>. 

| backbone  | arch | checkpoint |
| ------------- | ------------- | ------------- |
| DINO | ViT-S/16  | <a href="https://drive.google.com/file/d/1LDw2UBPq6Xf8xMUk0G3IOhi2ZBFT-Zsq/view?usp=sharing">download</a> (pretrained model from <a href="https://github.com/facebookresearch/vissl">VISSL</a>) |
| DINO + SelfPatch | ViT-S/16 | <a href="https://drive.google.com/file/d/19eeQrK-nl4B9ksFQ_QnewGHVKYDQF4Yw/view?usp=sharing">download</a> |

## Evaluating video object segmentation on the DAVIS 2017 dataset
Step 1. Prepare DAVIS 2017 data

```
cd $HOME
git clone https://github.com/davisvideochallenge/davis-2017
cd davis-2017
./data/get_davis.sh
```

Step 2. Run Video object segmentation

```
python eval_video_segmentation.py --data_path /path/to/davis-2017/DAVIS/ --output_dir /path/to/saving_dir --pretrained_weights /path/to/model_dir --arch vit_small --patch_size 16
```

Step 3. Evaluate the obtained segmentation

```
git clone https://github.com/davisvideochallenge/davis2017-evaluation 
$HOME/davis2017-evaluation
python /path/to/davis2017-evaluation/evaluation_method.py --task semi-supervised --davis_path /path/to/davis-2017/DAVIS --results_path /path/to/saving_dir
```

### Video object segmentation examples on the DAVIS 2017 dataset

Video (left), DINO (middle) and our SelfPatch (right)
<p align="center">    
<img width="100%" alt="img"  src="https://user-images.githubusercontent.com/4075389/181043309-ebc8a329-ea15-4768-ae54-cab6dcf0b98a.gif" />
</p>


## Acknowledgement
Our code base is built partly upon the packages: 
<a href="https://github.com/facebookresearch/dino">DINO</a>, <a href=https://github.com/open-mmlab/mmdetection>mmdetection</a>, <a href=https://github.com/open-mmlab/mmsegmentation>mmsegmentation</a> and <a href=https://github.com/facebookresearch/xcit>XCiT</a>

## Citation
If you use this code for your research, please cite our papers.
```
@InProceedings{Yun_2022_CVPR,
    author    = {Yun, Sukmin and Lee, Hankook and Kim, Jaehyung and Shin, Jinwoo},
    title     = {Patch-Level Representation Learning for Self-Supervised Vision Transformers},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {8354-8363}
}
```
