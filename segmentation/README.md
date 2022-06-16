# Patch-level Representation Learning for Self-supervised Vision Transformers

Our code base is built upon the prior work, DINO (https://github.com/facebookresearch/dino).

Requirements: torch==1.7.0 torchvision==0.8.1

## Pretraining step for 200 traning epochs with a batch size of 1024
python -m torch.distributed.launch --nproc_per_node=8 main_selfpatch.py --arch vit_small --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir --epoch 200 --local_crops_number 8 --patch_size 16 --batch_size_per_gpu 128 --out_dim_selfpatch 4096 --k_num 4

## Evaluating object detection and instance segmentation on the COCO dataset
Step 1. Prepare COCO dataset
The dataset can be downloaded at https://cocodataset.org/#download

Step 2. Install mmdetection
git clone https://github.com/open-mmlab/mmdetection.git

Step 3. Fine-tune on the COCO dataset using /detection/configs/selfpatch/mask_rcnn_vit_small_12_p16_1x_coco.py
cd detection
tools/dist_train.sh configs/selfpatch/mask_rcnn_vit_small_12_p16_1x_coco.py [number of gpu] --work-dir /path/to/saving_dir --seed 0 --deterministic --options model.pretrained=/path/to/model_dir

## Evaluating semantic segmentation on the ADE20K dataset
Step 1. Prepare ADE20K dataset
The dataset can be downloaded at http://groups.csail.mit.edu/vision/datasets/ADE20K/toolkit/index_ade20k.pkl
or following instruction of https://github.com/CSAILVision/ADE20K

Step 2. Install mmsegmentation
git clone https://github.com/open-mmlab/mmsegmentation.git

Step 3. Convert your model
cd segmentation
python tools/model_converters/vit2mmseg.py /path/to/model_dir /path/to/saving_dir

Step 4. Fine-tune on the ADE20K dataset using /segmentation/configs/semfpn_vit-s16_512x512_40k_ade20k.py
tools/dist_train.sh configs/selfpatch/semfpn_vit-s16_512x512_40k_ade20k.py [number of gpu] --work-dir /path/to/saving_dir --seed 0 --deterministic --options model.pretrained=/path/to/model_dir

## Evaluating video object segmentation on the DAVIS 2017 dataset
Step 1. Prepare DAVIS 2017 data
cd $HOME
git clone https://github.com/davisvideochallenge/davis-2017 && cd davis-2017
./data/get_davis.sh

Step 2. Run Video object segmentation
python eval_video_segmentation.py --data_path /path/to/davis-2017/DAVIS/ --output_dir /path/to/saving_dir --pretrained_weights /path/to/model_dir --arch vit_small --patch_size 16 --n 1

Step 3. Evaluate the obtained segmentation
git clone https://github.com/davisvideochallenge/davis2017-evaluation $HOME/davis2017-evaluation
python /path/to/davis2017-evaluation/evaluation_method.py --task semi-supervised --davis_path /path/to/davis-2017/DAVIS --results_path /path/to/saving_dir