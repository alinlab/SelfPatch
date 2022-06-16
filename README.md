# Patch-level Representation Learning for Self-supervised Vision Transformers (SelfPatch)

PyTorch implementation for SelfPatch (accepted in CVPR, 2022)

## Requirements
Requirements: `torch==1.7.0` `torchvision==0.8.1`

## Pretraining step for 200 traning epochs with a batch size of 1024
`python -m torch.distributed.launch --nproc_per_node=8 main_selfpatch.py --arch vit_small --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir --epoch 200 --local_crops_number 8 --patch_size 16 --batch_size_per_gpu 128 --out_dim_selfpatch 4096 --k_num 4`


## Evaluating video object segmentation on the DAVIS 2017 dataset
Step 1. Prepare DAVIS 2017 data

```
cd $HOME
git clone https://github.com/davisvideochallenge/davis-2017
cd davis-2017
./data/get_davis.sh
```

Step 2. Run Video object segmentation

`python eval_video_segmentation.py --data_path /path/to/davis-2017/DAVIS/ --output_dir /path/to/saving_dir --pretrained_weights /path/to/model_dir --arch vit_small --patch_size 16`

Step 3. Evaluate the obtained segmentation

```
git clone https://github.com/davisvideochallenge/davis2017-evaluation 
$HOME/davis2017-evaluation
python /path/to/davis2017-evaluation/evaluation_method.py --task semi-supervised --davis_path /path/to/davis-2017/DAVIS --results_path /path/to/saving_dir
```

## Acknowledgement
Our code base is built partly upon the package, DINO (https://github.com/facebookresearch/dino), mmdetection (https://github.com/open-mmlab/mmdetection), mmsegmentation (https://github.com/open-mmlab/mmsegmentation) and XCiT (https://github.com/facebookresearch/xcit)

## Citation
