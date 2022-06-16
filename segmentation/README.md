## Evaluating semantic segmentation on the ADE20K dataset

Step 1. Prepare ADE20K dataset

The dataset can be downloaded at 
`http://groups.csail.mit.edu/vision/datasets/ADE20K/toolkit/index_ade20k.pkl`

or following instruction of `https://github.com/CSAILVision/ADE20K`

Step 2. Install mmsegmentation

`git clone https://github.com/open-mmlab/mmsegmentation.git`

Step 3. Convert your model

`python tools/model_converters/vit2mmseg.py /path/to/model_dir /path/to/saving_dir`

Step 4. Fine-tune on the ADE20K dataset using /segmentation/configs/semfpn_vit-s16_512x512_40k_ade20k.py

`tools/dist_train.sh configs/selfpatch/semfpn_vit-s16_512x512_40k_ade20k.py [number of gpu] --work-dir /path/to/saving_dir --seed 0 --deterministic --options model.pretrained=/path/to/model_dir`

## Acknowledgement
This code is built using the : <a href=https://github.com/open-mmlab/mmsegmentation>mmsegmentation</a> libray. The optimization hyperarameters are adopted from <a href=https://github.com/facebookresearch/xcit>XCiT</a> and <a href=https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation>Swin</a>
