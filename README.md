# EFPN

This repo and account are part of the supplementary material for an anonymous submission, so we do not accept any PR or issue currently. 

## Requirements


This code has been developed under **Python3.6**, **PyTorch 0.3.1** and **CUDA 9.0** on Ubuntu 16.04.

The python packages can be installed with

```shell
pip3 install -r requirements.txt
```

## Compilation

Compile the CUDA code:

```shell
cd lib  # please change to this directory
sh make.sh
```

`CUDA_PATH` defaults to `/usr/loca/cuda`. If you want to use a CUDA library on different path, change this [line](https://github.com/anony899/EFPN-pytorch/blob/e099ccb378f3ed2439542d62178820e2db451b1c/lib/make.sh#L3) in `lib/make.sh` accordingly.

## Pretrainded Models

We provide pretrainded models on Tsinghua-Tencent 100K and COCO for inference. The models are both based on Faster R-CNN with ResNeXt-101-EFPN.


Download our trained models from  [GoogleDrive](https://drive.google.com/open?id=1icFx2uxjNr0SbRE5yHzj_3v44hGg-sW7) or [BaiduYun](https://pan.baidu.com/s/1DvIOZJ80dKsEbugJ2Ztctg)(Key:l24p), and put them into  `{repo_root}/checkpoints`.

## Inference 

At present, the code only supports **single GPU** inference. You can specify the GPU id on [line](https://github.com/anony899/EFPN-pytorch/blob/e099ccb378f3ed2439542d62178820e2db451b1c/tools/infer_simple.py#L122) in `tools/infer_simple.py` (default set to GPU 0).

To visualize examples of Tsinghua-Tencent 100K, run with:

```shell
python tools/infer_simple.py --dataset tt100k --cfg configs/EFPN_X-101_TT100K.yaml --load_ckpt checkpoints/EFPN_X101_TT100K.pth --image_dir examples/tt100k --output_dir examples/res_tt100k
```

To visualize examples of Tsinghua-Tencent 100K, run with:

```shell 
python tools/infer_simple.py --dataset coco2017 --cfg configs/EFPN_X-101_COCO.yaml --load_ckpt checkpoints/EFPN_X101_COCO.pth --image_dir examples/coco --output_dir examples/res_coco
```

and the detection results will be saved in `examples/res_tt100k` and `examples/res_coco`.

## Examples

We provide several detection examples of EFPN in `./examples`. You can also directly view them.

## Training

TBA




