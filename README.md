
<h2 align="center">
  <img src="assets/bench2drive.jpg" style="width: 100%; height: auto;">
</h2>
<h2 align="center">
Bench2DriveZoo
</h2>
<h2 align="center">
  <img src="assets/bench2drivezoo.png" style="width: 100%; height: auto;">
</h2>


# Introduction

- This repo contains the training, open-loop evaluation, and closed-loop evaluation code for [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [UniAD](https://github.com/OpenDriveLab/UniAD) , [VAD](https://github.com/hustvl/VAD) in [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive).
- We merge multiple dependencies of UniAD and VAD including mmcv, mmseg, mmdet, and mmdet3d (v0.17.1) into a single library. As a result, it could support latest pytorch and advanced frameworks like deepspeed for acceleration.
- Use "git checkout tcp/admlp" to obtain their corresponding training and evaluation code.


# Getting Started

- [Installation](docs/INSTALL.md)
- [Prepare Dataset](docs/INSTALL.md)
- [Train and Open-Loop Eval](docs/TRAIN_EVAL.md)
- [Closed-Loop Eval in Carla](docs/EVAL_IN_CARLA.md)
- [Convert Codes from Nuscenes to Bench2Drive](docs/CONVERT_GUIDE.md)

# Results and Pre-trained Models

## UniAD and VAD

| Method | L2 (m) 2s | Driving Score | Success Rate(%) | Config | Download |
| :---: | :---: | :---: | :---: |  :---: |:---: |
| UniAD-Tiny |0.80 | 32.00  |  9.54 | [config](adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py) | [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/bevformer_tiny_b2d.pth)/[Baidu Cloud](https://pan.baidu.com/s/1psr7AKYHD7CitZ30Bz-9sA?pwd=1234 )|
| UniAD-Base |0.73 | 37.72  |  9.54 | [config](adzoo/uniad/configs/stage2_e2e/tiny_e2e_b2d.py) | [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/uniad_base_b2d.pth)/[Baidu Cloud](https://pan.baidu.com/s/11p9IUGqTax1f4W_qsdLCRw?pwd=1234) |
| VAD        |0.91 | 39.4  | 10.0 | [config](adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py) | [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/vad_b2d_base.pth)/[Baidu Cloud]( https://pan.baidu.com/s/11p9IUGqTax1f4W_qsdLCRw?pwd=1234) |

## BEVFormer

| Method | mAP | NDS | Config | Download |
| :---: | :---: | :---: | :---: |  :---: |
| BEVFormer-Tiny | 0.37 | 0.43  | [config](adzoo/bevformer/configs/bevformer/bevformer_tiny_b2d.py) | [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/bevformer_tiny_b2d.pth)/[Baidu Cloud](https://pan.baidu.com/s/1TWMs9YgKYm2DF5YfXF8i3g?pwd=1234) |
| BEVFormer-Base | 0.63 | 0.67  | [config](adzoo/bevformer/configs/bevformer/bevformer_base_b2d.py) | [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/bevformer_base_b2d.pth)/[Baidu Cloud](https://pan.baidu.com/s/1Y4VkE1gc8BU0zJ4z2fmIkQ?pwd=1234) |


# Related Resources

- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [UniAD](https://github.com/OpenDriveLab/UniAD) 
- [VAD](https://github.com/hustvl/VAD)
