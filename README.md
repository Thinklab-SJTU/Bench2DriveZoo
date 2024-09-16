
<h2 align="center">
  <img src="assets/bench2drive.jpg" style="width: 100%; height: auto;">
</h2>
<h2 align="center">
Bench2DriveZoo (with Think2Drive as Teacher Model)
</h2>
<h2 align="center">
  <img src="assets/bench2drivezoo.png" style="width: 100%; height: auto;">
</h2>


# Introduction

- This repo contains the training, open-loop evaluation, and closed-loop evaluation code for [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [UniAD](https://github.com/OpenDriveLab/UniAD) , [VAD](https://github.com/hustvl/VAD) in [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive). **All models are student models of the world model RL teacher - [Think2Drive](https://arxiv.org/abs/2402.16720).**
- We merge multiple dependencies of UniAD and VAD including mmcv, mmseg, mmdet, and mmdet3d (v0.17.1) into a single library. As a result, it could support latest pytorch and advanced frameworks like deepspeed for acceleration.
- Use "git checkout tcp/admlp" to obtain their corresponding training and evaluation code.
- **To calculate smoothness and efficiency**, remember to write self.metric_info related codes in your own team code agent. This two metrics require the state of the ego vehicle at each step in 20Hz. You may comment the lines about saving sensor data to save disk space.

# Citation <a name="citation"></a>

Please consider citing our papers if the project helps your research with the following BibTex:

```bibtex
@article{jia2024bench,
  title={Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving},
  author={Xiaosong Jia and Zhenjie Yang and Qifeng Li and Zhiyuan Zhang and Junchi Yan},
  journal={arXiv preprint arXiv:2406.03877},
  year={2024}
}

@inproceedings{li2024think,
  title={Think2Drive: Efficient Reinforcement Learning by Thinking in Latent World Model for Quasi-Realistic Autonomous Driving (in CARLA-v2)},
  author={Qifeng Li and Xiaosong Jia and Shaobo Wang and Junchi Yan},
  booktitle={ECCV},
  year={2024}
}
```

# Getting Started

- [Installation](docs/INSTALL.md)
- [Prepare Dataset](docs/INSTALL.md)
- [Train and Open-Loop Eval](docs/TRAIN_EVAL.md)
- [Closed-Loop Eval in CARLA](docs/EVAL_IN_CARLA.md)
- [Convert Codes from Nuscenes to Bench2Drive](docs/CONVERT_GUIDE.md)

# Results and Pre-trained Models

## UniAD and VAD

As stated in the [news](https://github.com/Thinklab-SJTU/Bench2Drive) at 2024/08/27, there are several fixed bugs and changed protocols. Thus, the old version of closed-loop performance is deprecated.


| Method | L2 (m) 2s | Driving Score | Success Rate(%) | Config | Download | Eval Json|
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: |
| UniAD-Tiny |0.80 | 40.73 (deprecated 32.00)  |  13.18 (deprecated 9.54) | [config](adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py) | [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/bevformer_tiny_b2d.pth)/[Baidu Cloud](https://pan.baidu.com/s/1psr7AKYHD7CitZ30Bz-9sA?pwd=1234 )| [New Version](analysis/UniAD-Tiny.json) |
| UniAD-Base |0.73 | 45.81 (deprecated  37.72)  |  16.36 (deprecated 9.54) | [config](adzoo/uniad/configs/stage2_e2e/tiny_e2e_b2d.py) | [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/uniad_base_b2d.pth)/[Baidu Cloud](https://pan.baidu.com/s/11p9IUGqTax1f4W_qsdLCRw?pwd=1234) | [New Version](analysis/UniAD-Base.json) |
| VAD        |0.91 | 42.35 (deprecated 39.42)  | 15.00 (deprecated 10.00) | [config](adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py) | [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/vad_b2d_base.pth)/[Baidu Cloud](https://pan.baidu.com/s/1rK7Z_D-JsA7kBJmEUcMMyg?pwd=1234) | [New Version](analysis/VAD.json) |

## BEVFormer

| Method | mAP | NDS | Config | Download |
| :---: | :---: | :---: | :---: |  :---: |
| BEVFormer-Tiny | 0.37 | 0.43  | [config](adzoo/bevformer/configs/bevformer/bevformer_tiny_b2d.py) | [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/bevformer_tiny_b2d.pth)/[Baidu Cloud](https://pan.baidu.com/s/1TWMs9YgKYm2DF5YfXF8i3g?pwd=1234) |
| BEVFormer-Base | 0.63 | 0.67  | [config](adzoo/bevformer/configs/bevformer/bevformer_base_b2d.py) | [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/bevformer_base_b2d.pth)/[Baidu Cloud](https://pan.baidu.com/s/1Y4VkE1gc8BU0zJ4z2fmIkQ?pwd=1234) |


# Failure Cases Analysis

We provide some visualization videos and qualitatively analysis for TCP-traj, UniAD-Base, VAD-Base at [here](analysis/analysis.md).  You may refer to https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/uniad/vad/team_code/vad_b2d_agent_visualize.py to write your own visualization code.

# Related Resources

- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [UniAD](https://github.com/OpenDriveLab/UniAD) 
- [VAD](https://github.com/hustvl/VAD)

