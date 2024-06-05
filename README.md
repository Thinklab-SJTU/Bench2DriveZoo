# TCP/ADMLP Bench2Drive

# Checkpoint
- TCP
    - [Hugging Face Link](https://huggingface.co/rethinklab/Bench2DriveZoo/tree/main)
    - [Baidu Cloud Link](https://pan.baidu.com/s/1CgYscY2esIJLRepkO3FBvQ?pwd=1234)
- [ADMLP]
    - [Hugging Face Link](https://huggingface.co/rethinklab/Bench2DriveZoo/tree/main)
    - [Baidu Cloud Link](https://pan.baidu.com/s/1RefJxk0B4kYcnf63Vi-ISA?pwd=1234)

# Training
First, set the dataset path in ``TCP/config.py`` or ``ADMLP/config.py``.
Training:
```bash
    cd Bench2Drive-Zoo/
    # TCP
    python TCP/train.py --gpus NUM_OF_GPUS
    # ADMLP
    python ADMLP/train.py --gpus NUM_OF_GPUS
```

# Open Loop Evaluation
```bash
    # TCP
    python TCP/test.py
    # ADMLP
    python ADMLP/test.py
```

# Closed Loop Evaluation    
Please follow these steps to evaluate TCP/ADMLP in Carla:

## Preparations
- Install Bench2Drive from [here](https://github.com/Thinklab-SJTU/Bench2Drive).
- Follow [this](https://github.com/Thinklab-SJTU/Bench2Drive/tree/main#setup) to install Carla.

## Link this repo to Bench2Drive

```bash
    # Add your agent code
    cd Bench2Drive/leaderboard
    mkdir team_code
    cd Bench2Drive/leaderboard/team_code
    ln -s YOUR_TEAM_AGENT ./  # link your agent code

    cd Bench2Drive/
    ln -s Bench2DriveZoo/team_code/*  ./ # link entire repo to Bench2Drive
```

## Run Evaluation 
Follow [this](https://github.com/Thinklab-SJTU/Bench2Drive?tab=readme-ov-file#eval-tools) to use evaluation tools of Bench2Drive.