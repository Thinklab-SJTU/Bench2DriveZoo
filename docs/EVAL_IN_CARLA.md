# Closed Loop Evaluation    

Please follow these steps to evaluate UniAD and VAD in Carla:

## Preparations

- Install this repo as [doc](docs/INSTALL.md). 
- Install Bench2Drive from [here](https://github.com/Thinklab-SJTU/Bench2Drive).


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

## Run evaluation 

Follow [this](https://github.com/Thinklab-SJTU/Bench2Drive?tab=readme-ov-file#eval-tools) to use evaluation tools of Bench2Drive.

