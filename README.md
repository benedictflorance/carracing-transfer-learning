# CarRacing Transfer Learning
## _ESE 650 Final Project Visualization_

Benedict Florance Arockiaraj, Richard Change, Wesley Yee
University of Pennsylvania, Spring 2022

## Description

This repo contains scripts to train various transfer learning algorithms as well as to visualize the trained transfer learning algorithms.

## Other Repositories:
- [Dreamerv2 Car Racing](https://github.com/benedictflorance/dreamerv2_carracing)
- [Customized Car Racing Gym Environment](https://github.com/benedictflorance/gym)

## Contributions

For this project, the individual contributions were as follows:
- Benedict: literature review, customized and trained base and transfer learning RL algorithms (Dreamerv2, SB3) and selected the best models/obtained plots, report writing
- Richard: literature review, parameter tuning for customized Carracing gym environments, and analysis on each model's performance, report writing
- Wesley: literature review, customizing Carracing gym environments, running and development of best model visualizations, report writing


## Training the models
We utilized 3 model-free transfer learning algorithms for this project: DDPG, PPO, SAC. The 3 scripts labeled 
```sh
carracing_ddpg.py
carracing_ppo.py
carracing_sac.py
```
are used to generate and train each respective algorithm. In order to run one, simply run the following prefix followed by a set number of required input parameters
```sh
python3 carracing_ddpg.py
```
These input parameters are as follows:
- logdir: the directory to which you want logs to be outputted to
- game_mode: for the CarRacing OpenAI environment, these can be either "high_friction", "high_accel", or "slow_braking"
- train_map_id: the random seed number for which the training map will be generated
- eval_map_id: the random seed number for which the evaluation map will be generated
- load_path: the directory to which the model will be saved for checkpointing purposes

## Visualizing the models
The 3 scripts which allow you to visualize and save the models are labeled:
```sh
carracing_ddpg_vis.py
carracing_ppo_vis.py
carracing_sac_vis.py
```
Note that in order to run these scripts, the library ```stable_baselines3``` and ```ffmpeg``` should already be installed. Additionally, the models for each respective TL algorithm should already be saved in a .zip file under the following folder directory structure:
```
carracing_ddpg_vis.py
carracing_ppo_vis.py
carracing_sac_vis.py
|___high_accel
|_______best_model_ppo.zip
|_______best_model_sac.zip
|___high_friction
| ...
|___slow_braking
| ...
```
Next, simply run each visualization script in the same manner as the training scripts, with the following arguments:
- logdir: the directory to which you want logs to be outputted to
- game_mode: for the CarRacing OpenAI environment, these can be either "high_friction", "high_accel", or "slow_braking"
- eval_map_id: the random seed number for which the evaluation map will be generated
- model: the name of the transfer learning model being used (used for saving/loading)

The videos will be automatically saved in a folder labeled ```videos``` in the same folder as the project directory.
