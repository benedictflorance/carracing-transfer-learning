import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from torch import nn
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', default="./logs")
parser.add_argument('--game_mode', default="map")
parser.add_argument('--eval_map_id', default=2)
parser.add_argument('--model', default="ppo")
args = parser.parse_args()

log_dir = args.logdir
os.makedirs(log_dir, exist_ok=True)   
eval_env = gym.make("CarRacing-v1", game_mode=args.game_mode, map_id=int(args.eval_map_id))
env_wrapper = gym.wrappers.RecordVideo(eval_env, './videos/'+args.game_mode+'/'+args.model)

model = PPO.load(path='C:\\Users\\Wesley Yee\\Documents\\Github\\carracing-transfer-learning\\'+args.game_mode+'\\best_model_'+args.model, env=eval_env)

obs = env_wrapper.reset()

dones = False
while not dones:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env_wrapper.step(action)
    # eval_env.render()

env_wrapper.close()
