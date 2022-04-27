import gym
from stable_baselines3 import DDPG
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
import os 
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np
from torch import nn
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--logdir', default="/home/benedict/car_racing/ddpg/")
parser.add_argument('--game_mode', default=None)
parser.add_argument('--train_map_id', default=None)
parser.add_argument('--eval_map_id', default=None)
parser.add_argument('--load_path', default=None)
args = parser.parse_args()

log_dir = args.logdir
os.makedirs(log_dir, exist_ok=True)   
def create_env(env):
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)    
    return env 
train_env = create_env(gym.make("CarRacing-v1", game_mode=args.game_mode, map_id=int(args.train_map_id)))
eval_env = create_env(gym.make("CarRacing-v1", game_mode=args.game_mode, map_id=int(args.eval_map_id)))

new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
checkpoint_callback = CheckpointCallback(save_freq=1e4, save_path=log_dir)
eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir+'/best_model/', log_path=log_dir, eval_freq=1e4)
callback = CallbackList([checkpoint_callback, eval_callback])
n_actions = train_env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = DDPG(policy='MlpPolicy', env=train_env, verbose=1,
  gamma = 0.98,
  buffer_size = 200000,
  learning_starts = 5000,
  action_noise = action_noise,
  gradient_steps = -1,
  learning_rate = 1e-3,
  policy_kwargs = dict(net_arch=[400, 300]))
if args.load_path:
    model.load(args.load_path)
    print("Checkpoint loaded")  
model.set_logger(new_logger)
model.learn(total_timesteps=int(1e6), callback=callback)
