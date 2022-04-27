import gym
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
import os 
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from torch import nn
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--logdir', default="/home/benedict/car_racing/sac/")
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
model = SAC(policy='MlpPolicy', env=train_env, verbose=1,
  learning_rate = 7.3e-4,
  buffer_size = 300000,
  batch_size = 256,
  ent_coef = 'auto',
  gamma = 0.99,
  tau = 0.02,
  train_freq = 8,
  gradient_steps = 10,
  learning_starts = 1000,
  use_sde = True,
  use_sde_at_warmup = True)
if args.load_path:
    model.load(args.load_path)
    print("Checkpoint loaded") 
model.set_logger(new_logger)
model.learn(total_timesteps=int(1e6), callback=callback)
