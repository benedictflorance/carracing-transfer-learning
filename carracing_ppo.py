import gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
import os 
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', default="/home/benedict/car_racing/ppo/")
parser.add_argument('--game_mode', default=None)
parser.add_argument('--map_id', default=None)
parser.add_argument('--load_path', default=None)
args = parser.parse_args()

log_dir = args.logdir
os.makedirs(log_dir, exist_ok=True)   
def create_env(env):
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)    
    return env 
train_env = create_env(gym.make("CarRacing-v1", game_mode=args.game_mode, map_id=args.map_id))
eval_env = create_env(gym.make("CarRacing-v1", game_mode=args.game_mode, map_id=args.map_id))

new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
checkpoint_callback = CheckpointCallback(save_freq=1e4, save_path=log_dir)
eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir+'/best_model/', log_path=log_dir, eval_freq=1e4)
callback = CallbackList([checkpoint_callback, eval_callback])
model = PPO(policy='MlpPolicy', env=train_env, verbose=1,
  n_steps =  512,
  batch_size = 128,
  gamma = 0.99,
  gae_lambda = 0.9,
  learning_rate = 3e-5,
  n_epochs = 20,
  ent_coef = 0.0,
  sde_sample_freq = 4,
  max_grad_norm = 0.5,
  vf_coef = 0.5,
  use_sde = True,
  clip_range = 0.4,
  policy_kwargs = dict(
                    log_std_init=-2,
                    ortho_init=False,
                    activation_fn=nn.ReLU,
                    net_arch=[dict(pi=[256, 256], vf=[256, 256])]
                  ))
if args.load_path:
    model.load(args.load_path)
    print("Checkpoint loaded")                
model.set_logger(new_logger)
model.learn(total_timesteps=int(1e6), callback=callback)
model.save("ppo_optimal")