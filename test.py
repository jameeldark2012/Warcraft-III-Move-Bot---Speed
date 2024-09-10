from stable_baselines3.ppo import PPO
from environment import WarAMBOT
from stable_baselines3.common.env_checker import check_env

env =WarAMBOT.make_env(WarAMBOT)


model = PPO(env= env , policy="MlpPolicy" , tensorboard_log="logs//PPO" )

model.learn(total_timesteps=1000 , log_interval=1 , progress_bar= True)