import numpy as np
from gymnasium import spaces
import gymnasium as gym
import time
# import pandas as pd
import pickle
from stable_baselines3.common.monitor import Monitor
# #Number of rows in the table
# num_rows = 100
# from stable_baselines3.common.env_checker import check_env
# import pandas as pd

# # Generate random x and y coordinates
# x_coords = np.random.uniform(0.1, 0.9, num_rows)
# y_coords = np.random.uniform(0.1, 0.9, num_rows)

# # Round coordinates to 2 decimal places
# x_coords_rounded = np.round(x_coords, 2)
# y_coords_rounded = np.round(y_coords, 2)

# # Create a DataFrame with rounded coordinates
# data = {
#     'X': x_coords_rounded,
#     'Y': y_coords_rounded
# }
# df = pd.DataFrame(data)

# # Generate random rewards for unique rounded coordinates
# unique_coords = df.drop_duplicates()
# unique_coords['Reward'] = np.random.uniform(0.1, 1, len(unique_coords))

# # Merge the rewards back to the original DataFrame
# action_reward_table = df.merge(unique_coords, on=['X', 'Y'], how='left')


# with open('action_reward_table.pkl', 'wb') as file:
#     pickle.dump( action_reward_table,file)

with open('action_reward_table.pkl', 'rb') as file:
    action_reward_table = pickle.load(file)


class TabularEnv(gym.Env):

    def __init__(self):

        self.observation_space = spaces.Box(
                low = -np.ones((10, )),
                high=np.ones((10, )),
                dtype=np.float32
            )
        
        self.action_space =    spaces.Discrete(100)
        
        # Initialize the reward table
        self.action_reward_space =     spaces.Box(
                                low=np.array([0, 0 , -8]),
                                high=np.array([1, 1 , 8]),
                                dtype=np.float32)  # 3D table with rewards in range [0, 1]
        
        # Number of rows in the table

        self.table = action_reward_table.copy()

        self.time_track = time.time()
        self.episode_duration = 1
        self.cum_reward = 1
        
        
    def reset(self , seed =0):
        self.table = action_reward_table.copy()
        self.cum_reward = 1
        observation = self.table
        return np.array(observation , dtype=np.float32) , {}
        
    
    def step(self , action):
        
        reward = self.get_reward(action  , self.table)

        info = {}
        truncated = False # calculate truncated here

        # if abs(time.time() - self.time_track) > self.episode_duration:
        #     done_env = True
        #     self.time_track = time.time()

        # elif len(self.table) <2 :
        #     done_env = True
        #     self.time_track = time.time()

        done_env = False

        episode_finished = True 

        for r in self.table["Reward"]:
            if r > 0 :
                episode_finished = False

        if episode_finished:
            done_env = True
        
        return np.array(self.table , dtype=np.float32)  ,reward, done_env, truncated, info
    
    def get_reward(self ,index , table):
        table = self.table.copy()
        index = int(index)
    # Find the matching row
        try:
            match = table.iloc[index]["Reward"].astype(np.float32)
        except:
            match = None

        if match is None:
            return np.float32(-0.9 *10)

        if float(match) >0:
            self.table.at[index, "Reward"] = -0.9   
            self.cum_reward+=1
            #return match * self.cum_reward
            if bool(match == max(env.table.iloc[:]["Reward"])):
                return np.float32(0.9999)
            else :
                return match
        else:
            self.cum_reward = 1
            return np.float32(-0.9999)
        
    def make_env():
     return Monitor(TabularEnv())


# env = TabularEnv.make_env()
# observation_ = env.reset()
# action_ = env.action_space.sample()
# observation_  ,reward_, done_, truncated, info = env.step(action_)
# done = False

# actions = []
# rewards = []
# observations = []
# # observations.append(observation)

# while not done:

#     action = env.action_space.sample()
#     observation  ,reward, done, truncated, info = env.step(action)
#     actions.append(action)
#     rewards.append(reward)
#     observations.append(observation)


# actions = []
# rewards = []
# observations = []
# dones = []

# done = False
# counter = 0

# while len(actions) < 100000:

#     observation , _ = env.reset()
#     done = False

#     while not done :

#         observations.append(observation)
#         action = env.action_space.sample()
#         observation  ,reward, done, truncated, info = env.step(action)

#         actions.append(np.round(action , 2))
#         rewards.append(reward)
#         dones.append(done)

#         counter+=1
#         print("Collecting samples" , len(actions))






# main_data = pd.DataFrame({
#     'Actions': actions,
#     'Observations': observations,
#     'Rewards': rewards
# })

# with open('main_data.pkl', 'wb') as file:
#      pickle.dump(main_data , file)


# with open('main_data.pkl', 'rb') as file:
#     main_data = pickle.load(file)

# main_data = main_data[main_data["Rewards"] > 0]

# print()
# main_data = pd.DataFrame({
#     'Actions': actions,
#     'Observations': observations,
#     'Rewards': rewards
# })





# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# import torch.nn as nn
# import torch.optim as optim
# import torch


# X = torch.tensor(np.stack( main_data['Observations'].values , dtype=np.float32 ) , device = 'cuda')
# y = torch.tensor(np.array(main_data['Actions'].tolist(), dtype=np.float32) , device = 'cuda') 
# z = torch.tensor(np.array(main_data['Rewards'].tolist(), dtype=np.float32).reshape(-1 , 1) , device = 'cuda')
# # X = X + z

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.fc1 = nn.Linear(len(action_reward_table) * 3, 1024)  # Flatten the input
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(1024, 512)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(512, 128)
#         self.relu3 = nn.ReLU()
#         self.fc4 = nn.Linear(128, 2)  # Output layer for 2 predictions

#     def forward(self, x):
#         x = x.contiguous().view(x.size(0), -1)  # Flatten the input
#         x = self.fc1(x)
#         x = self.relu1(x)
#         x = self.fc2(x)
#         x = self.relu2(x)
#         x = self.fc3(x)
#         x = self.relu3(x)
#         x = self.fc4(x)
#         return x


# model = SimpleModel().to('cuda')

# # Define the loss function and optimizer
# criterion = nn.HuberLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.00002)
# outputs = model(X_train)
# loss = criterion(outputs, y_train)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

# num_epochs = 3000  # Number of epochs
# batch_size = 64  # Batch size

# for epoch in range(num_epochs):
#     for i in range(0, len(X), batch_size):
#         # Get the batch
#         X_batch = X[i:i+batch_size]
#         y_batch = y[i:i+batch_size]

#         # Forward pass
#         outputs = model(X_batch)
#         loss = criterion(outputs, y_batch)

#         # Backward pass and optimization
#         optimizer.zero_grad()  # Clear previous gradients
#         loss.backward()        # Compute gradients
#         optimizer.step()       # Update parameters

#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# model.eval()

# # model.fit(X_train, y_train)
# with torch.no_grad():
#     y_pred = model(X_test)

# mse = mean_squared_error(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
# r2 = r2_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

# print()

# observation , _ = env.reset()
# done = False
# counter =0

# while counter < 100:
#         episode_reward =0
#         while not done :

#             action = model(torch.tensor(observation , device = 'cuda' , dtype=torch.float32).reshape(-1 , len(action_reward_table) , 3)).reshape((2,-1))
#             observation  ,reward, done, truncated, info = env.step(action.cpu().detach().numpy())
#             episode_reward+=reward


#             counter+=1
#         print("Episode Reward: " , episode_reward)

# print()

# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt

# # Assuming X and y are numpy arrays with shape (n_samples, 2)
# X_x = X[:100, 0]
# X_y = X[:100, 1]
# y_x = y[:100, 0]
# y_y = y[:100, 1]

# plt.figure(figsize=(10, 5))

# # Plot X coordinates
# plt.subplot(1, 2, 1)
# plt.scatter(X_x, X_y, c='blue', label='X coordinates')
# plt.xlabel('X_x')
# plt.ylabel('X_y')
# plt.title('X Coordinates')
# plt.legend()

# # Plot y coordinates
# plt.subplot(1, 2, 2)
# plt.scatter(y_x, y_y, c='red', label='y coordinates')
# plt.xlabel('y_x')
# plt.ylabel('y_y')
# plt.title('y Coordinates')
# plt.legend()

# plt.tight_layout()
# plt.show()
# print()

from stable_baselines3 import PPO , DDPG , DQN , HerReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
# import torch
from stable_baselines3.common.logger import configure
logger = configure("logs\\", ["stdout", "csv", "tensorboard"])
import optuna


# check_env(TabularEnv())
# env = TabularEnv.make_env(TabularEnv)
# model = PPO(env = env,  policy = 'MlpPolicy' ,learning_rate= 0.00002 , n_epochs=100)
# model.set_logger(logger)
# model.learn(total_timesteps=10000 , log_interval=1 , progress_bar=True  )

# results = evaluate_policy(model ,env , n_eval_episodes=10 )



# observation , _ = env.reset()
# done = False
# counter =0

# while counter < 10:
#         episode_reward =0
#         while not done :

#             action = model.predict(observation)
#             observation  ,reward, done, truncated, info = env.step(action[0])
#             episode_reward+=reward


#             counter+=1
#         print("Episode Reward: " , episode_reward)




def optimize_PPO(trail):
    n_layers = trail.suggest_int("n_layers", 3, 9)
    net_arch = [trail.suggest_int(f"neurons_hidden_layer_{i+1}", 1024, 4096) for i in range(n_layers)]

    return{
        "policy":trail.suggest_categorical("policy", ["MlpPolicy"]),
        "learning_rate":trail.suggest_float("learning_rate" , 0.00005 ,  0.003 ),
        "batch_size":trail.suggest_int("batch_size" , 16 , 256),
        "gamma":trail.suggest_float("gamma" , 0.8, 0.99),



    } , net_arch


TOTAL_TIME_STEPS = 100000
N_ENVS = 1
N_TRIALS = 20

# import optuna
turn = 0
def optimize_agent(trail):
    global turn , TOTAL_TIME_STEPS
    turn+=1
    logger = configure("logs\\PPO"+str(turn), ["stdout", "csv", "tensorboard"])
    
    env = TabularEnv.make_env(TabularEnv)
    model_params , net_arch = optimize_PPO(trail)

    model = DQN( env=env ,  device="cuda" , policy_kwargs={ "net_arch" : net_arch} , verbose= 1      , **model_params    )
    model.set_logger(logger)
    model.learn(total_timesteps= TOTAL_TIME_STEPS , log_interval=1 , reset_num_timesteps=True , progress_bar=True)
    
    mean_reward , _ = evaluate_policy(model ,env,n_eval_episodes= 10)
    print("Evaluated trail and mean reward is" , mean_reward)

    return mean_reward

if __name__ == '__main__':

        # study = optuna.create_study(direction='maximize')
        # study.optimize(optimize_agent, n_trials=N_TRIALS, n_jobs=N_ENVS ,show_progress_bar= True )
        
        # with open("best_params/params_PPO", "wb") as file:
        #     pickle.dump(study.best_params, file)

        # with open("best_params/study_14_april", "wb") as file:
        #     pickle.dump(study, file)
        logger = configure("logs\\PPO", ["stdout", "csv", "tensorboard"])
        model_params = {
        "policy":"MlpPolicy",
        "gamma":0.1,
 } 

        env = TabularEnv.make_env()
        model = PPO( env=env ,  device="cuda" , verbose= 1    , **model_params  )
        model.set_logger(logger)
        model.learn(total_timesteps= TOTAL_TIME_STEPS , log_interval=1 , reset_num_timesteps=True , progress_bar=True )
        model.save("PPO")
        mean_reward , _ = evaluate_policy(model ,env,n_eval_episodes= 5 , render=True  )
        print("Evaluated trail and mean reward is" , mean_reward)
        # model = model.load("ppo")
        # model.set_logger(logger)

        print()
        # for i in range (1):
        #      episode_reward = 0
        #      observation = env.reset()[0]
        #      done = False
             
        #      while not done:

        #         action , _ = model.predict(observation)
        #         print(action)
        #         print(env.table.iloc[int(action)])
        #         observation  ,reward, done, truncated, info = env.step(int(action))
        #         print(env.table.iloc[int(action)])
        #      episode_reward+=reward
        #      print("Ep reward:" , episode_reward)


                 
            


