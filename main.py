import torch
import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from environment import WarAMBOT
import threading
from torch.utils.tensorboard import SummaryWriter
import shutil
import time
import logging
from sklearn.preprocessing import MinMaxScaler


logging.basicConfig(level=logging.INFO)



device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
global memory , i_episode, time_steps

#__________________________ WARNING BATCH SIZE MUST BE DIVIDABLE BY 10 ________________________ 
BATCH_SIZE = 100
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 4000
TAU = 0.020
LR = 1e-6
TAU_FREQUENCY = 10000




#env = WarAMBOT.make_env(WarAMBOT)
N_ENVS = 25
envs = [WarAMBOT.make_env(WarAMBOT) for i  in range(N_ENVS)]
TOTAL_TIME_STEPS = 30000

if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 50000
else:
        num_episodes = 50

directory = 'logs'
#shutil.rmtree(directory)
writer = SummaryWriter(log_dir='logs/DQNMod')

MIN_REWARD = -8
MAX_REWARD =  8
REWARD_RANGE_GENERATED = torch.linspace(MIN_REWARD, MAX_REWARD, 100).reshape(-1 ,2)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(REWARD_RANGE_GENERATED)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(2, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 2)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer3(x)

n_actions = 1
info = {}
n_observations = 1
steps_done = 0
time_steps = 0
i_episode = 0

policy_net = DQN(n_observations, n_actions).to(device).double()
target_net = DQN(n_observations, n_actions).to(device).double()
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)




x = torch.linspace(0.01, 1, 200, device='cuda', dtype=torch.float64)
y = torch.linspace(0.01, 1, 200, device='cuda', dtype=torch.float64)
xy = torch.stack((x, y), dim=1)

thread_local = threading.local()
thread_local.xy = xy.clone()

def select_action(state , env):
    global steps_done 
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            rewards_x_y_plane = policy_net(thread_local.xy).clone()
            sums = rewards_x_y_plane.sum(dim=1)
            maximum_reward = rewards_x_y_plane[torch.argmax(sums)].reshape(1,2)
            thread_local.xy = thread_local.xy[thread_local.xy!=torch.argmax(sums)].reshape(-1,2)

            max_index = torch.argmax(sums)
            mask = torch.ones(sums.size(), dtype=torch.bool)
            mask[torch.argmax(sums)] = False
            thread_local.xy = thread_local.xy[mask]

            return torch.tensor(maximum_reward).reshape((1,1,2))
            
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.float64)

def optimize_model():
    global memory , last_executed_loss

    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    #reward_batch[reward_batch == 0] = -1
    
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    policy_net_predictions = policy_net(state_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros((BATCH_SIZE , 2), device=device)
    target_network_predictions = target_net(non_final_next_states)
    next_state_values =  target_net(non_final_next_states)
   
    # Compute the expected Q values
    reward_batch_expanded = reward_batch.unsqueeze(1)
    reward_batch_expanded = reward_batch[non_final_mask].unsqueeze(1).expand(-1, 2)
    
    reward_batch_expanded = scaler.transform(reward_batch_expanded.cpu())
    reward_batch_expanded = torch.from_numpy(reward_batch_expanded)
    reward_batch_expanded = reward_batch_expanded.to('cuda')


    expected_state_action_values = (next_state_values * GAMMA) + reward_batch_expanded

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    policy_net_predictions_non_terminal = policy_net_predictions[non_final_mask]
    loss = criterion(policy_net_predictions_non_terminal, expected_state_action_values)
    print("Policy network optimized")

    if (current_time - last_executed_loss >= 10):
        print("Loss: " , float(loss))
        writer.add_scalar('Loss Time Steps', float(loss), time_steps)
        last_executed_loss = time.time()


    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    #torch.nn.utils.clip_grad_value_(policy_net.parameters(), 3)
    optimizer.step()

def train(env):
    global memory , time_steps, i_episode
    thread_local.xy = xy.clone()

    for i in range(num_episodes):
        global i_episode
        i_episode+=1
        local_episode = i_episode
        memory_list = []

        episode_reward = 0
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)

        state = env.reset()
        info = {}
        state = torch.tensor(state, dtype=torch.float64, device=device).unsqueeze(0)

        for t in count():

            print("\nTime Steps: " , time_steps,"/",TOTAL_TIME_STEPS)
            time_steps+=1
            action = select_action(state , env)
            observation  ,reward, done, truncated, info = env.step(action.cpu())
            reward = torch.tensor([reward], device=device)

            if done:
                next_state = None
                thread_local.xy = torch.stack((x, y), dim=1)

            else:
                next_state = torch.tensor(observation, dtype=torch.float64, device=device).unsqueeze(0)

            

            episode_reward+=int(reward[0])

            memory_list.append({'state' : state, 'action' : action, 'next_state' : next_state, 'reward' : reward})
            
            state = next_state
            if done:

                
                with lock:
                    for l in memory_list:
                        memory.push(l["state"], l["action"], l["next_state"], l["reward"])

                logging.info("Pushed memory " + threading.current_thread().name)

                print("Episode : " , local_episode , "Episode Reward: " , episode_reward)
                print("Epsilon Threshold: " , eps_threshold)
                # Log the episode reward
                writer.add_scalar('Episode Reward', episode_reward, local_episode)
                writer.add_scalar('Epsilon Threshold', eps_threshold, local_episode)
                break

threads = []
lock = threading.Lock()

for i in range(N_ENVS):
    thread = threading.Thread(target=train , args=[envs[i]])
    threads.append(thread)
    thread.start()

last_executed = time.time()
last_executed_loss = time.time()
last_executed_optimze = time.time()
interval = 10
interval_optimize = 10

while True:
    current_time = time.time()


    if (current_time - last_executed_optimze>= interval_optimize) and len(memory) % BATCH_SIZE == 0 and len(memory) != 0 :
        optimize_model()
        last_executed_optimze = time.time()

    time.sleep(0.1)

    if time_steps % TAU_FREQUENCY == 0 and time_steps != 0 and (current_time - last_executed >= interval):

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        print("Target network weights updated")
        last_executed = time.time()

    if time_steps >= TOTAL_TIME_STEPS:
        print("Training Finished")
        torch.save(policy_net.state_dict(), 'models/DQNMod/policy.pth')
        torch.save(target_net.state_dict(), 'models/DQNMod/target.pth')
        torch.save(memory, 'models/DQNMod/memory.pth')
        torch.save({'time_steps': time_steps,
                    'steps_done' : steps_done,
                    'i_episode' : i_episode} ,  'checkpoint.pth')
        break

print()
    

    

