import numpy as np
import pandas as pd
import copy
#### item embedding dict
df_items = pd.read_csv('item_space.csv')
item_ids = df_items['item_ids']
item_emb = df_items['item_emb']
item_ids_emb_dict = dict(zip(item_ids,item_emb))
#### samples
df_samples = pd.read_csv('samples.csv')

def get_state_ids(loc,sample_data):
    state_ids = sample_data.iloc[loc]
    state_ids = np.array(state_ids.split('|'),dtype=int)
    return state_ids

def get_state_emb(state_ids,dict):
    state_emb = np.array([np.array(dict[v].split(','),dtype=float) for v in state_ids])
    state_emb = state_emb.reshape((2,30))
    return state_emb

def next_s(state,action,reward_detail):
    next_state = copy.copy(state)
    for i in range(0,len(reward_detail)):
        if reward_detail[i]>0:
            next_state.append(action[i])
    next_state = next_state[-6:]
    return next_state

class FundEnv(object):
    def __init__(self):
        self.state = get_state_ids(0,df_samples['state'])
        self.user_count = 0
        self.step_count = 0
        self.max_step = 10
        self.reset()

    def reset(self):
        if self.user_count > 9999:
            self.user_count = 0
        self.step_count = 0
        self.state = get_state_ids(self.user_count,df_samples['state'])
        return self.state

    def step(self,action):
        ##### click network and purchase network
        reward_detail = np.random.choice([0,0,0,1,1,5],3)
        ##### click network and purchase network
        reward = sum(reward_detail)
        self.state = next_s(self.state,action,reward_detail)
        self.step_count += 1
        if self.step_count > self.max_step:
            done = 1
            self.user_count += 1
        else:
            done = 0
        return self.state,reward,done


