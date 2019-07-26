import pandas as pd
import numpy as np
import copy

# df = pd.read_csv('samples.csv')
# ans = df['reward_detail'][0]
# ans = np.array(ans[1:-1].split(" "),dtype=int)
# print (ans)

##################generate sample file
# df = pd.read_csv("item_space.csv")
# item_ids = df['item_ids']
# item_emb = df['item_emb']
# item_ids_emb_dict = dict(zip(item_ids,item_emb))
# reward_choice = [0,0,0,1,1,5]
#
# sample_num = 10000
# def next_s(state,action,reward_detail):
#     next_state = copy.copy(state)
#     for i in range(0,len(reward_detail)):
#         if reward_detail[i]>0:
#             next_state.append(action[i])
#     next_state = next_state[-6:]
#     return next_state
# sample_data = []
# for i in range(0,sample_num):
#     state_len = 6
#     action_len = 3
#     state = [str(v) for v in np.random.choice(item_ids,state_len)]
#
#     action = [str(v) for v in np.random.choice(item_ids,action_len)]
#     reward_detail = np.random.choice(reward_choice,action_len)
#     next_state = next_s(state,action,reward_detail)
#     reward = sum(reward_detail)
#     sample_data.append((state,action,reward,next_state,reward_detail))
#
# sample_now = []
# for sample in sample_data:
#     state_n = '|'.join(sample[0])
#     action_n = '|'.join(sample[1])
#     next_state_n = '|'.join(sample[3])
#     #reward_detail_n = ','.join(sample[4])
#     sample_now.append((state_n,action_n,sample[2],next_state_n,sample[4]))
# column_name = ['state','action','reward','next_state','reward_detail']
# data = pd.DataFrame(sample_now,columns=column_name)
# print (data)
# data.to_csv("samples.csv",index=0)

##############################generate item space file
# item_num = 500
# item_ids = [i for i in range(0,item_num)]
# item_ids = [str(id) for id in item_ids]
# item_emb = np.random.rand(item_num,10)
# item_ids_emb_dict = dict(zip(item_ids,item_emb))
# col_names = ['item_ids','item_emb']
# sample_data = []
#
# for id,emb in item_ids_emb_dict.items():
#     emb_str = ",".join([str(v) for v in emb])
#     sample_data.append((id,emb_str))
# data = pd.DataFrame(sample_data,columns=col_names)
# data.to_csv("item_space.csv",index=0)
#
# df = pd.read_csv("item_space.csv")
# print (df["item_emb"])

############################## test data FundEnv file
# print (item_ids_emb_dict)
# print(item_ids_emb_dict[0])
# print(np.array(item_ids_emb_dict[0].split(','),dtype=float))
#
# state = np.array(df_samples['state'].iloc[0].split('|'),dtype=int)
#
# state = np.array([np.array(item_ids_emb_dict[v].split(','),dtype=float) for v in state])
# state = state.reshape((2,30))
# print (state)