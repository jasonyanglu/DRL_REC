import argparse
import pprint as pp
from FundEnv import FundEnv, item_ids_emb_dict
from util.logger import logger
from ddpg_rec import DDPG_REC


def main(args):
    env = FundEnv()
    item_space = item_ids_emb_dict
    agent = DDPG_REC(int(args['state_item_num']),
                     int(args['action_item_num']),
                     int(args['embedding']),
                     int(args['batch_size']),
                     float(args['tau']),
                     float(args['actor_lr']),
                     float(args['critic_lr']),
                     float(args['gamma']),
                     int(args['buffer_size']),
                     item_space,
                     args['summary_dir'])

    for i in range(int(args['max_episodes'])):
        ep_reward = 0.
        ep_q_value = 0.
        loss = 0.
        state = env.reset()
        # update average parameters every 1000 episodes
        if (i + 1) % 10 == 0:
            env.rewards, env.group_sizes, env.avg_states, env.avg_actions = env.avg_group()

        for j in range(args['max_episodes_len']):
            action = agent.action(state)
            reward, n_state, done = env.step(action)
            ep_reward += reward
            ep_q_value_, critic_loss = agent.perceive_and_train(state, action, reward, n_state, done)
            ep_q_value += ep_q_value_
            loss += critic_loss
            state = n_state
            if done:
                break
            if (j + 1) % 50 == 0:
                logger.info("=========={0} episode of {1} round: {2} reward=========".format(i, j, ep_reward))
            agent.write_summary(ep_reward, ep_q_value_, loss)

    agent.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="provide arguments for DDPG agent")

    # agent parameters
    parser.add_argument("--embedding", help="dimension of item embedding", default=10)
    parser.add_argument("--state_item_num", help="click history list length for user", default=6)
    parser.add_argument("--action_item_num", help="length of the recommendation item list", default=3)
    parser.add_argument("--actor_lr", help="actor network learning rate", default=0.0001)
    parser.add_argument("--critic_lr", help="critic network learning rate", default=0.001)
    parser.add_argument("--gamma", help="discount factor for critic updates", default=0.99)
    parser.add_argument("--tau", help="soft target update parameter", default=0.001)
    parser.add_argument("--buffer_size", help="max size of the replay buffer", default=100000)
    parser.add_argument("--batch_size", help="size of minibatch for minbatch-SGD", default=64)

    # run parameters
    parser.add_argument("--max_episodes", help="max num of episodes to do while training", default=50000)
    parser.add_argument("--max_episodes_len", help="max length of 1 episode", default=100)
    parser.add_argument("--summary_dir", help="directory for storing tensorboard info", default='./results')

    args_ = vars(parser.parse_args())
    logger.info(pp.pformat(args_))

    main(args_)
