import argparse
import tensorflow as tf
import pprint as pp
from replay_buffer import ReplayBuffer
from simulator import Simulator
from util.logger import logger
from ddpg import DDPG


def main(args):
    s_dim = int(args['embedding']) * int(args['state_item_num'])
    a_dim = int(args['embedding']) * int(args['action_item_num'])
    batch_size = int(args['batch_size'])
    emb_dim = int(args['embedding'])
    env = Simulator()

    agent = DDPG()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="provide arguments for DDPG agent")

    # agent parameters
    parser.add_argument("--embedding", help="dimension of item embedding", default=30)
    parser.add_argument("--state_item_num", help="click history list length for user", default=12)
    parser.add_argument("--action_item_num", help="length of the recommendation item list", default=4)
    parser.add_argument("--actor_lr", help="actor network learning rate", default=0.0001)
    parser.add_argument("--critic_lr", help="critic network learning rate", default=0.001)
    parser.add_argument("--gamma", help="discount factor for critic updates", default=0.99)
    parser.add_argument("--tau", help="soft target update parameter", default=0.001)
    parser.add_argument("--buffer_size", help="max size of the replay buffer", default=1000000)
    parser.add_argument("--batch_size", help="size of minibatch for minbatch-SGD", default=64)

    # run parameters
    parser.add_argument("--max_episodes", help="max num of episodes to do while training", default=50000)
    parser.add_argument("--max_episodes_len", help="max length of 1 episode", default=100)
    parser.add_argument("--summary_dir", help="directory for storing tensorboard info", default='./results')

    args_ = vars(parser.parse_args())
    logger.info(pp.pformat(args_))

    main(args_)
