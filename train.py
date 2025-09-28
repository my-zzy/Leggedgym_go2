import numpy as np
import os
from datetime import datetime
from configs.go2 import Go2Cfg, Go2CfgPPO

import isaacgym
from utils.helpers import get_args
from envs import LeggedRobot
from utils.task_registry import task_registry

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.onpolicy.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    task_registry.register("THUrl",LeggedRobot,Go2Cfg(),Go2CfgPPO())
    args = get_args()
    train(args)
