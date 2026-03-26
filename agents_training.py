from pathlib import Path
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import argparse
import time

import numpy as np
import torch

from rl_adn.environments.env import PowerNetEnv
from rl_adn.agent_td3 import AgentTD3, AgentTD3_GCN, AgentTD3_TAGConv, AgentTD3_GAT
from rl_adn.utils import Config,ReplayBuffer,ReplayBufferGraph,get_episode_return,get_episode_return_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="td3", choices=["td3", "gcn", "tagconv", "gat"])
parser.add_argument("--warm_up", type=int, default=5000)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--learning_rate", type=float, default=6e-5)
parser.add_argument("--num_episode", type=int, default=1000)
args_from_cmd = parser.parse_args()


project_root = Path(__file__).resolve().parent.parent

line_data_path = project_root / "rl_adn" / "data_sources" / "network_data" / "node_34" / "Lines_34.csv"
node_data_path = project_root / "rl_adn" / "data_sources" / "network_data" / "node_34" / "Nodes_34.csv"
time_series_data_path = project_root / "rl_adn" / "data_sources" / "time_series_data" / "34_node_time_series.csv"

env_config = {
    "voltage_limits": [0.95, 1.05],
    "algorithm": "Laurent",
    "battery_list": [11, 15, 26, 29, 33],
    "year": 2021,
    "month": 1,
    "day": 1,
    "train": True,
    "state_pattern": "default",
    "network_info": {
        "vm_pu": 1.0,
        "s_base": 1000,
        "bus_info_file": str(node_data_path),
        "branch_info_file": str(line_data_path),
    },
    "time_series_data_path": str(time_series_data_path),
}

env = PowerNetEnv(env_config)
env_args = {
    "env_name": "PowerNetEnv",
    "state_dim": env.state_space.shape[0],
    "action_dim": env.action_space.shape[0],
    "edge_index": env._get_edge_index(),
    "edge_attr": env._get_edge_attr(),
    "if_discrete": False,
}


if args_from_cmd.model == "td3":
    args = Config(agent_class=AgentTD3, env_class=None, env_args=env_args)
    args.run_name = f"td3_training_{time.strftime('%Y%m%d_%H%M%S')}_{args_from_cmd.random_seed}"
else:
    agent_class_map = {
        "gcn": AgentTD3_GCN,
        "tagconv": AgentTD3_TAGConv,
        "gat": AgentTD3_GAT,
    }
    args = Config(agent_class=agent_class_map[args_from_cmd.model], env_class=None, env_args=env_args)
    args.run_name = f"{args_from_cmd.model}_training_{time.strftime('%Y%m%d_%H%M%S')}_{args_from_cmd.random_seed}"

args.random_seed = args_from_cmd.random_seed

args.train = True
args.buffer_size = 100000
args.warm_up = args_from_cmd.warm_up
args.target_step = 512
args.repeat_times = 1
args.batch_size = 64

GPU_ID = 0
args.gpu_id = GPU_ID
args.num_workers = 4

args.learning_rate = args_from_cmd.learning_rate
args.explore_noise_std = 0.05
args.policy_noise_std = 0.1
args.clip_grad_norm = 3.0
args.state_value_tau = 0
args.soft_update_tau = 5e-3
args.gamma = 0.995
args.num_episode = args_from_cmd.num_episode

if args_from_cmd.model == "td3":
    args.net_dims = (256, 256, 256)
else:
    args.num_features = 6
    args.num_embeddings = 8
    args.bat_index = env_config["battery_list"]
    args.num_nodes = env.node_num
    args.k = 2
    args.heads = 4
    args.act_lr_factor = 1
    args.weight_decay = 0
    args.wdecay_act = 0
    args.wdecay_cri = 0

args.cwd = str(project_root / "runs" / args_from_cmd.model.upper() / args.run_name)

args.init_before_training()
args.print()

if args_from_cmd.model == "td3":
    agent = args.agent_class(
        args.net_dims,
        args.state_dim,
        args.action_dim,
        gpu_id=args.gpu_id,
        args=args,
    )
elif args_from_cmd.model == "gcn":
    agent = args.agent_class(
        args.num_features,
        args.num_embeddings,
        args.action_dim,
        args.bat_index,
        args.num_nodes,
        gpu_id=args.gpu_id,
        args=args,
    )
elif args_from_cmd.model == "tagconv":
    agent = args.agent_class(
        args.num_features,
        args.num_embeddings,
        args.action_dim,
        args.k,
        args.bat_index,
        args.num_nodes,
        gpu_id=args.gpu_id,
        args=args,
    )
elif args_from_cmd.model == "gat":
    agent = args.agent_class(
        args.num_features,
        args.num_embeddings,
        args.action_dim,
        args.heads,
        args.bat_index,
        args.num_nodes,
        gpu_id=args.gpu_id,
        args=args,
    )

if args.if_off_policy:
    if args_from_cmd.model == "td3":
        buffer = ReplayBuffer(
            gpu_id=args.gpu_id,
            num_seqs=args.num_envs,
            max_size=args.buffer_size,
            state_dim=args.state_dim,
            action_dim=1 if args.if_discrete else args.action_dim,
            if_use_per=args.if_use_per,
            args=args,
        )
    else:
        buffer = ReplayBufferGraph(
            gpu_id=args.gpu_id,
            num_seqs=args.num_envs,
            max_size=args.buffer_size,
            state_dim=args.state_dim,
            action_dim=1 if args.if_discrete else args.action_dim,
            node_num=args.num_nodes,
            num_features=args.num_features,
            if_use_per=args.if_use_per,
            args=args,
        )

seed = int(args.random_seed)
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if args.train:
    collect_data = True
    while collect_data:
        print(f"buffer: {buffer.cur_size}")
        with torch.no_grad():
            if args_from_cmd.model == "td3":
                buffer_items = agent.explore_env(env, args.target_step, if_random=True)
            else:
                buffer_items = agent.explore_env_buffer(env, args.target_step, if_random=True)
            buffer.update(buffer_items)
        if buffer.cur_size >= args.warm_up:
            collect_data = False

    torch.set_grad_enabled(False)

    reward_list = []
    reward_for_power_list = []
    reward_for_penalty_list = []

    for i_episode in range(args.num_episode):
        torch.set_grad_enabled(True)
        if args_from_cmd.model == "td3":
            critic_loss, actor_loss = agent.update_net(buffer)
        else:
            critic_loss, actor_loss = agent.update_net_buffer(buffer)
        torch.set_grad_enabled(False)

        eva_epi_reward_list = []
        eva_violation_time_list = []
        eva_violation_value_list = []
        eva_reward_for_power_list = []
        eva_reward_for_penalty_list = []

        for _ in range(5):
            if args_from_cmd.model == "td3":
                episode_reward, violation_time, violation_value, reward_for_power, reward_for_good_action, reward_for_penalty, state_list = get_episode_return(
                    env, agent.act, agent.device
                )
            else:
                episode_reward, violation_time, violation_value, reward_for_power, reward_for_good_action, reward_for_penalty, state_list = get_episode_return_graph(
                    env, agent.act, agent.edge_index, agent.device
                )

            eva_epi_reward_list.append(episode_reward)
            eva_violation_time_list.append(violation_time)
            eva_violation_value_list.append(violation_value)
            eva_reward_for_power_list.append(reward_for_power)
            eva_reward_for_penalty_list.append(reward_for_penalty)

            reward_list.append(episode_reward)
            reward_for_power_list.append(reward_for_power)
            reward_for_penalty_list.append(reward_for_penalty)

        episode_reward = np.mean(eva_epi_reward_list)
        violation_time = np.mean(eva_violation_time_list)
        violation_value = np.mean(eva_violation_value_list)
        reward_for_power = np.mean(eva_reward_for_power_list)
        reward_for_penalty = np.mean(eva_reward_for_penalty_list)

        print(
            f"episode {i_episode}, "
            f"critic_loss={critic_loss:.6f}, "
            f"actor_loss={actor_loss:.6f}, "
            f"reward={episode_reward:.4f}, "
            f"avg_reward_50={np.mean(reward_list[-50:]):.4f}, "
            f"violation_time={violation_time:.2f}, "
            f"violation_value={violation_value:.4f}, "
            f"reward_for_power={reward_for_power:.4f}, "
            f"reward_for_penalty={reward_for_penalty:.4f}"
        )

        with torch.no_grad():
            if args_from_cmd.model == "td3":
                buffer_items = agent.explore_env(env, args.target_step, if_random=False)
            else:
                buffer_items = agent.explore_env_buffer(env, args.target_step, if_random=False)
            buffer.update(buffer_items)

    agent.save_or_load_agent(args.cwd, if_save=True)
    print(f"model saved to {args.cwd}")