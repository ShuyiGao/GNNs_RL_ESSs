from pathlib import Path
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import copy
import argparse
import random

import numpy as np
import pandas as pd
import torch

from rl_adn.environments.env import PowerNetEnv
from rl_adn.agent_td3 import AgentTD3, AgentTD3_GCN, AgentTD3_TAGConv, AgentTD3_GAT
from rl_adn.utils import Config


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="all", choices=["td3", "gcn", "tagconv", "gat", "all"])
parser.add_argument("--td3_ckpt", type=str, default="test_TD3")
parser.add_argument("--gcn_ckpt", type=str, default="test_TD3_GCN")
parser.add_argument("--tagconv_ckpt", type=str, default="test_TD3_TAGConv")
parser.add_argument("--gat_ckpt", type=str, default="test_TD3_GAT")
args_from_cmd = parser.parse_args()

project_root = Path(__file__).resolve().parent.parent

env_config = {
    "voltage_limits": [0.95, 1.05],
    "algorithm": "Laurent",
    "battery_list": [11, 15, 26, 29, 33],
    "year": 2021,
    "month": 1,
    "day": 1,
    "train": False,
    "state_pattern": "default",
    "network_info": {
        "vm_pu": 1.0,
        "s_base": 1000,
        "bus_info_file": str(project_root / "rl_adn" / "data_sources" / "network_data" / "node_34" / "Nodes_34.csv"),
        "branch_info_file": str(project_root / "rl_adn" / "data_sources" / "network_data" / "node_34" / "Lines_34.csv"),
    },
    "time_series_data_path": str(project_root / "rl_adn" / "data_sources" / "time_series_data" / "34_node_time_series.csv"),
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_list = ["td3", "gcn", "tagconv", "gat"] if args_from_cmd.model == "all" else [args_from_cmd.model]
results = []

for model_name in model_list:
    env_copy = copy.deepcopy(env)
    state = env_copy.reset()

    if model_name == "td3":
        td3_args = Config(agent_class=AgentTD3, env_class=None, env_args=env_args)
        td3_args.gpu_id = 0
        td3_args.num_workers = 4
        td3_args.random_seed = args_from_cmd.random_seed
        td3_args.net_dims = (256, 256, 256)

        agent = td3_args.agent_class(
            td3_args.net_dims,
            td3_args.state_dim,
            td3_args.action_dim,
            gpu_id=td3_args.gpu_id,
            args=td3_args,
        )
        agent.save_or_load_agent(cwd=args_from_cmd.td3_ckpt, if_save=False)

    elif model_name == "gcn":
        gcn_args = Config(agent_class=AgentTD3_GCN, env_class=None, env_args=env_args)
        gcn_args.gpu_id = 0
        gcn_args.num_workers = 4
        gcn_args.random_seed = args_from_cmd.random_seed
        gcn_args.num_features = 6
        gcn_args.num_embeddings = 8
        gcn_args.bat_index = env_copy.battery_list
        gcn_args.num_nodes = env_copy.node_num

        agent = gcn_args.agent_class(
            gcn_args.num_features,
            gcn_args.num_embeddings,
            gcn_args.action_dim,
            gcn_args.bat_index,
            gcn_args.num_nodes,
            gpu_id=gcn_args.gpu_id,
            args=gcn_args,
        )
        agent.save_or_load_agent(cwd=args_from_cmd.gcn_ckpt, if_save=False)
        agent.act.bat_index = torch.tensor(gcn_args.bat_index, dtype=torch.long)
        agent.act.non_bat_index = torch.tensor(
            [i for i in range(gcn_args.num_nodes) if i not in gcn_args.bat_index], dtype=torch.long
        )
        agent.act.num_nodes = gcn_args.num_nodes

    elif model_name == "tagconv":
        tag_args = Config(agent_class=AgentTD3_TAGConv, env_class=None, env_args=env_args)
        tag_args.gpu_id = 0
        tag_args.num_workers = 4
        tag_args.random_seed = args_from_cmd.random_seed
        tag_args.num_features = 6
        tag_args.num_embeddings = 8
        tag_args.bat_index = env_copy.battery_list
        tag_args.num_nodes = env_copy.node_num
        tag_args.k = 2

        agent = tag_args.agent_class(
            tag_args.num_features,
            tag_args.num_embeddings,
            tag_args.action_dim,
            tag_args.k,
            tag_args.bat_index,
            tag_args.num_nodes,
            gpu_id=tag_args.gpu_id,
            args=tag_args,
        )
        agent.save_or_load_agent(cwd=args_from_cmd.tagconv_ckpt, if_save=False)
        agent.act.bat_index = torch.tensor(tag_args.bat_index, dtype=torch.long)
        agent.act.non_bat_index = torch.tensor(
            [i for i in range(tag_args.num_nodes) if i not in tag_args.bat_index], dtype=torch.long
        )
        agent.act.num_nodes = tag_args.num_nodes

    elif model_name == "gat":
        gat_args = Config(agent_class=AgentTD3_GAT, env_class=None, env_args=env_args)
        gat_args.gpu_id = 0
        gat_args.num_workers = 4
        gat_args.random_seed = args_from_cmd.random_seed
        gat_args.num_features = 6
        gat_args.num_embeddings = 8
        gat_args.bat_index = env_copy.battery_list
        gat_args.num_nodes = env_copy.node_num
        gat_args.heads = 4

        agent = gat_args.agent_class(
            gat_args.num_features,
            gat_args.num_embeddings,
            gat_args.action_dim,
            gat_args.heads,
            gat_args.bat_index,
            gat_args.num_nodes,
            gpu_id=gat_args.gpu_id,
            args=gat_args,
        )
        agent.save_or_load_agent(cwd=args_from_cmd.gat_ckpt, if_save=False)
        agent.act.bat_index = torch.tensor(gat_args.bat_index, dtype=torch.long)
        agent.act.non_bat_index = torch.tensor(
            [i for i in range(gat_args.num_nodes) if i not in gat_args.bat_index], dtype=torch.long
        )
        agent.act.num_nodes = gat_args.num_nodes

    start_time = time.time()
    episode_cost = 0.0
    episode_saved_cost = 0.0
    episode_battery_profit = 0.0
    voltage_violation_count = 0
    total_voltage_violation = 0.0

    for _ in range(96):
        if model_name == "td3":
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=agent.device)
            action = agent.act.get_action(state_tensor.unsqueeze(0)).squeeze(0)
        else:
            node_feature = env_copy._state_to_node_features(state).to(agent.device)
            action = agent.act.get_action(node_feature, agent.edge_index).squeeze(0)

        ary_action = action.detach().cpu().numpy()
        next_state, reward, done, _ = env_copy.step(ary_action)
        episode_saved_cost += env_copy.saved_cost

        for node in env_copy.battery_list:
            v_after = env_copy.after_control[node]
            if v_after < 0.95 or v_after > 1.05:
                voltage_violation_count += 1
                total_voltage_violation += abs(1.0 - v_after)

        state = env_copy.reset() if done else next_state

    solve_duration = time.time() - start_time

    results.append(
        {
            "model": model_name,
            "saved_cost": episode_saved_cost,
            "voltage_violation_count": voltage_violation_count,
            "total_voltage_violation": total_voltage_violation,
            "solve_duration": solve_duration,
        }
    )

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))