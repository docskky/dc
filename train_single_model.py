#!/usr/bin/env python3
import os
import ptan
import numpy as np

import torch
import torch.optim as optim

from cnf.APConfig import sconfig as sconfig
import lib.environ as environ
import lib.models as models
import lib.data as data
import lib.common as common
import lib.validation as validation
import argparse
# import datetime

from tensorboardX import SummaryWriter

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-r", "--run", default=sconfig.run_name, help="Run name")
    # parser.add_argument("--data", default=DEFAULT_STOCKS, help="Stocks file or dir to train on, default=" + DEFAULT_STOCKS)
    # parser.add_argument("--year", type=int, help="Year to be used for training, if specified, overrides --data option")
    # parser.add_argument("--valdata", default=DEFAULT_VAL_STOCKS, help="Stocks data for validation, default=" + DEFAULT_VAL_STOCKS)
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    saves_path = os.path.join("saves", sconfig.run_name)
    os.makedirs(saves_path, exist_ok=True)

    prices_list, val_prices_list = data.load_prices(sconfig.choices)

    stock_env = environ.StocksEnvS(prices_list)
    val_stock_env = environ.StocksEnvS(val_prices_list)

    writer = SummaryWriter(comment="-single-" + args.run)
    net = models.SimpleFFDQN(stock_env.observation_space.shape[0], stock_env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(sconfig.epsilon_start)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(stock_env, agent, sconfig.gamma, steps_count=sconfig.reward_steps)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, sconfig.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=sconfig.learning_rate)

    # main training loop
    step_idx = 0
    eval_states = None
    best_mean_val = None

    with common.RewardTracker(writer, np.inf, group_rewards=100) as reward_tracker:
        while True:
            step_idx += 1
            buffer.populate(1)
            selector.epsilon = max(sconfig.epsilon_stop, sconfig.epsilon_start - step_idx / sconfig.epsilon_steps)

            new_rewards = exp_source.pop_rewards_steps()
            if new_rewards:
                reward_tracker.reward(new_rewards[0], step_idx, selector.epsilon)

            if len(buffer) < sconfig.replay_initial:
                continue

            if eval_states is None:
                print("Initial buffer populated, start training")
                eval_states = buffer.sample(sconfig.states_to_evaluate)
                eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
                eval_states = np.array(eval_states, copy=False)

            if step_idx % sconfig.eval_every_step == 0:
                mean_val = common.calc_values_of_states(eval_states, net, device=device)
                writer.add_scalar("values_mean", mean_val, step_idx)
                if best_mean_val is None or best_mean_val < mean_val:
                    if best_mean_val is not None:
                        print("%d: Best mean value updated %.3f -> %.3f" % (step_idx, best_mean_val, mean_val))
                    best_mean_val = mean_val
                    torch.save(net.state_dict(), os.path.join(saves_path, "mean_val-%.3f.data" % mean_val))

            optimizer.zero_grad()
            batch = buffer.sample(sconfig.batch_size)
            loss_v = common.calc_loss(batch, net, tgt_net.target_model, sconfig.gamma ** sconfig.reward_steps, device=device)
            loss_v.backward()
            optimizer.step()

            if step_idx % sconfig.target_net_sync == 0:
                tgt_net.sync()

            if step_idx % sconfig.checkpoint_every_step == 0:
                idx = step_idx // sconfig.checkpoint_every_step
                torch.save(net.state_dict(), os.path.join(saves_path, "checkpoint-%3d.data" % idx))

            if step_idx % sconfig.validation_every_step == 0:
                res = validation.validation_run(stock_env, net, device=device)
                for key, val in res.items():
                    writer.add_scalar(key + "_test", val, step_idx)
                res = validation.validation_run(val_stock_env, net, device=device)
                for key, val in res.items():
                    writer.add_scalar(key + "_val", val, step_idx)
