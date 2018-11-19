#!/usr/bin/env python3
import os
import ptan
import numpy as np

import torch
import torch.optim as optim

from cnf.APConfig import sconfig
from cnf.APConfig import mconfig
from cnf.APConfig import pconfig
import lib.environ as environ
import lib.pdenviron as pdenviron
import lib.models as models
import lib.data as data
import lib.common as common
import lib.validation as validation
import argparse
# import datetime

from tensorboardX import SummaryWriter


def train_model(cuda, phase, premodel, pdays):
    """
    cuda : True / False
    phase : 1~3
    premodel: data/phase1_model.data
    pdays: integer
    """
    device = torch.device("cuda" if cuda else "cpu")
    phase = int(phase)
    if phase == 1:
        config = sconfig
    elif phase == 2:
        config = mconfig
    elif phase == 3:
        config = pconfig

    run_name = "v" + config.version + "-phase" + str(phase)
    saves_path = os.path.join("saves", run_name)
    os.makedirs(saves_path, exist_ok=True)

    save_name = ""

    writer = SummaryWriter(comment=run_name)

    prices_list, val_prices_list = data.load_prices(config.choices)

    if phase == 1:
        s_env = environ.StocksEnvS(prices_list)
        stock_env = s_env
        val_stock_env = environ.StocksEnvS(val_prices_list)
        save_name = "{}.data".format(run_name)
    elif phase == 2:
        # phase 1 의 network 그래프를 로드한다.
        s_env = environ.StocksEnvS(prices_list)
        prenet = models.SimpleFFDQN(s_env.observation_space.shape[0], s_env.action_space.n) #.to(device)
        models.load_model(premodel, prenet)

        # phase2 환경 생성
        stock_env = environ.StocksEnvM(prices_list, prenet)
        val_stock_env = environ.StocksEnvM(val_prices_list, prenet)
        save_name = "{}.data".format(run_name)
    elif phase == 3:
        predict_days = int(pdays)
        stock_env = pdenviron.PredEnv(prices_list=prices_list, predict_days=7)
        val_stock_env = pdenviron.PredEnv(prices_list=prices_list, predict_days=7)
        save_name = "{}-{}.data".format(run_name, predict_days)

    net = models.SimpleFFDQN(stock_env.observation_space.shape[0], stock_env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(config.epsilon_start)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(stock_env, agent, config.gamma,
                                                           steps_count=config.reward_steps)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, config.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)

    # main training loop
    step_idx = 0
    eval_states = None
    best_mean_val = None

    with common.RewardTracker(writer, np.inf, group_rewards=100) as reward_tracker:
        while step_idx < config.end_step:
            step_idx += 1
            buffer.populate(1)
            selector.epsilon = max(config.epsilon_stop, config.epsilon_start - step_idx / config.epsilon_steps)

            new_rewards = exp_source.pop_rewards_steps()
            if new_rewards:
                reward_tracker.reward(new_rewards[0], step_idx, selector.epsilon)

            if len(buffer) < config.replay_initial:
                continue

            if eval_states is None:
                print("Initial buffer populated, start training")
                eval_states = buffer.sample(config.states_to_evaluate)
                eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
                eval_states = np.array(eval_states, copy=False)

            if step_idx % config.eval_every_step == 0:
                mean_val = common.calc_values_of_states(eval_states, net, device=device)
                writer.add_scalar("values_mean", mean_val, step_idx)
                if best_mean_val is None or best_mean_val < mean_val:
                    if best_mean_val is not None:
                        print("%d: Best mean value updated %.3f -> %.3f" % (step_idx, best_mean_val, mean_val))
                    best_mean_val = mean_val
                    #torch.save(net.state_dict(), os.path.join(saves_path, "mean_val-%.3f.data" % mean_val))

            optimizer.zero_grad()
            batch = buffer.sample(config.batch_size)
            loss_v = common.calc_loss(batch, net, tgt_net.target_model, config.gamma ** config.reward_steps,
                                      device=device)
            loss_v.backward()
            optimizer.step()

            if step_idx % config.target_net_sync == 0:
                tgt_net.sync()

            if step_idx % config.checkpoint_every_step == 0:
                idx = step_idx // config.checkpoint_every_step
                torch.save(net.state_dict(), os.path.join(saves_path, "checkpoint-%d.data" % idx))

            if step_idx % config.validation_every_step == 0:
                res = validation.validation_run(stock_env, net, device=device)
                for key, val in res.items():
                    writer.add_scalar(key + "_test", val, step_idx)
                res = validation.validation_run(val_stock_env, net, device=device)
                for key, val in res.items():
                    writer.add_scalar(key + "_val", val, step_idx)

        models.save_model(os.path.join(saves_path, save_name), net, {})


# example:
# $python train_model.py --cuda --phase 1
# $python train_model.py --cuda --phase 2
# $python train_model.py --cuda --phase 2 --premodel data/phase1_model.data
# $python train_model.py --cuda --phase 3 --pdays 14
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-phase", "--phase", default="3", help="Phase[1-3]: 1-single, 2-multi, 3-prediction")
    parser.add_argument("-pdays", "--pdays", default="7", help="Predict days")

    # phase 2인 경우 적용
    parser.add_argument("-pm", "--premodel", default="data/phase1_model.data", help="Model file to load")
    args = parser.parse_args()

    train_model(args.cuda, args.phase, args.premodel, args.pdays)
