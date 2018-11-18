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

# example:


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-phase", "--phase", default="3", help="Phase[1-3]: 1-single, 2-multi, 3-prediction")
    parser.add_argument("-pdays", "--pdays", default="7", help="Predict days")
    parser.add_argument("-m", "--model", default="data/phase3.data", help="Model file to load")
    args = parser.parse_args()

    phase = int(args.phase)
    if phase == 1:
        config = sconfig
    elif phase == 2:
        config = mconfig
    elif phase == 3:
        config = pconfig

    run_name = "v" + config.version + "-phase" + str(phase)
    saves_path = os.path.join("saves", run_name)

    save_name = ""

    writer = SummaryWriter(comment=run_name)

    prices_list, val_prices_list = data.load_prices(config.choices)

    predict_days = int(args.pdays)
    stock_env = pdenviron.PredEnv(prices_list=prices_list, predict_days=7)
    net = models.SimpleFFDQN(stock_env.observation_space.shape[0], stock_env.action_space.n)  # .to(device)
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))

    done = False
    obs = stock_env.reset(0, 100)
    while not done:
        values = environ.apply_model_from_state(obs, net)
        obs, reward, done, info = stock_env.step(pdenviron.PredAction.Idle)
        print("reward:{}, values:{}".format(reward, values))