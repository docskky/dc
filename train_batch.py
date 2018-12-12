#!/usr/bin/env python3
import train_model
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

if __name__ == "__main__":
    sconfig.
    for pdays in [7, 14, 30]:
        train_model.train_model(cuda=True, phase=3, premodel=None, pdays=pdays)
