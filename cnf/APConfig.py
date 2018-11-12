import json
from configparser import ConfigParser

import io

# bars_count : 바라보는 일자수
# commission_rate : 결제수수료 비율


class APConfig:
    env = None

    def __init__(self):
        self.env = ConfigParser()
        self.env.read('./cnf/config.ini', 'utf-8')

        assert (self.env is not None)

        self.version = self.env["global"]["version"]
        self.db_host = self.env["global"]["db_host"]
        self.db_user = self.env["global"]["db_user"]
        self.db_pw = self.env["global"]["db_pw"]
        self.db_name = self.env["global"]["db_name"]

        self.commission_rate = float(self.env["global"]["commission_rate"])
        self.sale_tax_rate = float(self.env["global"]["sale_tax_rate"])

        self.history_begin = self.env["global"]["history_begin"]

        self.bar_download_limit = int(self.env["global"]["bar_download_limit"])


class SinglePhaseConfig(APConfig):
    def __init__(self):
        super(SinglePhaseConfig, self).__init__()

        penv = self.env["single_phase"]
        assert (penv is not None)

        self.normal_bars_count = int(penv["normal_bars_count"])
        self.long_bars_count = int(penv["long_bars_count"])
        self.bars_count = self.normal_bars_count + self.long_bars_count
        self.choices = [e.strip() for e in penv["choices"].split(',')]

        self.play_days = int(penv["play_days"])

        self.batch_size = int(penv["batch_size"])
        self.target_net_sync = int(penv["target_net_sync"])
        self.gamma = float(penv["gamma"])
        self.replay_size = int(penv["replay_size"])
        self.replay_initial = int(penv["replay_initial"])
        self.reward_steps = int(penv["reward_steps"])
        self.learning_rate = float(penv["learning_rate"])
        self.states_to_evaluate = int(penv["states_to_evaluate"])
        self.eval_every_step = int(penv["eval_every_step"])
        self.epsilon_start = float(penv["epsilon_start"])
        self.epsilon_stop = float(penv["epsilon_stop"])
        self.epsilon_steps = int(penv["epsilon_steps"])
        self.checkpoint_every_step = int(penv["checkpoint_every_step"])
        self.validation_every_step = int(penv["validation_every_step"])
        self.run_name = penv["run_name"]

        self.max_play_days = int(penv["max_play_days"])


class MultiPhaseConfig(APConfig):
    def __init__(self):
        super(MultiPhaseConfig, self).__init__()

        penv = self.env["multi_phase"]
        assert (penv is not None)

        self.choices = [e.strip() for e in penv["choices"].split(',')]
        self.position_limit = int(penv["position_limit"])
        self.watch_size = int(penv["watch_size"])
        self.play_days = int(penv["play_days"])

        self.batch_size = int(penv["batch_size"])
        self.target_net_sync = int(penv["target_net_sync"])
        self.gamma = float(penv["gamma"])
        self.replay_size = int(penv["replay_size"])
        self.replay_initial = int(penv["replay_initial"])
        self.reward_steps = int(penv["reward_steps"])
        self.learning_rate = float(penv["learning_rate"])
        self.states_to_evaluate = int(penv["states_to_evaluate"])
        self.eval_every_step = int(penv["eval_every_step"])
        self.epsilon_start = float(penv["epsilon_start"])
        self.epsilon_stop = float(penv["epsilon_stop"])
        self.epsilon_steps = int(penv["epsilon_steps"])
        self.checkpoint_every_step = int(penv["checkpoint_every_step"])
        self.validation_every_step = int(penv["validation_every_step"])
        self.run_name = penv["run_name"]

        #self.max_play_days = penv["max_play_days"]


sconfig = SinglePhaseConfig()

mconfig = MultiPhaseConfig()
