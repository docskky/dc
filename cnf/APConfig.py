import json


# bars_count : 바라보는 일자수
# commission_rate : 결제수수료 비율

class APConfig:
    env = None

    def __init__(self, phase):
        with open('./cnf/config.json') as file:
            self.env = json.load(file)
        assert (self.env is not None)
        self.version = self.env["version"]
        self.db_host = self.env["db_host"]
        self.db_user = self.env["db_user"]
        self.db_pw = self.env["db_pw"]
        self.db_name = self.env["db_name"]
        self.bars_count = self.env["bars_count"]
        self.commission_rate = self.env["commission_rate"]
        self.sale_tax_rate = self.env["sale_tax_rate"]

        self.bar_download_limit = self.env["bar_download_limit"]
        self.choices = self.env["choices"]
        self.play_days = self.env["play_days"]

        self.batch_size = self.env["batch_size"]
        self.target_net_sync = self.env["target_net_sync"]
        self.gamma = self.env["gamma"]
        self.replay_size = self.env["replay_size"]
        self.replay_initial = self.env["replay_initial"]
        self.reward_steps = self.env["reward_steps"]
        self.learning_rate = self.env["learning_rate"]
        self.states_to_evaluate = self.env["states_to_evaluate"]
        self.eval_every_step = self.env["eval_every_step"]
        self.epsilon_start = self.env["epsilon_start"]
        self.epsilon_stop = self.env["epsilon_stop"]
        self.epsilon_steps = self.env["epsilon_steps"]
        self.checkpoint_every_step = self.env["checkpoint_every_step"]
        self.validation_every_step = self.env["validation_every_step"]
        self.run_name = self.env["run_name"]

        self.max_play_days = self.env["max_play_days"]


config = APConfig()
