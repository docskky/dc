import gym
import gym.spaces
from gym.utils import seeding
import enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.environ import StateS

from cnf.APConfig import pconfig


def seed(seed=None):
    np_random, seed1 = seeding.np_random(seed)
    seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
    return [np_random, seed1, seed2]

"""
class PredAction(enum.Enum):
    BigDown = 0
    Down = 1
    Up = 2
    BigUp = 3
    
    
pred_action_limits = [-0.05, 0.0, 0.05]
"""


class PredAction(enum.Enum):
    BigDown = 0
    Down = 1
    Idle = 2
    Up = 3
    BigUp = 4


pred_action_limits = [-0.05, -0.01, 0.01, 0.05]


def action_for_netprice(netprice: float):
    for v in range(0, len(PredAction)-1):
        if netprice < pred_action_limits[v]:
            return PredAction(v)
    return PredAction.BigUp


# Generates rewards in range -0.25 or 1
def reward_for_netprice(netprice: float, action: PredAction):
    exp_action = action_for_netprice(netprice)
    if exp_action == action:
        return 1
    else:
        return -1.0 / (len(PredAction)-1.0)


# Calculate real close price for the offset
def close_price(prices, offset):
    return prices.open[offset] * (1.0 + prices.close[offset])


class PredState(StateS):
    offset: int
    predict_days: int
    days: int

    # predict_days: 예측 기간
    def __init__(self, predict_days=7):
        super().__init__()

        self.normal_bars_count = pconfig.normal_bars_count
        self.long_bars_count = pconfig.long_bars_count
        self.prices = None
        self.offset = 0
        self.days = 0
        self.predict_days = predict_days

    def reset(self, prices, offset):
        self.prices = prices
        self.offset = offset
        self.days = 0

    def step(self, action):
        assert isinstance(action, PredAction)

        reward = 0.0
        done = False

        # 휴일에 대한 액션은 취하지 않게 한다.
        if self.prices.work[self.offset]:
            open_price = self.prices.open[self.offset]
            future_price = close_price(self.prices, self.offset+self.predict_days)
            reward = reward_for_netprice((future_price-open_price) / open_price, action)

        # 게임 종료 여부 체크
        if self.days + 1 >= pconfig.play_days \
                or self.offset+self.predict_days + 1 >= self.prices.close.shape[0] - 1:
            done = True

        self.offset += 1
        self.days += 1

        return reward, done


class PredEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices_list, predict_days):
        self._prices_list = prices_list
        self.stock_idx = 0
        self._state = PredState(predict_days)
        self.action_space = gym.spaces.Discrete(n=len(PredAction))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)
        self.np_random = seed()[0]

    def reset(self, stock_index=None, offset=None):
        # make selection of the instrument and it's offset. Then reset the state
        _offset = 0
        if stock_index and offset:
            self.stock_idx = stock_index
            _offset = offset
        else:
            bars = pconfig.bars_count
            _offset = self.np_random.choice(self._prices_list[0].high.shape[0] - bars - bars) + bars
            self.stock_idx = self.np_random.choice(len(self._prices_list))

        self._state.reset(self._prices_list[self.stock_idx], _offset)
        return self._state.encode()

    def step(self, action_idx):
        action = PredAction(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {
            "offset": self._state.offset
        }
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

