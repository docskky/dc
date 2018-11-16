import gym
import gym.spaces
from gym.utils import seeding
import enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cnf.APConfig import sconfig
from cnf.APConfig import mconfig


def seed(seed=None):
    np_random, seed1 = seeding.np_random(seed)
    seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
    return [np_random, seed1, seed2]


class PredAction(enum.Enum):
    Idle = 0
    Up = 1
    BigUp = 2
    Down = 3
    BigDown = 4


def action_for_netprice(netprice: float):
    if netprice > 0.05:
        return PredAction.BigUp
    elif netprice > 0.01:
        return PredAction.Up
    elif netprice > -0.01:
        return PredAction.Idle
    elif netprice > -0.05:
        return PredAction.Down
    else:
        return PredAction.BigDown


def close_price(prices, offset):
    """
    Calculate real close price for the offset
    """
    return prices.open[offset] * (1.0 + prices.close[offset])


class PredState:
    offset: int
    term: int
    days: int

    # term: 예측 기간
    def __init__(self, term=7):
        self.normal_bars_count = sconfig.normal_bars_count
        self.long_bars_count = sconfig.long_bars_count
        self.prices = None
        self.offset = 0
        self.days = 0
        self.term = term

    def reset(self, prices, offset):
        self.prices = prices
        self.offset = offset
        self.days = 0

    @property
    def shape(self):
        # choices+1, [h, l, c, v] * bars_count
        return 5 * (self.normal_bars_count+self.long_bars_count) + 1 + 1,

    def encode(self):
        """
        Convert current state into numpy array.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)

        # bars_count 기간 이전 부터는 기간이 멀수록 각 바의 범위가 더 넓어진다.
        shift = 0
        compress_factor = 1.0
        hist_idx = -self.normal_bars_count
        bar_size = 2
        epsilon = 1.08
        res_size = self.long_bars_count * 5
        for bar_idx in range(-self.normal_bars_count, -self.normal_bars_count - self.long_bars_count, -1):
            if self.offset + bar_idx - bar_size < 0:
                res[res_size-shift-1-4: res_size-shift] = 0.
            else:
                res[res_size-shift-1-4] = 0
                res[res_size-shift-1-3] = 0
                res[res_size-shift-1-2] = 0
                open_price = self.prices.open[self.offset + bar_idx - 1]
                res[res_size-shift-1-1] = (open_price - self.prices.open[self.offset + bar_idx - bar_size]) / open_price + self.prices.close[self.offset + bar_idx]
                res[res_size-shift-1] = self.prices.volume[self.offset + bar_idx - bar_size: self.offset + bar_idx].mean()
            shift += 5

            compress_factor *= epsilon
            hist_idx -= bar_size
            bar_size = int(compress_factor * 2)

        # bars_count 기간의 값을 할당
        for bar_idx in range(-self.normal_bars_count + 1, 1):
            res[shift] = 1.0 if self.prices.work[self.offset + bar_idx] else 0.0
            shift += 1
            res[shift] = self.prices.high[self.offset + bar_idx]
            shift += 1
            res[shift] = self.prices.low[self.offset + bar_idx]
            shift += 1
            res[shift] = self.prices.close[self.offset + bar_idx]
            shift += 1
            res[shift] = self.prices.volume[self.offset + bar_idx]
            shift += 1
        res[shift] = float(self.has_position)
        shift += 1
        if self.has_position:
            res[shift] = (self._cur_close() - self.open_price) / self.open_price
        else:
            res[shift] = 0.0

        #print("================================================================")
        #np.set_printoptions(precision=2, linewidth = 60)
        #print(res)
        return res

    def step(self, action):
        assert isinstance(action, PredAction)

        reward = 0.0
        done = False
        close = self._cur_close()

        # 휴일에 대한 액션은 취하지 않게 한다.
        if self.prices.work[self.offset]:
            future_price = close_price(self.prices, self.offset+self.term)
            if action == PredAction.BigUp:
                reward = (future_price-self.prices.open[self.offset])
            elif action == PredAction.Sell and self.has_position:
                reward -= reward * (sconfig.commission_rate + sconfig.sale_tax_rate)
                done = True
                reward += 100.0 * (close - self.open_price) / self.open_price
                self.has_position = False
                self.open_price = 0.0

        # 게임 종료 여부 체크
        if self.days + 1 >= sconfig.max_play_days \
                or self.offset+self.term + 1 >= self.prices.close.shape[0] - 1:
            done = True

        self.offset += 1
        self.days += 1
        # prev_close = close
        # close = self._cur_close()

        # if self.have_position and not self.reward_on_close:
        #    reward += 100.0 * (close - prev_close) / prev_close

        return reward, done

    def _cur_close(self):
        """
        Calculate real close price for the current bar
        """
        open = self.prices.open[self.offset]
        rel_close = self.prices.close[self.offset]
        return open * (1.0 + rel_close)



class StocksEnvS(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices_list):
        self._prices_list = prices_list
        self.stock_idx = 0
        self._state = StateS()
        self.action_space = gym.spaces.Discrete(n=len(Action))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)
        self.np_random = seed()[0]

    def reset(self, stock_index=None, offset=None):
        # make selection of the instrument and it's offset. Then reset the state
        _offset = 0
        if stock_index and offset:
            self.stock_idx = stock_index
            _offset = offset
        else:
            bars = sconfig.bars_count
            _offset = self.np_random.choice(self._prices_list[0].high.shape[0] - bars - bars) + bars
            self.stock_idx = self.np_random.choice(len(self._prices_list))

        self._state.reset(self._prices_list[self.stock_idx], offset)
        return self._state.encode()

    def step(self, action_idx):
        action = Action(action_idx)
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

