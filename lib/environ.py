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


def apply_model_from_state(data, net):
    state_v = torch.tensor(data)
    values_v = net(state_v)
    return values_v.detach().numpy()


class Action(enum.Enum):
    Idle = 0
    Buy = 1
    Sell = 2


class DayAction:
    positions: [int]

    def __init__(self, action_index=0):
        action_index = int(action_index)

        choice_size = len(mconfig.choices)+1
        n_positions = mconfig.position_limit
        self.positions = np.zeros(n_positions, dtype=int)

        x = action_index
        for idx in range(0, n_positions):
            pos = x % choice_size
            self.positions[idx] = pos
            x = int(x / choice_size)

    @classmethod
    def action_size(cls):
        return (len(mconfig.choices)+1) * mconfig.position_limit

    # 액션에 의해 반영된 수량 리스트 계산
    def position_amounts(self):
        choice_size = len(mconfig.choices)+1
        amounts = np.zeros(choice_size, dtype=int)
        for pos in self.positions:
            amounts[pos] += 1

        return amounts


# 주식과 현금 소유 정보 관리
# 거래 단위 총합은 len(config.choices)+1 개이고 1개씩 거래할수 있다.
# amounts, values 리스트의 마지막 항목이 현금이다.
class Assets:
    def __init__(self):
        n_stocks = len(mconfig.choices)
        self.amounts = np.zeros((n_stocks + 1,), dtype=int)
        self.values = np.zeros((n_stocks + 1,), dtype=float)

        self.reset()

    def reset(self):
        n_positions = mconfig.position_limit
        self.amounts[:] = 0
        self.values[:] = 0.0
        self.set_cash(100, n_positions)

    def set_cash(self, cash, amount):
        self.amounts[-1] = amount
        self.values[-1] = cash

    def get_cash(self):
        return self.amounts[-1], self.values[-1]

    def total_value(self):
        return self.values.sum()

    # 액션을 적용한다.
    def apply_action(self, action):
        assert isinstance(action, DayAction)

        # 판매를 모두 처리하고 구매를 처리한다.
        new_amounts = action.position_amounts()
        for sell_mode in [False, True]:
            for idx in range(0, len(self.amounts)-1):
                net_amt = new_amounts[idx] - self.amounts[idx]
                c_amt, c_val = self.get_cash()
                # 판매
                if sell_mode and net_amt < 0:
                    payment = -net_amt * self.values[idx] / self.amounts[idx]

                    # 수수료+세금을 뺀 금액을 현금으로 추가
                    self.set_cash(c_val + payment * (1 - mconfig.commission_rate - mconfig.sale_tax_rate), c_amt + 1)

                    self.amounts[idx] += net_amt
                    self.values[idx] -= payment

                # 구매
                if not sell_mode and net_amt > 0:
                    payment = net_amt * c_val / c_amt

                    # 수수료를 뺀 금액을 주식으로 추가
                    self.set_cash(c_val - payment, c_amt + 1)

                    self.amounts[idx] += net_amt
                    self.values[idx] += payment * (1 - mconfig.commission_rate)


class StateM:
    offset: int
    s_net: nn.Module

    def __init__(self):
        self.position_limit = mconfig.position_limit
        self.watch_size = mconfig.watch_size
        self.assets = Assets()
        self.prices_list = None
        self.offset = 0
        self.days = 0
        self.s_net = None
        self.s_states = []
        self.s_action_values = []
        for idx in range(0, len(mconfig.choices)):
            self.s_states.append(StateS())
            self.s_action_values.append(None)

    def reset(self, prices_list, offset, s_net):
        self.prices_list = prices_list
        self.offset = offset
        self.days = 0
        self.s_net = s_net
        for idx in range(0, len(prices_list)):
            self.s_states[idx].reset(prices_list[idx], offset)
            self.s_action_values[idx] = self.calculate_action_values(self.s_states[idx])
        self.assets.reset()

    def apply_prices(self):
        n_stocks = len(self.prices_list)
        for stock_idx in range(0, n_stocks):
            self.assets.values[stock_idx] *= 1 + self.prices_list[stock_idx].close[self.offset]

    def calculate_action_values(self, s_state):
        data = s_state.encode()
        state_v = torch.tensor(data)
        values_v = self.s_net(state_v)
        return values_v.detach().numpy()

    @property
    def shape(self):
        return len(self.s_action_values) * len(Action) + len(self.assets.amounts) * 2 + self.s_states[0].shape[0]*self.watch_size + 1,

    def encode(self):
        """
        Convert current state into numpy array.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for idx in range(0, len(self.s_action_values)):
            action_size = self.s_action_values[idx].shape[0]
            res[shift: shift+action_size] = self.s_action_values[idx]
            shift += action_size
        n_amounts = len(self.assets.amounts)
        res[shift: shift+n_amounts] = self.assets.amounts
        shift += n_amounts
        res[shift: shift+n_amounts] = self.assets.values
        shift += n_amounts
        res[shift] = self.days
        shift += 1

        sorted_idxs = np.array(self.s_action_values).argsort(axis=0)

        # 현재 보유하고 있는 주식의 히스토리를 encode 한다.
        n_watching = 0
        for idx, amt in enumerate(self.assets.amounts):
            if amt > 0 and idx < n_amounts-1:
                sorted_idxs[idx][0] = -1
                size = self.s_states[idx].shape[0]
                res[shift: shift+size] = self.s_states[idx].encode()
                shift += size
                n_watching += 1

        # 나머지 주식중 s_action_values 값이 높은 주식을 encode 한다.
        cnt = self.watch_size-n_watching
        idx = 0
        while cnt > 0:
            while sorted_idxs[idx][0] < 0:
                idx += 1
            s_idx = sorted_idxs[idx][0]
            size = self.s_states[s_idx].shape[0]
            res[shift: shift+size] = self.s_states[s_idx].encode()
            shift += size
            idx += 1
            cnt -= 1


        return res

    def step(self, action):
        """
        """
        assert isinstance(action, DayAction)

        n_stocks = len(self.prices_list)

        # 액션을 취하기 전의 가치
        prev_asset = self.assets.total_value()

        done = False

        # 휴일에 대한 액션은 취하지 않게 한다.
        holiday = False
        for idx in range(0, n_stocks):
            if not self.prices_list[idx].work[self.offset]:
                holiday = True
                break

        if not holiday:
            self.assets.apply_action(action)
            self.apply_prices()

        # 액션을 취하고, 금일 주가 변동액이 반영된 후의 가치
        cur_asset = self.assets.total_value()

        """
                competetion_reward = 0.0
                first_day = self.offset - self.days
                for idx in range(0, n_stocks):
                    prices = self.prices_list[idx]
                    # 그동안 변동 비율에 오늘 일자의 변동 비율을 곱한다.
                    competetion_reward += ((prices.open[self.offset] - prices.open[first_day]) / prices.open[first_day]) * 
                                          prices.close[self.offset] * 100.0 / n_stocks
                reward = (cur_asset - prev_asset) - competetion_reward
        """

        reward = cur_asset - prev_asset

        self.offset += 1
        self.days += 1

        # offset이 변경되었으므로 주식 action value를 업데이트한다.
        for idx in range(0, len(self.prices_list)):
            self.s_states[idx].step(Action(Action.Idle))
            self.s_action_values[idx] = self.calculate_action_values(self.s_states[idx])

        if self.days >= mconfig.play_days:
            done = True

        return reward, done


class StocksEnvM(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices_list, s_net):
        self._prices_list = prices_list
        self.s_net = s_net
        self._state = StateM()
        self.action_space = gym.spaces.Discrete(n=DayAction.action_size())
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)
        self.np_random = seed()[0]

    def reset(self):
        # make selection of the instrument and it's offset. Then reset the state
        bars = sconfig.bars_count
        len =self._prices_list[0].high.shape[0]
        offset = self.np_random.choice(len - bars - mconfig.play_days) + bars
        self._state.reset(self._prices_list, offset, self.s_net)
        return self._state.encode()

    def step(self, action_idx):
        action = DayAction(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {
            "offset": self._state.offset,
            "profit": self._state.assets.total_value() - 100
        }
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    # @classmethod
    # def from_dir(cls, data_dir, **kwargs):
    #    prices = {file: data.load_relative(file) for file in data.price_files(data_dir)}
    #    return StocksEnv(prices, **kwargs)


class StateS:
    offset: int

    def __init__(self):
        self.normal_bars_count = sconfig.normal_bars_count
        self.long_bars_count = sconfig.long_bars_count
        self.commission_rate = sconfig.commission_rate
        self.has_position = False
        self.prices = None
        self.offset = 0
        self.days = 0
        self.open_price = 0.0

    def reset(self, prices, offset):
        self.has_position = False
        self.prices = prices
        self.offset = offset
        self.days = 0
        self.open_price = 0.0

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
        assert isinstance(action, Action)

        reward = 0.0
        done = False
        close = self._cur_close()

        # 휴일에 대한 액션은 취하지 않게 한다.
        if not self.prices.work[self.offset]:
            action = Action.Idle

        if self.days + 1 >= sconfig.max_play_days \
                or self.offset + 1 >= self.prices.close.shape[0] - 1:
            action = Action.Sell
            done = True

        if action == Action.Buy and not self.has_position:
            self.has_position = True
            self.open_price = close
            reward -= reward * self.commission_rate
        elif action == Action.Sell and self.has_position:
            reward -= reward * (self.commission_rate + sconfig.sale_tax_rate)
            done = True
            reward += 100.0 * (close - self.open_price) / self.open_price
            self.has_position = False
            self.open_price = 0.0

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

