import gym
import gym.spaces
from gym.utils import seeding
import enum
import numpy as np
from cnf.APConfig import config


class Action(enum.Enum):
    Idle = 0
    Buy = 1
    Sell = 2


class DayAction:
    def __init__(self, action_index=0):
        action_index = int(action_index)

        n_stocks = len(config.choices)
        self.actions = np.zeros(n_stocks, dtype=Action)
        n_acts = len(Action)
        x = action_index
        for idx in range(0, n_stocks):
            act = x % n_acts
            self.actions[idx] = Action(act)
            x = int(x / n_acts)

    @classmethod
    def action_size(cls):
        return len(Action) ** len(config.choices)


# 주식과 현금 소유 정보 관리
# 거래 단위 총합은 len(config.choices)+1 개이고 1개씩 거래할수 있다.
# amounts, values 리스트의 마지막 항목이 현금이다.
class Assets:
    def __init__(self, cash=100.0):
        n_stocks = len(config.choices)
        self.amounts = np.zeros((n_stocks + 1,), dtype=int)
        self.values = np.zeros((n_stocks + 1,), dtype=float)

        self.set_cash(cash, n_stocks)

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

        # 구매를 먼저 처리한다.
        for idx, act in enumerate(action.actions):
            c_amt, c_val = self.get_cash()
            if Action.Buy == act and c_amt > 0:
                payment = c_val / c_amt
                self.set_cash(c_val - payment, c_amt - 1)
                # 수수료를 뺀 금액을 주식 가치로 추가
                self.amounts[idx] += 1
                self.values[idx] += payment * (1 - config.commission_rate)

        for idx, act in enumerate(action.actions):
            c_amt, c_val = self.get_cash()
            if Action.Sell == act and self.amounts[idx] > 0:
                payment = self.values[idx] / self.amounts[idx]

                # 수수료를 뺀 금액을 현금으로 추가
                self.set_cash(c_val + payment * (1 - config.commission_rate), c_amt + 1)

                self.amounts[idx] -= 1
                self.values[idx] -= payment


class StateD:
    offset: int

    def __init__(self):
        self.bars_count = config.bars_count
        self.assets = None
        self.prices_list = None
        self.offset = 0
        self.days = 0

    def reset(self, prices_list, offset):
        self.assets = Assets()
        self.prices_list = prices_list
        self.offset = offset
        self.days = 0

    def apply_prices(self):
        n_stocks = len(self.prices_list)
        for stock_idx in range(0, n_stocks):
            self.assets.values[stock_idx] *= 1 + self.prices_list[stock_idx].close[self.offset]

    @property
    def shape(self):
        # choices+1, [h, l, c, v] * bars_count
        return len(config.choices) + 1, 4 * self.bars_count

    def encode(self):
        """
        Convert current state into numpy array.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)

        begin_idx = self.offset - self.bars_count + 1
        n_stocks = len(self.prices_list)
        for stock_idx in range(0, n_stocks):
            cnt = 0
            res[stock_idx][cnt * self.bars_count:self.bars_count * (cnt + 1)] = \
                self.prices_list[stock_idx].high[begin_idx:self.offset + 1]
            cnt += 1
            res[stock_idx][cnt * self.bars_count:self.bars_count * (cnt + 1)] = \
                self.prices_list[stock_idx].low[begin_idx:self.offset + 1]
            cnt += 1
            res[stock_idx][cnt * self.bars_count:self.bars_count * (cnt + 1)] = \
                self.prices_list[stock_idx].close[begin_idx:self.offset + 1]
            cnt += 1
            res[stock_idx][cnt * self.bars_count:self.bars_count * (cnt + 1)] = \
                self.prices_list[stock_idx].volume[begin_idx:self.offset + 1]

        res[n_stocks][:] = 0
        idx = 0
        n_amounts = len(self.assets.amounts)
        res[n_stocks][idx:idx + n_amounts] = self.assets.amounts

        idx += n_amounts
        n_values = len(self.assets.values)
        res[n_stocks][idx:idx + n_values] = self.assets.values

        return res

    def step(self, action):
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param action:
        :return: reward, done
        """
        assert isinstance(action, DayAction)

        n_stocks = len(self.prices_list)

        # 액션을 취하기 전의 가치
        prev_asset = self.assets.total_value()

        done = False

        # 휴일에 대한 액션은 취하지 않게 한다.
        for idx in range(0, n_stocks):
            if not self.prices_list[idx].work[self.offset]:
                action.actions[idx] = Action.Idle

        self.assets.apply_action(action)

        self.apply_prices()

        # 액션을 취하고, 금일 주가 변동액이 반영된 후의 가치
        cur_asset = self.assets.total_value()

        competetion_reward = 0.0
        first_day = self.offset-self.days
        for idx in range(0, n_stocks):
            prices = self.prices_list[idx]
            # 그동안 변동 비율에 오늘 일자의 변동 비율을 곱한다.
            competetion_reward += ((prices.open[self.offset]-prices.open[first_day])/prices.open[first_day]) * prices.close[self.offset] * 100.0 / n_stocks
        reward = (cur_asset - prev_asset) - competetion_reward

        self.offset += 1
        self.days += 1

        if self.days >= config.play_days:
            done = True

        return reward, done


class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices_list):
        self._prices_list = prices_list

        self._state = StateD()
        self.action_space = gym.spaces.Discrete(n=DayAction.action_size())
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)
        self.seed()

    def reset(self):
        # make selection of the instrument and it's offset. Then reset the state
        bars = config.bars_count
        offset = self.np_random.choice(self._prices_list[0].high.shape[0] - bars - config.play_days) + bars
        self._state.reset(self._prices_list, offset)
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

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    #@classmethod
    #def from_dir(cls, data_dir, **kwargs):
    #    prices = {file: data.load_relative(file) for file in data.price_files(data_dir)}
    #    return StocksEnv(prices, **kwargs)
