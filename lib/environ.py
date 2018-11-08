import gym
import gym.spaces
from gym.utils import seeding
import enum
import numpy as np
from cnf.APConfig import config


def seed(seed=None):
    np_random, seed1 = seeding.np_random(seed)
    seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
    return [np_random, seed1, seed2]


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
        n_stocks = len(config.choices)
        return n_stocks * 4 + (n_stocks + 1) * 2, self.bars_count

    def encode(self):
        """
        Convert current state into numpy array.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)

        begin_idx = self.offset - self.bars_count + 1
        n_stocks = len(self.prices_list)
        for stock_idx in range(0, n_stocks):
            idx = stock_idx * 4
            res[idx] = self.prices_list[stock_idx].high[begin_idx:self.offset + 1]
            res[idx + 1] = self.prices_list[stock_idx].low[begin_idx:self.offset + 1]
            res[idx + 2] = self.prices_list[stock_idx].close[begin_idx:self.offset + 1]
            res[idx + 3] = self.prices_list[stock_idx].volume[begin_idx:self.offset + 1]

        s = n_stocks * 4

        n_amounts = len(self.assets.amounts)
        for i in range(0, n_amounts):
            res[s + i] = self.assets.amounts[i]

        s += n_amounts

        for i in range(0, n_amounts):
            res[s + i] = self.assets.values[i]

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
        for idx in range(0, n_stocks):
            if not self.prices_list[idx].work[self.offset]:
                action.actions[idx] = Action.Idle

        self.assets.apply_action(action)

        self.apply_prices()

        # 액션을 취하고, 금일 주가 변동액이 반영된 후의 가치
        cur_asset = self.assets.total_value()

        competetion_reward = 0.0
        first_day = self.offset - self.days
        for idx in range(0, n_stocks):
            prices = self.prices_list[idx]
            # 그동안 변동 비율에 오늘 일자의 변동 비율을 곱한다.
            competetion_reward += ((prices.open[self.offset] - prices.open[first_day]) / prices.open[first_day]) * \
                                  prices.close[self.offset] * 100.0 / n_stocks
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
        self.np_random = seed()[0]

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

    # @classmethod
    # def from_dir(cls, data_dir, **kwargs):
    #    prices = {file: data.load_relative(file) for file in data.price_files(data_dir)}
    #    return StocksEnv(prices, **kwargs)


class StateS:
    offset: int

    def __init__(self):
        self.bars_count = config.bars_count
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
        return 4 * self.bars_count + 1 + 1,

    def encode(self):
        """
        Convert current state into numpy array.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count + 1, 1):
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
        if not self.has_position:
            res[shift] = (self._cur_close() - self.open_price) / self.open_price
        else:
            res[shift] = 0.0
        return res

    def step(self, action):
        assert isinstance(action, Action)

        reward = 0.0
        done = False
        close = self._cur_close()

        # 휴일에 대한 액션은 취하지 않게 한다.
        if not self.prices.work[self.offset]:
            action = Action.Idle

        if self.days + 1 >= config.max_play_days \
                or self.offset + 1 >= self.prices.close.shape[0] - 1:
            action = Action.Sell
            done = True

        if action == Action.Buy and not self.has_position:
            self.has_position = True
            self.open_price = close
            reward -= reward * config.commission_rate
        elif action == Action.Sell and self.has_position:
            reward -= reward * (config.commission_rate + config.sale_tax_rate)
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

    def reset(self):
        # make selection of the instrument and it's offset. Then reset the state
        bars = config.bars_count
        offset = self.np_random.choice(self._prices_list[0].high.shape[0] - bars - bars) + bars
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
