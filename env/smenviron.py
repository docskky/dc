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
    def __init__(self, actions, positions):
        self.actions = actions
        self.positions = positions

class State:
    def __init__(self):
        self.bars_count = config.bars_count
        self.commission_percent = config.commission_percent
        self.own_cash = 0
        self.own_stocks = None
        self.prices_list = None
        self.offset = None
        self.days = 0

    def reset(self, prices_list, offset):
        self.own_cash = config.init_caches
        self.own_stocks = np.zeros((len(config.choices),), dtype=int)
        self.prices_list = prices_list
        self.offset = offset
        self.days = 0

    def _total_asset(self):
        self.own_cash + self.own_stocks.sum()

    @property
    def shape(self):
        # choices+1, [h, l, c, v] * bars_count
        return len(config.choices)+1, 4 * self.bars_count

    def encode(self):
        """
        Convert current state into numpy array.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)

        begin_idx = self.offset-self.bars_count+1
        n_stocks = len(self.prices_list)
        for stock_idx in range(0, n_stocks):
            cnt = 0
            res[stock_idx][cnt*self.bars_count:self.bars_count*(cnt+1)] = self.prices_list[stock_idx].high[begin_idx:self.offset+1]
            cnt += 1
            res[stock_idx][cnt*self.bars_count:self.bars_count*(cnt+1)] = self.prices_list[stock_idx].low[begin_idx:self.offset+1]
            cnt += 1
            res[stock_idx][cnt*self.bars_count:self.bars_count*(cnt+1)] = self.prices_list[stock_idx].close[begin_idx:self.offset+1]
            cnt += 1
            res[stock_idx][cnt*self.bars_count:self.bars_count*(cnt+1)] = self.prices_list[stock_idx].volume[begin_idx:self.offset+1]

        res[n_stocks][:] = 0
        idx = 0
        res[n_stocks][idx] = self.own_cash
        idx += 1
        res[n_stocks][idx:idx+len(self.own_stocks)] = self.own_stocks

        return res

    def step(self, action):
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param action:
        :return: reward, done
        """
        assert isinstance(action, DayAction)

        prev_asset = self._total_asset()

        reward = 0.0
        done = False
        close = self._cur_close()
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close
            reward -= self.commission_perc

        self.offset += 1
        self.days += 1

        if self.days >= config.play_days:
            done = True

        if self.have_position and not self.reward_on_close:
            reward += 100.0 * (close - prev_close) / prev_close

        cur_asset = self._total_asset()

        reward = cur_asset-
        return reward, done



class State1D(State):
    """
    State with shape suitable for 1D convolution
    """
    @property
    def shape(self):
        return (6, self.bars_count)
    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)
        ofs = self.bars_count-1
        res[0] = self._prices.high[self._offset-ofs:self._offset+1]
        res[1] = self._prices.low[self._offset-ofs:self._offset+1]
        res[2] = self._prices.close[self._offset-ofs:self._offset+1]
        res[3] = self._prices.volume[self._offset-ofs:self._offset+1]
        dst = 4
        if self.have_position:
            res[dst] = 1.0
            res[dst+1] = (self._cur_close() - self.open_price) / self.open_price
        return res


class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices, bars_count=DEFAULT_BARS_COUNT,
                 commission=DEFAULT_COMMISSION_PERC, reset_on_close=True, state_1d=False,
                 random_ofs_on_reset=True, reward_on_close=False, volumes=False):
        assert isinstance(prices, dict)
        self._prices = prices
        if state_1d:
            self._state = State1D(bars_count, commission, reset_on_close, reward_on_close=reward_on_close,
                                  volumes=volumes)
        else:
            self._state = State(bars_count, commission, reset_on_close, reward_on_close=reward_on_close,
                                volumes=volumes)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset
        self.seed()

    def reset(self):
        # make selection of the instrument and it's offset. Then reset the state
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = self.np_random.choice(prices.high.shape[0]-bars*10) + bars
        else:
            offset = bars
        self._state.reset(prices, offset)
        return self._state.encode()

    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {"instrument": self._instrument, "offset": self._state._offset}
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    @classmethod
    def from_dir(cls, data_dir, **kwargs):
        prices = {file: data.load_relative(file) for file in data.price_files(data_dir)}
        return StocksEnv(prices, **kwargs)
