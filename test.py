from env import data as data
from cnf.APConfig import config

prices = data.load_prices(config.choices, 1)

pc = prices[0]
print("shape {}".format(pc))