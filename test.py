import data.data as data
from cnf.APConfig import config

prices = data.load_prices(config.choices)

pc = prices[0]
print("shape"+pc.work.shape)