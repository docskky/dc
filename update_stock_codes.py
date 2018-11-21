import download.stockcode as stockcode
from lib.apdb import APDatabase as apdb
from cnf.APConfig import sconfig as config
import datetime
from pandas_datareader import data
import fix_yahoo_finance as yf
yf.pdr_override()
from contextlib import closing
import time

#!/usr/bin/env python3
import os
import ptan
import numpy as np

import torch
import lib.environ as environ
import lib.pdenviron as pdenviron
import lib.models as models
import lib.data as data


"""
어제 일자까지의 모든 주가를 업데이트 한다.
"""
stmt1 = "insert IGNORE into stocks (scode, name) VALUES (%s,%s);"
stmt2 = "select max(date) from history_days where scode=%s;"
stmt3 = "insert IGNORE into history_days (scode, date, open, high, low, close, volume) VALUES (%s,%s, %s, %s,%s,%s,%s);"
date_format = '%Y-%m-%d'


def update_stocks(db, df, ext):
    for index, row in df.iterrows():
        code = row['종목코드']
        scode = ext+code
        name = row['회사명']

        done = False
        count = 0
        while not done:
            try:
                with closing(db.cursor()) as cur:
                    cur.execute(stmt1, (scode, name))
                    db.commit()

                    # 마지막 저장일을 검색한다.
                    cur.execute(stmt2, (scode, ))
                    row = cur.fetchone()
                    last_date = row[0]
                    if last_date:
                        if last_date == datetime.datetime.now().date():
                            return
                    else:
                        from_date = config.history_begin

                    # 마지막 저장일 이후의 주식변동을 검색한다.
                    df = data.get_data_yahoo(code + '.'+ext, from_date, thread=20)
                    for i, row in df.iterrows():
                        date = row.name.strftime(date_format)
                        open = row['Open']
                        high = row['High']
                        low = row['Low']
                        close = row['Close']
                        volume = row['Volume']
                        cur.execute(stmt3, (scode, date, open, high, low, close, volume))
                        db.commit()
                    done = True
            except Exception as ex:
                time.sleep(10)
                count += 1
                if count >= 20:
                    done = True
                print(ex)


def update_prediction():

    prices_list, val_prices_list = data.load_prices()

    # 트래이닝 된 모델을 로드한다.
    predict_days = [7, 14, 30]
    nets = []
    envs = []
    for pdays in predict_days:
        file_path = "data/v3.0-phase3-{}.data".format(pdays)
        env = pdenviron.PredEnv(prices_list=prices_list, predict_days=pdays)
        net = models.SimpleFFDQN(env.observation_space.shape[0], env.action_space.n)
        models.load_model(file_path, net)

        nets.append(net)
        envs.append(env)

    for i in range(0, 10):
        done = False
        obs = stock_env.reset()
        while not done:
            values = environ.apply_model_from_state(obs, net)
            action = pdenviron.PredAction(np.argmax(values, axis=0))
            obs, reward, done, info = stock_env.step(action)
            print("action:{}, netprice:{}, reward:{}, values:{}".format(action.value, info["net_price"], reward, values))


if __name__ == "__main__":

    db = apdb()
    db.connect()

    kosdaq = stockcode.download_stock_codes('kosdaq')
    update_stocks(db, kosdaq, 'KQ')

    kospi = stockcode.download_stock_codes('kospi')
    update_stocks(db, kosdaq, 'KS')

    db.close()
