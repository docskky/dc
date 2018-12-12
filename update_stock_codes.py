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
import lib.log as log
import torch
import lib.environ as environ
import lib.pdenviron as pdenviron
import lib.models as models
import lib.data


"""
어제 일자까지의 모든 주가를 업데이트 한다.
"""
stmt1 = "insert IGNORE into stocks (scode, name) VALUES (%s,%s);"
stmt2 = "select max(date) from history_days where scode=%s;"
stmt3 = "insert into history_days (scode, date, open, high, low, close, volume) VALUES (%s,%s, %s, %s,%s,%s,%s) \
            ON DUPLICATE KEY UPDATE \
            open=%s, high=%s, low=%s, close=%s, volume=%s;"
stmt4 = "insert IGNORE into predicts (scode, date, predict_days, expect1, expect2, expect3, expect4, expect5) \
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s);"
date_format = '%Y-%m-%d'

logger = log.logger(os.path.basename(__file__))


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
                        if last_date >= datetime.datetime.now().date() - datetime.timedelta(days=1):
                            # 저장 일자가 하루 전날 까지면 스킵한다.
                            break
                        # 마지막 저장일 7일 전 데이터 부터 검색을 시작한다.
                        last_date - datetime.timedelta(days=7)
                        from_date = last_date
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
                        cur.execute(stmt3, (scode, date, open, high, low, close, volume,
                                            open, high, low, close, volume))
                        db.commit()
                    done = True
            except Exception as ex:
                time.sleep(10)
                count += 1
                if count >= 20:
                    done = True
                logger.error('get_data_yahoo() failed:'+str(ex))


def update_prediction(db):

    # 트래이닝 된 모델을 로드한다.
    predict_days = [7, 14, 30]
    nets = []
    # environment의 shape 샘플이 필요하므로 KQ003380 종목을 임의로 로드한다.
    sample_prices_list, _valid_list = lib.data.load_prices(["KQ003380"])
    for pdays in predict_days:
        file_path = "data/v3.0-phase3-{}.data".format(pdays)
        env = pdenviron.PredEnv(prices_list=sample_prices_list, predict_days=pdays)
        net = models.SimpleFFDQN(env.observation_space.shape[0], env.action_space.n)
        models.load_model(file_path, net)

        nets.append(net)

    today = datetime.datetime.now().date()
    with closing(db.cursor()) as cur:
        cur.execute("select scode from stocks")
        for row in cur.fetchall():
            scode = row[0]
            prices_list, val_prices_list = lib.data.load_prices([scode])

            if len(prices_list[0].open) < 60:
                continue

            try:
                with closing(db.cursor()) as cur2:
                    for i in range(0, len(nets)):
                        pdays = predict_days[i]
                        env = pdenviron.PredEnv(prices_list=prices_list, predict_days=pdays)

                        # offset을 마지막 일자로 한다.
                        obs = env.reset(0, len(prices_list[0].open)-1)
                        values = environ.apply_model_from_state(obs, nets[i])
                        # 예측결과 저장
                        cur2.execute(stmt4, (scode, today, pdays, values[0], values[1], values[2], values[3], values[4]))
            except Exception as ex:
                logger.error('update_prediction() failed:'+str(ex))


if __name__ == "__main__":

    db = apdb()
    db.connect()

    kosdaq = stockcode.download_stock_codes('kosdaq')
    logger.info('Downloading kosdaq codes complete.' + str(kosdaq.shape[0]))

    update_stocks(db, kosdaq, 'KQ')

    kospi = stockcode.download_stock_codes('kospi')
    logger.info('Downloading kospi codes complete.' + str(kospi.shape[0]))

    update_stocks(db, kospi, 'KS')

    update_prediction(db)

    db.close()
