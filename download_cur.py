import os, sys, inspect
# sys.path.insert(1, os.path.join(sys.path[0], '..'))

from pandas_datareader import data
import pickle
import pandas as pd
import re
import glob
import time
from cnf.APConfig import sconfig as config
from lib.apdb import APDatabase as apdb
import urllib.request
import datetime
import json

LIMIT = config.bar_download_limit
stmt1 = "insert IGNORE into currency (name, date, rate) VALUES (%s,%s,%s);"

local_path = os.path.dirname(os.path.abspath(__file__))


def download_currency():
    db = apdb()
    db.connect()

    date = datetime.datetime.strptime(config.history_begin, "%Y-%m-%d").date()

    today = datetime.datetime.now().date()

    while date <= today:
        count = 0
        done = False
        while not done and count < 100:
            try:
                url = 'https://ratesapi.io/api/{}?base=usd'.format(date.strftime('%Y-%m-%d'))
                cdata = urllib.request.urlopen(url)

                info = json.loads(cdata.read())
                rates = info["rates"]

                for name, rate in rates.items():
                    db.execute(stmt1, (name, date.strftime('%Y-%m-%d'), rate))
                db.commit()
                print('write:', str(date))
                done = True
            except Exception as ex:
                print(ex)
                print('Failed to download ' + name + '. Retrying...')
                time.sleep(10)
                count += 1
                continue

        # 다음날을 지정
        date = date + datetime.timedelta(days=1)


download_currency()
