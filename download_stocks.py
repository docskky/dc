import os,sys,inspect
#sys.path.insert(1, os.path.join(sys.path[0], '..'))

from pandas_datareader import data
import fix_yahoo_finance as yf
yf.pdr_override()
import pickle
import pandas as pd
import re
import glob
import time
from cnf.APConfig import sconfig
from lib.apdb import APDatabase as database

START_DATE = '2000-01-01'
LIMIT = sconfig.bar_download_limit
stmt1 = "insert IGNORE into stocks (scode, name) VALUES (%s,%s);"
stmt2 = "insert IGNORE into history_days (scode, date, open, high, low, close, volume) VALUES (%s,%s, %s, %s,%s,%s,%s);"
stmt3 = "select count(*), stocks.scode, name from stocks left join history_days on stocks.scode = history_days.scode group by stocks.scode;"
local_path = os.path.dirname(os.path.abspath(__file__))


def download_data(db, name, ext):
    pickle = pd.read_pickle(local_path+'/download/'+name+'.pickle')
    count = 0
    for stock in pickle.values:
        kor_name = stock[0]
        ticker = stock[1]
        _store_db(db, kor_name, ticker, ext)
        count += 1
        if LIMIT > 0 and LIMIT == count:
            break


def _store_db(db, name, ticker, ext):
    count = 0
    scode = ext+ticker

    cursor = db.cursor(prepared=True)
    cursor.execute(stmt1, (scode, name))

    print('Saving '+name+'['+ticker+']')
    while count < 100:
        try:
            #cursor = db.cursor(prepared=True)
            df = data.get_data_yahoo(ticker + '.'+ext, START_DATE, thread=20)
            for i, row in df.iterrows():
                date = row.name.strftime('%Y-%m-%d')
                open = row['Open']
                high = row['High']
                low = row['Low']
                close = row['Close']
                volume = row['Volume']
                cursor.execute(stmt2, (scode, date, open, high, low, close, volume))
                #df.to_csv('./kosdaq/{}.csv'.format(ticker))
            print('{} is saved with #{} count.'.format(ticker, df.shape[0]))
            break
        except Exception as ex:
            print(ex)
            print('Failed to download '+name+'. Retrying...')
            time.sleep( 10 )
            count += 1
            continue
    db.commit()

#def validate_db():
#    stmt3

db = database.connect()

download_data(db=db, name='kospi', ext='KS')
download_data(db=db, name='kosdaq', ext='KQ')
