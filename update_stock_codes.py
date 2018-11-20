import download.stockcode as stockcode
from lib.apdb import APDatabase as apdb
from cnf.APConfig import sconfig as config
import datetime
from pandas_datareader import data
import fix_yahoo_finance as yf
yf.pdr_override()
from contextlib import closing
import time

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


if __name__ == "__main__":

    db = apdb()
    db.connect()

    kosdaq = stockcode.download_stock_codes('kosdaq')
    update_stocks(db, kosdaq, 'KQ')

    kospi = stockcode.download_stock_codes('kospi')
    update_stocks(db, kosdaq, 'KS')

    db.close()
