import collections
from lib.apdb import APDatabase as apdb
import datetime
import numpy as np

Prices = collections.namedtuple('Prices', field_names=['work', 'open', 'high', 'low', 'close', 'volume'])


def load_prices(scodes):
    db = apdb()
    db.connect()

    # 주식중 가장 마지막 상장일자를 검색
    stmt = "select max(s_date) from (select min(hd.date) as s_date, hd.scode from history_days hd where hd.scode in (" \
           "%s) group by hd.scode) sdates; "

    cursor = db.execute_list(stmt, scodes)
    start_date = cursor.fetchone()[0]

    # 일자 오름 차순 검색
    stmt = "select date, open, close, high, low, volume from history_days hd where " \
           "hd.date >= %s and hd.scode =%s order by hd.date;"

    price_list = []

    next_date = None
    for sc in scodes:
        cursor = db.execute(stmt, (start_date, sc))
        w, o, h, l, c, v = [], [], [], [], [], []

        for row in cursor:
            date = row[0]
            # 비어있는 일자를 휴관일로 추가한다.
            if next_date is not None:
                while date > next_date:
                    last_price = c[-1]
                    w.append(False)
                    o.append(last_price)
                    h.append(last_price)
                    l.append(last_price)
                    c.append(last_price)
                    v.append(0)
                    next_date = next_date + datetime.timedelta(days=1)

            _open = row[1]
            close = row[2]
            high = row[3]
            low = row[4]
            volume = row[5]

            w.append(True)
            o.append(_open)
            h.append(high)
            l.append(low)
            c.append(close)
            v.append(volume)

            # 다음날을 지정
            next_date = date + datetime.timedelta(days=1)

        prices = Prices(work=np.array(w, dtype=np.bool_),
                        open=np.array(o, dtype=np.float32),
                        high=np.array(h, dtype=np.float32),
                        low=np.array(l, dtype=np.float32),
                        close=np.array(c, dtype=np.float32),
                        volume=np.array(v, dtype=np.float32))

        price_list.append(prices)

    return price_list
