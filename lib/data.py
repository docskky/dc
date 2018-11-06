import collections
from lib.apdb import APDatabase as apdb
import datetime
import numpy as np

Prices = collections.namedtuple('Prices', field_names=['work', 'open', 'high', 'low', 'close', 'volume'])


# valid_rate: validation 데이터 비율, 0이면 training 데이터와 동일
def load_prices(scodes, valid_rate=0):
#    assert (isinstance(scodes, []))

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

    prices_list = []
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

        # 금액 변동을 비율로 변환
        prices_list.append(prices_to_relative(prices))

    # 테이블을 2개(훈련, 검증)로 나눈다.
    n_days = prices_list[0].work.shape[0]
    train_days = int(n_days * (1 - valid_rate))
    val_prices_list = []

    if train_days == n_days:
        val_prices_list = prices_list
    else:
        for idx in range(0, len(scodes)):
            prices = prices_list[idx]
            t_w, v_w = np.split(prices.work, [train_days])
            t_o, v_o = np.split(prices.open, [train_days])
            t_h, v_h = np.split(prices.high, [train_days])
            t_l, v_l = np.split(prices.low, [train_days])
            t_c, v_c = np.split(prices.close, [train_days])
            t_v, v_v = np.split(prices.volume, [train_days])

            prices_list[idx] = Prices(work=t_w,
                                  open=t_o,
                                  high=t_h,
                                  low=t_l,
                                  close=t_c,
                                  volume=t_v)
            val_prices_list.append(Prices(work=v_w,
                                      open=v_o,
                                      high=v_h,
                                      low=v_l,
                                      close=v_c,
                                      volume=v_v))

    return prices_list, val_prices_list


def prices_to_relative(prices):
    """
    Convert prices to relative in respect to open price
    :param ochl: tuple with open, close, high, low
    :return: tuple with open, rel_close, rel_high, rel_low
    """
    assert isinstance(prices, Prices)
    rh = (prices.high - prices.open) / prices.open
    rl = (prices.low - prices.open) / prices.open
    rc = (prices.close - prices.open) / prices.open
    return Prices(work=prices.work, open=prices.open, high=rh, low=rl, close=rc, volume=prices.volume)
