import collections
from lib.apdb import APDatabase as apdb

Prices = collections.namedtuple('Prices', field_names=['open', 'high', 'low', 'close', 'volume'])


def load_prices(scodes):
    db = apdb()
    db.connect()

    # 주식중 가장 마지막 상장일자를 검색
    stmt = "select max(s_date) from (select min(hd.date) as s_date, hd.scode from history_days hd where hd.scode in (" \
           "%s) group by hd.scode) sdates; "

    cursor = db.execute_list(stmt, scodes)
    start_date = cursor.fetchone()[0]

    # 일자 오름 차순 검색
    stmt = "select hd.date, hd.scode, hd.open, hd.`close`, hd.high, hd.low, hd.volume from history_days hd where " \
           "hd.date >= '{}' and hd.scode in (%s) order by hd.date, hd.scode; "

    stmt = stmt.format(start_date)
    cursor = db.execute_list(stmt, scodes)

    for row in cursor:







