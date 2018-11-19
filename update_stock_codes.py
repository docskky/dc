import download.stockcode as stockcode
from lib.apdb import APDatabase as apdb

stmt1 = "insert IGNORE into stocks (scode, name) VALUES (%s,%s);"


def add_stocks(db, df, ext):
    cursor = db.cursor()
    for index, row in df.iterrows():
        scode = ext+row['종목코드']
        name = row['회사명']

        cursor.execute(stmt1, (scode, name))
    cursor.close()


if __name__ == "__main__":

    db = apdb()
    db.connect()

    kosdaq = stockcode.download_stock_codes('kosdaq')
    add_stocks(db, kosdaq, 'KQ')

    kospi = stockcode.download_stock_codes('kospi')
    add_stocks(db, kosdaq, 'KS')

    db.close()
