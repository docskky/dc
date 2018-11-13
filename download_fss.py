from urllib.request import urlopen
import pandas as pd
from bs4 import BeautifulSoup
import webbrowser
from html_table_parser import parser_functions as parser
from cnf.APConfig import sconfig as config
from lib.apdb import APDatabase as apdb
import urllib.request
from contextlib import closing
import datetime
import json
import re
from re import sub
from decimal import Decimal

FSS_KEY = "b267cd138ea9d9a76f43b483409c306589de86ff"


def download_list():
    db = apdb()
    db.connect()

    stmt_insert = "insert IGNORE into profits (scode, date, totals, pure_profit) VALUES (%s,%s,%s,%s);";

    s_dt = config.history_begin.replace('-', '')
    url_format = "http://dart.fss.or.kr/api/search.json?page_no={}&start_dt={}&auth={}&crp_cd={}"
    cursor = db.cursor()
    cursor.execute("select scode from stocks")

    for row in cursor:
        scode = row[0]
        code = scode[2:]
        page_no = 1
        total_page = 1
        while total_page >= page_no:
            report = urlopen(url_format.format(page_no, s_dt, FSS_KEY, code))
            info = json.load(report)
            if info["err_code"] == "000":
                total_page = int(info["total_page"])
                list = info["list"]
                for d in list:
                    rcp_no = d["rcp_no"]
                    result = get_fss(rcp_no)
                    if result:
                        dates = result[0]
                        total_values = result[1]
                        profit_values = result[2]

                        cursor2 = db.cursor()
                        for i in range(0, len(dates)):
                            cursor2.execute(stmt_insert, (scode, dates[i], total_values[i], profit_values[i]))

                        cursor2.close()
            page_no += 1
    cursor.close()
    db.close()


def get_fss(rcp_no):
    url1 = "http://dart.fss.or.kr/dsaf001/main.do?rcpNo={}"
    page = urlopen(url1.format(rcp_no))
    html = page.read().decode('utf-8')
    # viewDoc('11111', '22222', 함수를 검색해서 dcm_no 값을 추출한다.
    result = re.search(r'viewDoc\(\'(.*)\', \'(.*)\',', html)
    if not result:
        return

    try:
        dcm_no = result.group(2)

        url2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo={}&dcmNo={}&eleId=15&offset=297450&length=378975&dtd=dart3.xsd".format(
            rcp_no, dcm_no)
        page = urlopen(url2)
        r = page.read()
        xmlsoup = BeautifulSoup(r, 'html.parser')
        body = xmlsoup.find("body")
        tables = body.find_all("table")

        head = parser.make2d(tables[0])
        dates = []
        dstr = re.search(r'(\d+.\d+.\d+)', head[1][0])
        date = datetime.datetime.strptime(dstr.group(1), "%Y.%m.%d").date()
        dates.append(date)
        dstr = re.search(r'(\d+.\d+.\d+)', head[2][0])
        date = datetime.datetime.strptime(dstr.group(1), "%Y.%m.%d").date()
        dates.append(date)

        if not dates[0] or not dates[1]:
            return

        total_values = []
        totals = parser.make2d(tables[1])

        # 자산
        for idx in range(0, len(totals)):
            if totals[idx][0] == '자본과부채총계':
                m = strip_money(totals[idx][1])
                total_values.append(m)
                m = strip_money(totals[idx][2])
                total_values.append(m)
                break

        if len(total_values) < 2:
            return

        profit_values = []
        # 당기순이익
        profits = parser.make2d(tables[3])
        for idx in range(0, len(profits)):
            if '당기순이익' in profits[idx][0]:
                m = strip_money(profits[idx][1])
                profit_values.append(m)
                m = strip_money(profits[idx][3])
                profit_values.append(m)
                break

        if len(profit_values) < 2:
            return

        return dates, total_values, profit_values

    except:
        pass

    return


def strip_money(txt):
    value = float(Decimal(sub(r'[^\d.]', '', txt)))
    if '(' in txt:
        value = -value
    return value


download_list()