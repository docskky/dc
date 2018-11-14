import sys
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
import numpy as np

FSS_KEY = "b267cd138ea9d9a76f43b483409c306589de86ff"


def download_list():
    db = apdb()
    db.connect()

    stmt_insert = "insert IGNORE into fstatment (scode, date, current_asset, static_asset, current_dept, \
        static_dept, capital, total_assets, income, expense, pure_profit) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);";

    s_dt = config.history_begin.replace('-', '')
    url_format = "http://dart.fss.or.kr/api/search.json?crp_cd={}&page_no={}&start_dt={}&auth={}"
    cursor = db.cursor()
    cursor.execute("select scode from stocks")

    for row in cursor.fetchall():
        scode = row[0]
        try:
            scode = scode.decode()
        except AttributeError:
            pass

        code = scode[2:]
        page_no = 1
        total_page = 1
        retry_cnt = 0
        while total_page >= page_no and retry_cnt < 20:
            request_url = url_format.format(code, page_no, s_dt, FSS_KEY)
            try:
                report = urlopen(request_url)
            except:
                retry_cnt += 1
                continue
            info = json.load(report)
            if info["err_code"] == "000":
                total_page = int(info["total_page"])
                list = info["list"]
                for d in list:
                    rcp_no = d["rcp_no"]
                    result = get_fss(rcp_no)
                    if result:
                        try:
                            dates = result[0]
                            info_tbl = result[1]
                            zero_array = np.zeros(len(dates))
                            current_asset = get_value_for_key(info_tbl, "유동자산", zero_array)

                            static_asset = get_value_for_key(info_tbl, "비유동자산", zero_array)
                            current_dept = get_value_for_key(info_tbl, "유동부채", zero_array)
                            static_dept = get_value_for_key(info_tbl, "비유동부채", zero_array)
                            capital = get_value_for_key(info_tbl, "자본총계", zero_array)
                            total_assets = get_value_for_key(info_tbl, "자본과부채총계", zero_array)
                            income = get_value_for_key(info_tbl, ["영업수익", "수익(매출액)"], zero_array)
                            expense = get_value_for_key(info_tbl, ["영업비용", "매출원가"], zero_array)
                            pure_profit = get_value_for_key(info_tbl, "당기순이익(손실)", zero_array)

                            cursor2 = db.cursor()
                            for i in range(0, len(dates)):
                                cursor2.execute(stmt_insert, (scode, dates[i], current_asset[i], static_asset[i],
                                                              current_dept[i], static_dept[i], capital[i],
                                                              total_assets[i],
                                                              income[i], expense[i], pure_profit[i]))
                            db.commit()
                        except:
                            print("error:", sys.exc_info())
                        finally:
                            if cursor2:
                                cursor2.close()
            else:
                retry_cnt += 1
                continue

            page_no += 1
    cursor.close()
    db.close()


def get_fss(rcp_no):
    url1 = "http://dart.fss.or.kr/dsaf001/main.do?rcpNo={}"

    retry_cnt = 0
    while retry_cnt < 20:
        try:
            page = None
            try:
                page = urlopen(url1.format(rcp_no))
            except:
                retry_cnt += 1
                continue
            html = page.read().decode('utf-8')
            # viewDoc('11111', '22222', 함수를 검색해서 dcm_no 값을 추출한다.
            result = re.search(r'viewDoc\(\'(.*)\', \'(.*)\',', html)

            dcm_no = result.group(2)

            url2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo={}&dcmNo={}&eleId=15&offset=297450&length=378975&dtd=dart3.xsd".format(
                rcp_no, dcm_no)
            page = None
            try:
                page = urlopen(url2)
            except:
                retry_cnt += 1
                continue

            r = page.read()
            xmlsoup = BeautifulSoup(r, 'html.parser')
            body = xmlsoup.find("body")
            tables = body.find_all("table")

            head = parser.make2d(tables[0])
            dates = []
            for grp in range(1, len(head) - 1):
                dstr = re.search(r'(\d+.\d+.\d+)', head[grp][0])
                date = datetime.datetime.strptime(dstr.group(1), "%Y.%m.%d").date()
                dates.append(date)

            if len(dates) == 0:
                return

            info_tbl = {}
            totals = parser.make2d(tables[1])

            # 자산
            for idx in range(0, len(totals)):
                list = []
                for grp in range(1, len(dates) + 1):
                    list.append(strip_money(totals[idx][grp]))
                info_tbl[totals[idx][0].strip()] = list

            strip_money(totals[idx][2])
            # 당기순이익
            profits = parser.make2d(tables[3])
            for idx in range(0, len(profits)):
                list = []
                for grp in range(0, len(dates)):
                    list.append(strip_money(profits[idx][1 + grp * 2]))
                info_tbl[profits[idx][0].strip()] = list

            return dates, info_tbl

        except:
            #print("error:", sys.exc_info()[0])
            pass

    return


def strip_money(txt):
    try:
        value = float(Decimal(sub(r'[^\d.]', '', txt)))
        if '(' in txt:
            value = -value
        return value
    except:
        return 0.0


def find_value_contains_key(dict, subkey, default=None):
    values = [value for key, value in dict.items() if subkey in key]
    if len(values) > 0:
        return values[0]
    return default


def get_value_for_key(dict, key, default=None):
    if isinstance(key, list):
        for k in key:
            try:
                return dict[k]
            except:
                pass
    else:
        try:
            return dict[key]
        except:
            pass
    return default


download_list()
