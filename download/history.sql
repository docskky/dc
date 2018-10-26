create user 'ai'@'localhost' identified by 'ajuRi5@9';

create database aistocks character set=utf8;

grant all privileges on aistocks.* to 'ai'@'localhost';

DROP TABLE IF EXISTS aistocks.history_days;
CREATE TABLE IF NOT EXISTS aistocks.history_days (
    scode CHAR(10),
    date DATE,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume INT,
    PRIMARY KEY (scode, date)
)  ENGINE=INNODB character set = utf8;

DROP TABLE IF EXISTS aistocks.stocks;
CREATE TABLE IF NOT EXISTS aistocks.stocks (
    scode CHAR(10),
    name VARCHAR(100),
    PRIMARY KEY (scode)
)  ENGINE=INNODB character set = utf8;



# 상장폐지된 회사 제거
delete from stocks where stocks.scode not in (select hd.scode from history_days hd where hd.date=(select max(hd.date) from history_days hd));

# 365일 미만된 회사 제거
delete from stocks where stocks.scode in (select scode from stocks inner join (select count(*) as cnt, hd.scode as hd_scode from history_days hd group by hd.scode) dt on dt.hd_scode=stocks.scode and dt.cnt < 365);

# stocks 에 없는 history 제거
delete from history_days where history_days.scode not in (select st.scode from stocks st);

# 회사 조회
select count(*), min(hd.date) as start, stocks.scode, name from stocks left join history_days hd on stocks.scode = hd.scode group by stocks.scode;