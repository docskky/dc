import mysql.connector as mariadb
from cnf.APConfig import config


class APDatabase(object):
    def __init__(self):
        self._conn = None
        pass

    def connect(self):
        self._conn = mariadb.connect(
            host=config.db_host,
            user=config.db_user,
            passwd=config.db_pw,
            db=config.db_name,
            use_pure=True)
        return self._conn

    def commit(self):
        if self._conn is None:
            return
        self._conn.commit()

    def close(self):
        if self._conn is None:
            return
        self._conn.close()
