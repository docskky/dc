import mysql.connector as mariadb
from cnf.APConfig import sconfig


class APDatabase(object):
    def __init__(self):
        self._db = None
        pass

    def connect(self):
        self._db = mariadb.connect(
            host=sconfig.db_host,
            user=sconfig.db_user,
            passwd=sconfig.db_pw,
            db=sconfig.db_name,
            use_pure=True)
        return self._db

    def commit(self):
        if self._db is None:
            return
        self._db.commit()

    def close(self):
        if self._db is None:
            return
        self._db.close()

    def execute_list(self, stmt, params, cursor=None):
        if cursor is None:
            cursor = self._db.cursor(prepared=True)

        format_strings = ','.join(['%s'] * len(params))
        cursor.execute(stmt % format_strings, tuple(params))
        return cursor

    def execute(self, stmt, params, cursor=None):
        if cursor is None:
            cursor = self._db.cursor(prepared=True)
        cursor.execute(stmt, params)
        return cursor