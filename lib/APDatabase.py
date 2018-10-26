import mysql.connector as mariadb
from cnf.APConfig import config

class APDatabase(object):
    def __init__(self):
        pass

    @staticmethod
    def connect():
        return mariadb.connect(
            host=config.db_host, 
            user=config.db_user, 
            passwd=config.db_pw, 
            db=config.db_name,
            use_pure=True)
   
