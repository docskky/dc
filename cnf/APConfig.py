import json

class APConfig:
    env = None
    def __init__(self):
        with open('./cnf/config.json') as file:    
            self.env = json.load(file)
        if(self.env is not None):
            self.version = self.env["version"]
            self.db_host = self.env["db_host"]
            self.db_user = self.env["db_user"]
            self.db_pw = self.env["db_pw"]
            self.db_name = self.env["db_name"]
            self.bars_count = self.env["bars_count"]
            self.commission_percent = self.env["commission_percent"]
            self.init_caches = self.env["init_caches"]
            self.bar_download_limit = self.env["bar_download_limit"]
            self.choices = self.env["choices"]
config = APConfig()