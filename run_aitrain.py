import os
import argparse
import subprocess


class RunName(object):
    UPDATE_STOCK = "update_stock_codes.py"

RUN_NAME = RunName()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-r", "--run", default=RUN_NAME.UPDATE_STOCK, help="Run name")
    args = parser.parse_args()

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    process = subprocess.Popen("python "+args.run, stdout=subprocess.PIPE, shell=True)
    process.wait()
    for line in process.stdout:
        print(line)