import pandas as pd
import numpy as np
from stockparser import Parser

data = {2014: "data\\2014_Financial_Data.csv", 2015: "data\\2015_Financial_Data.csv", 2016: "data\\2016_Financial_Data.csv" , 2017:"data\\2017_Financial_Data.csv", 2018:"data\\2018_Financial_Data.csv"}
df = Parser(data)

print(df.parse())
'''
for i in range(2014,2019):
    with open(data[i], "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            print(row)
'''




