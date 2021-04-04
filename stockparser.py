
'''
Module that handles Data manipulation for the ML project

Classes:
    Parser
Functions:
    None
Misc Variables:
    None
'''
import csv
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import json



class Parser:
    '''
    Class that combines multiple .csv files into a dataframe.

    Attributes
    ----------
    data : dict
        A dictionary containing the years and relative paths of all csv files  
    
    Methods
    -------
    parse():
        Parses the SEC EDGAR database and returns a dataframe with SEC filing data from 10-K and 10-Q forms
     '''
    data = {}
    
    def __init__(self,data):
        self.data = data

    def parse(self)->pd.DataFrame:
        headers = []
        numbers = []
        for i in range(2014,2019):
            with open(self.data[i], "r") as csvfile:
                reader = list(csv.reader(csvfile))
                headers = reader[0] + ["Dates"]
                for row in reader[1:]:
                    numbers.append(row + [i])
        return pd.DataFrame(np.array(numbers),columns=headers)
    
    def scrape(self, ticker):
        r = requests.post("https://efts.sec.gov/LATEST/search-index",data=json.dumps({"dateRange":"all","entityName":"(" + ticker +")","category":"custom", "forms":["10-Q","10-K"]}))
        response = json.loads(r.text)
        urls = []
        for i in response["hits"]["hits"]:
            sep = i["_id"].split(":")
            urls.append("https://www.sec.gov/Archives/edgar/data/" + i["_source"]["ciks"][0] + "/" + sep[0].replace("-","") + "/" + sep[1])
Parser({}).scrape("AAPL")
