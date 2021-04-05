
'''
Module that handles Data manipulation for the ML project

Classes:
    Parser
Functions:
    None
Misc Variables:
    None
'''
from typing import List, Dict
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import csv
import pandas as pd
import numpy as np
import requests
import json

@dataclass
class EDGARStorage:
    name: str
    attribs: Dict[str, str]

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
    def __init__(self):
        pass
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
    
    def scrape(self, ticker: str) -> List[str]:
        r = requests.post("https://efts.sec.gov/LATEST/search-index",data=json.dumps({"dateRange":"all","entityName":"(" + ticker +")","category":"custom", "forms":["10-Q","10-K"]}))
        response = json.loads(r.text)
        urls = []
        for i in response["hits"]["hits"]:
            sep = i["_id"].split(":")
            urls.append("https://www.sec.gov/Archives/edgar/data/" + i["_source"]["ciks"][0] + "/" + sep[0].replace("-","") + "/" + sep[1])
        return urls
    def parserecords(self, urls: List[str])-> List[List[str]]:
        contexts = {}
        for url in urls:
            r = requests.get(url)
            root = ET.fromstring(r.text)
            for child in root:
                childtag = child.tag[child.tag.index("}") + 1: ]
                if(childtag == "context"):
                    attribs = {}
                    for values in child.iter():
                        attribs.update(values.attrib)
                        attribs["context"] = values.text
                    xmlid = attribs.pop("id")
                    contexts[xmlid] = attribs
                elif(childtag !="schemaRef" ):
                    if( "id" not in child.attrib):
                        print(child.tag, child.attrib)

                    if(not child.attrib["id"] in self.data):
                        xmlid = child.attrib.pop("id")
                        child.attrib["value"] = child.text
                        self.data[xmlid] = EDGARStorage(childtag, child.attrib)
                        if("contextRef" in child.attrib):
                            self.data[xmlid].attribs.update(contexts[child.attrib["contextRef"]])
                    else:
                        print(child.tag, child.attrib)
            print(len(self.data))
                        
    
        
Parser().parserecords(["https://www.sec.gov/Archives/edgar/data/320193/000032019321000010/aapl-20201226_htm.xml"])
