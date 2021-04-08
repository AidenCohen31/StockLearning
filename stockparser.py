
'''
Module that handles Data manipulation for the ML project

Classes:
    Parser
    EDGARStorage
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
import time

@dataclass
class EDGARStorage:
    '''
     Class that stores XML data scraped from 10-Q EDGAR form

    Attributes
    ----------
    name : str
        Name of XML element in GAAP Taxonomy
    attribs: Dict[str,str]
        Dictionary storing attributes of XML element
    '''
    name: str
    attribs: Dict[str, str]

@dataclass
class Grouper:
    '''
     Class that groups related XML gaap elements

    Attributes
    ----------
    elements: List[str]
        List that stores all related elements
    '''
    elements: List[str]

class Parser:
    '''
    Class that combines multiple .csv files into a dataframe.

    Attributes
    ----------
    data : dict
        A dictionary containing the data parsed from 10-Q forms  
    
    Methods
    -------
    scrape():
        Parses the SEC EDGAR database and returns a list of urls with SEC filing data from 10-Q forms
    parserecords():
        Parses 10-Q forms and takes important information
    parse():
        Takes in data from 10-Q forms and processes it into a pandas Dataframe
    filterEDGAR():
        helper method to filter through EDGAR forms
     '''
    data = {}
    def __init__(self):
        pass
    def parse(self)->pd.DataFrame:
        headers = []
        numbers = []
        for i in self.data.items():
            i = i[1]
            grouped = [i for i in [i.name,i.attribs.get("dimension",""),i.attribs.get("contextitMember","")] if i != ""]
            headers.append(Grouper(grouped))
            numbers.append(i.attribs["value"])
        
    
    def scrape(self, ticker: str) -> List[str]:
        #Found out that making a POST request to this site returns a json with the search results
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
            print(url[-16:-8])
            r = requests.get(url)
            root = ET.fromstring(r.text)
            for child in root:
                childtag = child.tag[child.tag.index("}") + 1: ]
                if(childtag == "context"):
                    attribs = {}
                    for values in child.iter():
                        attribs.update(values.attrib)
                        if(not (values.text.isspace() or not values.text)):
                            attribs["context" + values.tag[child.tag.index("}") + 1: ] ] = values.text
                    xmlid = attribs.pop("id")
                    contexts[xmlid] = attribs
                else:
                    if( "id" not in child.attrib or child.attrib["id"] in self.data):
                        print(child.tag, child.attrib)
                        continue
                    xmlid = child.attrib.pop("id")
                    child.attrib["value"] = child.text
                    obj = EDGARStorage(childtag, child.attrib)
                    if("contextRef" in child.attrib):
                            obj.attribs.update(contexts[child.attrib["contextRef"]])
                    if(not xmlid in self.data and self.filterEDGAR(child.tag, obj,url)):
                        self.data[xmlid] = obj


    def filterEDGAR(self,tag: str, obj: EDGARStorage , url: str) -> bool:
        #Helper Method built to condense massive if statements in parserecords() by checking for conditions in an array instead
        conditions = [ "schemaRef" in tag, not "http://fasb.org/us-gaap/2020-01-31" in tag,  "TextBlock" in tag, 
                    not obj.attribs.get("contextendDate",url[-16:-8]).replace("-","") == url[-16:-8], not (isinstance(obj.attribs["value"],str) and obj.attribs["value"].isnumeric())]
        for i in range(len(conditions)):
            if( conditions[i]):
                return False
        return True


        
parser = Parser()
parser.parserecords(["https://www.sec.gov/Archives/edgar/data/320193/000032019321000010/aapl-20201226_htm.xml"])
parser.parse()

