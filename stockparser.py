
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
      returns a dataframe with combined values from the csv files in data
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
        