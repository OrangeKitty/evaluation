# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 23:48:39 2018

@author: x_Zero
"""

import pandas as pd
import numpy as np
pd.Series

class rSeries(object):
    
    def __init__(self, series):
        self.series = series
        if not isinstance(series, pd.Series):
            raise TypeError('Input is not pandas.Series type.')            
    
    @property
    def cumulative_return(self):
        return (self.series + 1).cumprod(skipna=True)
    
    @property
    def accumulative_return(self):
        return self.cumulative_return.iloc[-1]
    
    @property
    def mean_return(self):
        return self.series.mean(skipna=True)
    
    @property
    def std(self):
        return self.series.std(skipna=True)
    
    @property
    def cumulative_max(self):
        return self.cumulative_return.cummax(skipna=True)
    
    @property
    def drawdowns(self):
        return self.cumulative_return/self.cumulative_max - 1
        


if __name__ == '__main__':
    
    
    test_return_array = (np.random.rand(20) - 0.4)/10
    test_return_array[:2] = np.nan
    test_return_array[6] = np.nan
    test_return_array[12] = np.nan
    test_series = pd.Series(test_return_array)
    tools = rSeries(test_series)
    print(tools.drawdowns)
    
    test_frame = pd.DataFrame((np.random.rand(20,2) - 0.4)/10)
    print(test_frame.apply(lambda x:rSeries(x).drawdowns, axis=0))