# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import numpy.ma as ma


class indicators(object):
    
    def __init__(self):
        pass
    
    @staticmethod
    def cumulative_return(return_arr, axis=0):
        return np.nancumprod(1 + return_arr, axis = axis)
    
    @staticmethod
    def mean_return(return_arr, axis=0):
        return np.nanmean(return_arr, axis = axis)
    
    @staticmethod
    def volitility(return_arr, axis=0):
        return np.nanstd(return_arr, axis = axis)
    
    @staticmethod
    def annualized_return(return_arr, annualization_multiplier=252, axis=0):
        power = annualization_multiplier/return_arr.shape[axis]
        return np.power(1 + return_arr, power) - 1
    
    @staticmethod
    def annualized_vol(return_arr, annnualization_multiplier=252, axis=0):
        multiplier = np.sqrt(annnualization_multiplier)
        return np.nanstd(return_arr, axis=axis)*multiplier
    
#    @staticmethod
#    def 

if __name__ == '__main__':
    
    
    test_return_array = (np.random.rand(20,2) - 0.4)/10
    test_return_array[:2,:] = np.nan
    test_return_array[6,0] = np.nan
    test_return_array[12,1] = np.nan
    print(indicators.annualized_vol(test_return_array))
    
    
    
    
    
    