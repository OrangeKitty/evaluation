# -*- coding: utf-8 -*-

import numpy as np
import numpy.ma as ma
import scipy.stats as stats


class indicators():
    
    def __init__(self):
        pass
    
    @staticmethod
    def best(return_arr, axis=0):
        return np.nanmax(return_arr, axis = axis)
    
    @staticmethod
    def worst(return_arr, axis=0):
        return np.nanmin(return_arr, axis = axis)

    @staticmethod
    def annualized_return(return_arr, multiplier=252, axis=0):
        return np.power(np.nanprod(return_arr + 1, axis = axis), 
                        multiplier/return_arr.shape[0]) - 1 
                        
    @staticmethod
    def annualized_vol(return_arr, multiplier=252, axis=0):
        return np.nanstd(return_arr, axis = axis) * np.sqrt(multiplier) 
    
    @staticmethod
    def average_return(return_arr, axis=0):
        return np.nanmean(return_arr, axis = axis)
    
    @staticmethod
    def frequency_transformation(return_arr, multiplier=252/1):
        return np.power(1 + return_arr, multiplier) - 1 
    
    @staticmethod
    def cumulative_return(return_arr, axis=0):
        return np.nancumprod(return_arr + 1, axis = axis)
    
    @staticmethod
    def final_cumulative_return(return_arr, axis=0):        
        return indicators.cumulative_return(return_arr, axis=axis)[-1]

    @staticmethod
    def cumulative_max(return_arr, axis=0):
        cumulative_return = indicators.cumulative_return(return_arr, axis = axis)
        return np.maximum.accumulate(cumulative_return, 
                                     axis = axis)    
    @staticmethod
    def draw_down(return_arr, axis=0):        
        cumulative_return = indicators.cumulative_return(return_arr, axis = axis)        
        cumulative_max = indicators.cumulative_max(return_arr, axis = axis)                
        return np.divide(cumulative_return, cumulative_max) -1     
    
    @staticmethod
    def max_drawdown(return_arr, axis=0):
        draw_downs = indicators.draw_down(return_arr, axis = axis)
        return np.nanmin(draw_downs, axis = axis)    
    
    @staticmethod
    def _slice_draw_down_duration(return_arr, top=1, axis=0):
        
        nav = indicators.cumulative_return(return_arr, axis = axis)
        high_water_marks = np.maximum.accumulate(nav, axis = axis)
        high_water_marks_counts = stats.itemfreq(high_water_marks)
        high_water_marks_positions = np.argsort(high_water_marks_counts[:,-1],axis = axis)[-1*top]
        longest_cumulative_max = high_water_marks_counts[high_water_marks_positions, 0]
        max_draw_downs_durations = np.where(high_water_marks==longest_cumulative_max)[0]
        
        return max_draw_downs_durations.min(), max_draw_downs_durations.max()    
    
    @staticmethod
    def draw_down_range(return_arr, axis=0, top=1):
        return np.apply_along_axis(indicators._slice_draw_down_duration,
                                   axis = axis,
                                   arr=return_arr,
                                   top=top)
    @staticmethod
    def draw_down_duration(return_arr, axis=0, top=1):
        start, end = np.apply_along_axis(indicators._slice_draw_down_duration,
                                   axis = axis,
                                   arr=return_arr,
                                   top=top)
        return end - start
    
    @staticmethod
    def _continuation(return_arr):
                
        climbing_up = np.nancumsum(return_arr>=0)
        climbing_down = np.nancumsum(return_arr<0)        
        continuous_growing_counts = stats.itemfreq(climbing_down)
        continuous_falling_counts = stats.itemfreq(climbing_up)
        
        return continuous_growing_counts[:,-1].max(), continuous_falling_counts[:,-1].max()
    
    @staticmethod
    def longest_continuations(return_arr, axis=0):
        return np.apply_along_axis(indicators._continuation,
                                   axis = axis,
                                   arr = return_arr)
       
    @staticmethod
    def down_side_risk(return_arr, multiplier=252, axis=0):
        returns = return_arr.copy()
        returns[returns>0] = 0
        return np.nanstd(returns, axis = axis) * np.sqrt(multiplier)
    
    @staticmethod
    def up_side_risk(return_arr, multiplier=252, axis=0):
        returns = return_arr.copy()
        returns[returns<0] = 0
        return np.nanstd(returns, axis = axis) * np.sqrt(multiplier) 
   
    @staticmethod
    def skewness(return_arr, axis=0):
        return np.array(stats.skew(return_arr, axis = axis))
    
    @staticmethod
    def kurtosis(return_arr, axis=0):
        return np.array(stats.kurtosis(return_arr, axis = axis))
    
    @staticmethod
    def winning_ratio(return_arr, axis=0):
        winning = np.sum(return_arr, axis = axis)
        total = return_arr.shape[1:]        
        return np.divide(winning, total)
  
    @staticmethod
    def VaR(return_arr, alpha=0.05, axis=0):
        return np.nanpercentile(return_arr, alpha*100, axis=axis)
    
    @staticmethod
    def CVaR(return_arr, alpha=0.05, axis=0):
        VaR = indicators.VaR(return_arr, alpha=alpha, axis=axis)
        return np.nanmean(ma.array(return_arr, mask = return_arr>VaR), axis=axis)
   
    @staticmethod
    def tailRisk(return_arr, alpha=0.05, axis=0):
        VaR = indicators.VaR(return_arr, alpha=alpha, axis=axis)
        arr = return_arr.copy()
        arr[arr>VaR] = np.nan
        return np.nanstd(arr, axis=axis)
        
    @staticmethod
    def annualized_absolute_return(return_arr, risk_free_arr=0, multiplier=252, axis=0):
        return indicators.annualized_return(return_arr - risk_free_arr, 
                                            multiplier = multiplier,
                                            axis = axis) 
    
    @staticmethod
    def sharpe(return_arr, risk_free_arr=0, multiplier=252, axis=0):
        absolute_return = indicators.annualized_absolute_return(return_arr = return_arr,
                                                                risk_free_arr = risk_free_arr,
                                                                multiplier = multiplier,
                                                                axis = axis)        
        annualized_volitility = indicators.annualized_vol(return_arr = return_arr,
                                                         multiplier = multiplier,
                                                         axis = axis)        
        return np.divide(absolute_return, annualized_volitility)
    
    @staticmethod
    def calmar(return_arr, risk_free_arr=0, multiplier=252, axis=0):
        absolute_return = indicators.annualized_absolute_return(return_arr = return_arr,
                                                                risk_free_arr = risk_free_arr,
                                                                multiplier = multiplier,
                                                                axis = axis)        
        max_draw_down = abs(indicators.max_drawdown(return_arr = return_arr,
                                                    axis = axis))        
        max_draw_down[max_draw_down == 0] = np.inf
        return np.divide(absolute_return, max_draw_down)
    
    @staticmethod
    def sortino(return_arr, risk_free_arr=0, multiplier=252, axis=0):
        absolute_return = indicators.annualized_absolute_return(return_arr = return_arr,
                                                                risk_free_arr = risk_free_arr,
                                                                multiplier = multiplier,
                                                                axis = axis)         
        down_side_risk = indicators.down_side_risk(return_arr = return_arr,
                                                   multiplier = multiplier,
                                                   axis = axis)        
        down_side_risk[down_side_risk == 0] = np.inf
        return np.divide(absolute_return, down_side_risk)
    
    @staticmethod
    def annualized_active_return(return_arr, bench_mark_arr, multiplier=252, axis=0):
        return indicators.annualized_return(return_arr - bench_mark_arr, 
                                            multiplier = multiplier, 
                                            axis = axis)
    
    @staticmethod
    def annualized_active_vol(return_arr, bench_mark_arr, multiplier=252, axis=0):
        return indicators.annualized_vol(return_arr - bench_mark_arr, 
                                            multiplier = multiplier, 
                                            axis = axis)
    
    @staticmethod
    def information_ratio(return_arr, bench_mark_arr, multiplier=252, axis=0):
        annualized_active_return = indicators.annualized_active_return(return_arr = return_arr,
                                                                       bench_mark_arr = bench_mark_arr, 
                                                                       multiplier = multiplier, 
                                                                       axis = axis)
        annualized_active_vol = indicators.annualized_active_vol(return_arr = return_arr,
                                                               bench_mark_arr = bench_mark_arr, 
                                                               multiplier = multiplier, 
                                                               axis = axis)
        return np.divide(annualized_active_return, annualized_active_vol)
       

if __name__ == '__main__':
    
    test_array = (np.random.rand(40,2)-0.45)/10
    test_benchmark = (np.random.rand(40,2)-0.55)/10 
    test_nav = indicators.cumulative_return(test_array)

    print(indicators.draw_down_duration(test_array))



