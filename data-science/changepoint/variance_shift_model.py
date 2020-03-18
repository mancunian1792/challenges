import numpy as np
import pandas as pd
from builtins import object
import scipy.stats as stat
from .base_shift_model import BaseShiftModel
from .utils import ts_stats_significance

__author__ = "Harish Ramani"
__email__ = "ramani.h@husky.neu.edu"

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

class VarianceShiftModel(BaseShiftModel):
    def __init__(self, window_size, desired_fpr, step_change, threshold=None, min_change=0, isGaussian=False, change_type="positive"):
        super().__init__(window_size, desired_fpr, step_change, threshold, min_change,change_type)
        self.stat_ts_func = self.__compute_mood_statistic_ts
            
    def __shuffle_timeseries(self, ts):
        """ Shuffle the time series. This also can serve as the NULL distribution. """
        return np.random.permutation(ts)
  
    def __compute_mood_statistic_ts(self, ts):
        statistics = [stat.mood(ts[t+1:], ts[:t+1])[0] for t in np.arange(len(ts)-1)]
        return pd.Series(statistics).dropna().reset_index(drop=True)
    
    def validate_threshold(self, window_size, designed_fpr,step_change,fast=False, normalize=True,isGaussian=False, num_iter=100000, max_decision_multiplier=3):
        return super().validate_threshold(self.stat_ts_func, window_size, designed_fpr, step_change, fast, normalize, isGaussian, num_iter, max_decision_multiplier)
    
    
       
    def detectVarianceShift(self, ts, threshold=None,method=None, B=None, omit_na=True, fast_threshold_calculation=False, normalize_statistic=True, isGaussian=False):
        # Usually the permuation method can be chosen if the number of samples is less.
        null_ts_func = self.__shuffle_timeseries
        return super().detectShift(ts, self.stat_ts_func,threshold=threshold,method=method, B=B, omit_na=omit_na, fast_threshold_calculation=fast_threshold_calculation, normalize_statistic=normalize_statistic, isGaussian=isGaussian, null_ts_func=null_ts_func)
        

    def offlineDetection(self, ts, omit_na=True,index=None, multipleBreakpoints=False, supressPlot=False,fast_threshold_calculation=True, isGaussian=False, normalize_statistic=False):
        return super().offlineDetection(ts, self.stat_ts_func, omit_na=omit_na,index=index, multipleBreakpoints=multipleBreakpoints, supressPlot=supressPlot,fast_threshold_calculation=fast_threshold_calculation, isGaussian=isGaussian, normalize_statistic=normalize_statistic)
            