import numpy as np
import pandas as pd
import scipy.stats as stat
from .utils import ts_stats_significance, gaussanityCheck, parallelize_func
from collections import namedtuple
import matplotlib.pyplot as plt

__author__ = "Harish Ramani"
__email__ = "ramani.h@husky.neu.edu"

'''
https://pdfs.semanticscholar.org/cf34/18eb74ec703c2eed3ff9c2461dd9c0c2cddd.pdf

Please refer the above paper for explainations behind the implementation.

The purpose of this method is to find a single threshold point that could work well for multiple systems
as the threshold isn't set based on magnitude but based on rank ordered statistics.

The threshold is found based on parameters like 
    1. window size (number of samples to be considered to make a decision)
    2. Desired False positive rate (Desired number of false positives . When making sequential decisions we need to be vary of the false positives
    while computing threshold. fpr ranges from ~>0 -100 although you wouldnt set more than 5% false positives cause who needs that ? .4
    FPR= (Number of false positives/ Number of total decisions) * 100)
    3. Step Change - Number of samples before which we make a sequential decision.
    4. Change type - If its positive then whatever change we are detecting (mean/variance) we try to detect a increase in mean/variance.
    Our test statistic and the way we set threshold changes due to this.

Under null hypothesis, we dont expect change, hence we simulate random normal variables of desired window_length
and run multiple simulations to find a threshold using `_find_optimal_threshold` method. Now, when we expect change, irrespective of the magnitude,
the test statistic changes beyond the threshold value, if the change is significant. 


P.S - I worked on this during my co-op where i made the above paper into a package. I had to re-work on this from scratch based on my notes
that i had while implementing it 4-5 months back.
I just had to implement the variance detection part. 
'''


ShiftResult = namedtuple("ShiftResult", ('isShift','statistic','breakpoint'))
OfflineDetectionResult = namedtuple("OfflineDetectionResult", ("breakpoints", "intervals", "message", "firstBreakpointDecisionNumber", "numberofDecisions"))
ThresholdValidationResult = namedtuple("ThresholdValidationResult", ("isThresholdValid", "firstBreakpointDecisionNums", "messages"))

class BaseShiftModel(object):
    def __init__(self, window_size, desired_fpr, step_change, threshold=None, min_change=0, change_type="positive"):
        '''
        Constructor. Set the parameters.Minimum number of samples set as 20
        '''
        if window_size < 20:
            self.window_size = 20
        else:
            self.window_size = window_size
        self.threshold = threshold
        self.desired_fpr = desired_fpr
        self.step_change = step_change
        self.null_threshold_distribution = None
        self.min_change = min_change
        self.change_type = change_type

    def __find_optimal_threshold(self, window_size, designed_fpr, stat_func, fast=False, normalize=False):
        '''
        This method computes the optimal threshold based on the arguments.
        '''
        # Here we compute the stat function for the desired window size from a distribution from the null hypothesis
        # Here we have no mean, shift or variance shift or any other shift. Hence , our base hypothesis is that there is no shift.
        num_iter = 1000 if fast else 100000 # Fast will be less accurate as the number of iterations is less. The difference shouldnt be that huge.
        if (self.threshold == None):
            null_ts_args = np.array([[0,1,window_size]]*num_iter)
            null_ts = [np.random.normal(*arg) for arg in null_ts_args]
            stat_all = parallelize_func(null_ts, stat_func)
            if normalize:
                self.null_threshold_distribution = [max((stat-np.mean(stat))/np.var(stat)) for stat in stat_all]
            else:
                self.null_threshold_distribution = [max(stat) for stat in stat_all]
            if self.change_type == "positive":
                self.threshold = np.percentile(self.null_threshold_distribution, (100-designed_fpr))
            elif self.change_type == "negative":
                self.threshold = np.percentile(self.null_threshold_distribution, designed_fpr)
            return self.threshold
        return self.threshold
    
    def validate_threshold(self, stat_ts_func, window_size, designed_fpr,step_change,fast=False, normalize=True,isGaussian=False, num_iter=100000, max_decision_multiplier=3):
        '''
        This method validates whether the threshold obtained satisfies the desired false positive rate under the null hypothesis (no change observed)
        '''
        self.__find_optimal_threshold(window_size, designed_fpr, stat_ts_func, fast=fast, normalize=normalize)
        min_num_decisions = round(1/ (designed_fpr/100))
        test_sample_size = (window_size) + (max_decision_multiplier * (min_num_decisions* step_change))
        # create null hypothesis sample
        null_ts = [np.random.normal(0,1,test_sample_size) for i in range(1,num_iter)]
        stat_all = parallelize_func(null_ts, self.offlineDetection, [stat_ts_func,False, None, False,False,False, False, True])
        first_breakpoint_num_decisions = [stat.firstBreakpointDecisionNumber for stat in stat_all]
        messages = [stat.message for stat in stat_all]
        if np.nanmean(first_breakpoint_num_decisions) > min_num_decisions:
            return ThresholdValidationResult(True,first_breakpoint_num_decisions,messages)
        else:
            return ThresholdValidationResult(False, first_breakpoint_num_decisions,messages)

    def __permutationMethod(self, ts, stat_ts_func, null_ts_func,B):
        result = ts_stats_significance(ts, stat_ts_func, null_ts_func, B=B, permute_fast=True, bootstrap=True)
        return result
    
    def plot_changepoint_estimate(self, ts, index, allBkps, intervals, multipleBreakpoints):
        '''
        Method to plot the detection time and the changepoint estimate.
        '''
        plt.figure(figsize=(50,20))
        if index is not None:
            plt.plot(index, ts, label="timeseries")
            # We need to change the intervals and breakpoints according to the index.(usually dates)
            for ind in range(0,len(allBkps)):
                interval = intervals[ind]
                interval_changed = (index.iloc[interval[0]], index.iloc[interval[1]])
                intervals[ind] = interval_changed
                allBkps[ind] = index.iloc[interval[0]+allBkps[ind]]
        else:
            plt.plot(ts, label="timeseries")
            for ind in range(0,len(allBkps)):
                allBkps[ind] = intervals[ind][0] + allBkps[ind]
        if (len(allBkps) > 0) and (multipleBreakpoints):
            for i in range(0, len(allBkps)):
                plt.plot([allBkps[i], allBkps[i]], [min(ts), max(ts)], label="changepoint estimate "+str(i))
                plt.plot([intervals[i][1],intervals[i][1]],[min(ts),max(ts)], label="detection time "+str(i))
        if (multipleBreakpoints!=True) and (len(allBkps)==1):
            plt.plot([intervals[0][1],intervals[0][1]],[min(ts),max(ts)], label="detection time")
            plt.plot([allBkps[0], allBkps[0]], [min(ts),max(ts)], label="changepoint estimate")
        plt.xticks(rotation=90)
        plt.legend()

    def detectShift(self, ts, stat_ts_func,threshold=None,method=None, B=None, omit_na=True, fast_threshold_calculation=False, normalize_statistic=True, isGaussian=False, null_ts_func=None):
        '''
        Online testing method.
        '''
        # Usually the permuation method can be chosen if the number of samples is less.
        ts = pd.Series(ts)
        if threshold is None:
            threshold = self.__find_optimal_threshold(self.window_size, self.desired_fpr, stat_ts_func, fast=fast_threshold_calculation, normalize=normalize_statistic)
        assert ts.ndim == 1
        if omit_na:
            ts = ts.dropna().reset_index(drop=True)
        if method=="permutation":
            if B is None:
                B=1000 # Running it with default number of boostraps
            return self.__permutationMethod(ts, stat_ts_func, null_ts_func,B=B)
        # Depending upon the guassanity of the data we choose the function.
        significance = ts_stats_significance(ts, stat_ts_func)
        stistcs = list(significance["statistics"])
        if normalize_statistic:
            stistcs = (stistcs - np.mean(stistcs))/np.var(stistcs)
        max_stat = max(stistcs)
        if (self.change_type=="positive") and (max_stat > threshold):
            bkp = list(stistcs).index(max_stat)
            return ShiftResult(True, stistcs,bkp)
        elif (self.change_type=="negative") and (max_stat < threshold):
            bkp = list(stistcs).index(max_stat)
            return ShiftResult(True, stistcs,bkp)
        else:
            return ShiftResult(False,stistcs, None)
    
    def offlineDetection(self, ts, stat_ts_func,omit_na=True,index=None, multipleBreakpoints=False, supressPlot=False,fast_threshold_calculation=True, isGaussian=False, normalize_statistic=False):
        '''
        Offline Detection.
        '''
        try:
            ts = pd.Series(ts)
            assert ts.ndim == 1
            if omit_na:
                # Check condition for index.
                remove_indexes = ts.isna()
                index = index[~remove_indexes.values]
                ts = ts.dropna().reset_index(drop=True)
            if ts.shape[0] < self.window_size:
                return OfflineDetectionResult([], [], "Samples less than window size", 0,0)
            runThis=True
            start = 0
            end = start+self.window_size
            result = []
            bkps = 0
            allBkps = []
            intervals = []
            resetWindow = False
            num_decisions = 0
            first_breakpoint_decision_num = 0
            # Step 1 -> Check Gaussanity
            #isGaussian = gaussanityCheck(ts)
            #print("Gaussian test completed. Is it gaussian ?", isGaussian)
            #pbar.update(5)
            # Step 2 -> Get Threshold
            threshold = self.__find_optimal_threshold(self.window_size, self.desired_fpr, stat_ts_func, fast=fast_threshold_calculation, normalize=normalize_statistic)
            #pbar.update(25)
            #Step 3-> Run the rolling window.
            
            # Corner cases. When total number of samples is less than window size . ! Should i proceed with the test ?
            while(runThis):
                if end < ts.shape[0]:
                    #print("start is %d and end is %d"%(start,end))
                    detection = self.detectShift(ts[start:end], stat_ts_func, threshold=threshold, normalize_statistic=normalize_statistic)
                    num_decisions+=1
                    result.append(detection.isShift)
                    if detection.isShift:
                        if first_breakpoint_decision_num==0:
                            first_breakpoint_decision_num = num_decisions
                        bkps = detection.breakpoint
                        allBkps.append(bkps)
                        intervals.append((start, end))
                        if multipleBreakpoints!=True:
                            runThis=False
                        else:
                            resetWindow=True
                    else:
                        resetWindow=False
                    if(runThis!=False):
                        if(resetWindow==True):
                            start = end
                            end = end + self.window_size
                        else:
                            start = start+self.step_change
                            if end+self.step_change > ts.shape[0]:
                                end = ts.shape[0]
                                runThis=False
                            else:
                                end=end+self.step_change
                else:
                    runThis=False
            # Step 4 -> Plot the changepoint estimate.
            #pbar.update(60)
            if supressPlot != True:
                self.plot_changepoint_estimate(ts, index, allBkps, intervals, multipleBreakpoints)
            #pbar.update(10)
            message = "Completed"
            if multipleBreakpoints:
                return OfflineDetectionResult(allBkps, intervals, message, first_breakpoint_decision_num, num_decisions)
            return OfflineDetectionResult(allBkps, intervals, message, num_decisions, num_decisions)
        except Exception as e:
            return OfflineDetectionResult([], [], e.args, 0,0)