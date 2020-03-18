import numpy as np
import scipy.stats as stat
from joblib import Parallel, delayed

GAUSSIAN_TOLERANCE = 1e-4
MIN_SAMPLES_REQUIRED = 30

def parallelize_func(iterable, func, args=[], chunksz=1, n_jobs=16):
    '''
    Parallelize a function over each element of the iterable
    '''
    chunker = func
    chunks_results = Parallel(n_jobs=n_jobs, verbose=False)(
        delayed(chunker)(chunk, *args) for chunk in iterable
    )
    return chunks_results

def ts_stats_significance(ts, ts_stat_func, null_ts_func=None, B=1000, permute_fast=False, bootstrap=False):
    '''
    Compute the statistical significance of a test statistic at each point of the time series
    '''
    stats_ts = ts_stat_func(ts)
    if bootstrap:
        if permute_fast:
            # Permute it in 1 shot 
            null_ts = list(map(np.random.permutation, np.array([ts, ]*B)))
        else:
            null_ts = np.vstack([null_ts_func(ts) for _ in np.arange(0, B)])
        stats_null_ts = np.vstack([ts_stat_func(nts) for nts in null_ts])
        pvals = []
        for i in np.arange(0, len(stats_ts)):
            num_samples = np.sum((stats_null_ts[:,1] >= stats_ts[i]))
            pval = round(num_samples/float(B), 3)
            pvals.append(pval)
        result = {"statistics": stats_ts, "pvalue": pvals}
        return result
    else:
        result  = {"statistics": stats_ts, "pvalue": None}
        return result

def gaussanityCheck(points):
    if len(list(points)) < MIN_SAMPLES_REQUIRED:
        return False
    result = stat.normaltest(points)
    if result.pvalue < GAUSSIAN_TOLERANCE:
        # Rejecting the null hypothesis, that its coming from gaussian distribution.
        return False
    return True