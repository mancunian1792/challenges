'''
Importing libraries
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import math
from scipy import interpolate
from changepoint.variance_shift_model import VarianceShiftModel
import os
from scipy import interpolate
import pickle
from tqdm import tqdm


THRESHOLD = 100
MODEL_FILENAME_FULL_PATH = './model/changepoint_model.pkl'
MODEL_DIR = './model/'
MODELNAME = 'changepoint_model'
DATAPATH = "./data/exampleco_data/"
RESULTPATH = "./results/"
FIGURES_PATH = './pictures/'

'''
Helper functions
'''

def load_model():
    '''
    Instantiates or loads the changepoint model.
    '''
    model = VarianceShiftModel(50,1,5) #Set it directly after multiple experiments with it.
    return model

def save_model(model):
    outfile = open(MODEL_FILENAME_FULL_PATH, 'wb')
    pickle.dump(model, outfile)
    outfile.close()



def fill_nan(A, threshold):
    '''
    interpolate to fill nan values
    '''
    A = A.apply(lambda x: x if (abs(x) < threshold) else None)
    A = A.values
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
    B = np.where(np.isfinite(A),A,f(inds))
    return B

def med_values_without_outliers(signal, threshold):
    '''
    Use median values to fill the outliers.
    '''
    sig_no_out = signal.apply(lambda x: x if (abs(x) < threshold) else None)
    sig_no_out = [x for x in sig_no_out if x]
    med_val =  np.nanmedian(sig_no_out)
    # Filling the outliers with median values
    sig_no_out = signal.apply(lambda x: x if (pd.isna(x)==False) and (abs(x) < threshold) else med_val)
    return sig_no_out


def findFaultTimeForAMachine(data, model, mname):
    '''
    Given a set of data and timestamp, find the time at which the machine went to a fault mode.
    :param data - All the signals.
    :param model - changepoint model
    :param mname - machine name
    :returns str (timestamp in string)
    :returns int (timestamp index)
    '''
    signal_cleaned = data.iloc[:, 1:].reset_index(drop=True).apply(lambda x: fill_nan(x, THRESHOLD))
    _, ax = plt.subplots(2,2, figsize=(26,6))

    ax[0,0].plot(range(len(signal_cleaned)), signal_cleaned.iloc[:, 0])
    ax[0,1].plot(range(len(signal_cleaned)), signal_cleaned.iloc[:, 1])
    ax[1,0].plot(range(len(signal_cleaned)), signal_cleaned.iloc[:, 2])
    ax[1,1].plot(range(len(signal_cleaned)), signal_cleaned.iloc[:, 3])

    plt.savefig(FIGURES_PATH+mname+'.png')
    plt.clf()

    signal1_result = model.offlineDetection(signal_cleaned.iloc[:, 0].values, omit_na=False, supressPlot=False)
    signal2_result = model.offlineDetection(signal_cleaned.iloc[:, 1].values, omit_na=False, supressPlot=False)
    signal3_result = model.offlineDetection(signal_cleaned.iloc[:, 2].values, omit_na=False, supressPlot=False)
    signal4_result = model.offlineDetection(signal_cleaned.iloc[:, 3].values, omit_na=False, supressPlot=False)

    bkps1 = signal1_result.breakpoints[0] if len(signal1_result.breakpoints) > 0 else signal_cleaned.shape[0]
    bkps2 = signal2_result.breakpoints[0] if len(signal2_result.breakpoints) > 0 else signal_cleaned.shape[0]
    bkps3 = signal3_result.breakpoints[0] if len(signal3_result.breakpoints) > 0 else signal_cleaned.shape[0]
    bkps4 = signal4_result.breakpoints[0] if len(signal4_result.breakpoints) > 0 else signal_cleaned.shape[0]
    plt.savefig(FIGURES_PATH+mname+'_changepoint.png')
    print(bkps1, bkps2, bkps3, bkps4)

    min_fault_index = min(bkps1, bkps2, bkps3,bkps4)
    min_fault_time = data.iloc[min_fault_index, 0]
    return min_fault_time, min_fault_index


if __name__=='__main__':
    # Load the changepoint model.
    model = load_model()
    min_fault_times = []
    min_fault_indexes = []
    #Loop through the directory and load the data
    machines = os.listdir(DATAPATH)
    for machine in tqdm(machines):
        if machine == 'machine_5.csv':
            data = pd.read_csv(DATAPATH+machine)
            min_fault_time, min_fault_index = findFaultTimeForAMachine(data, model, machine)
            min_fault_times.append(min_fault_time)
            min_fault_indexes.append(min_fault_index)
    
    # Save the model to the path
    #save_model(model)

    result = pd.DataFrame({"machines": ['machine_5'], "fault_times": min_fault_times, "fault_indexes": min_fault_indexes})
    result.to_csv(RESULTPATH+'results.csv', index=False)

