# the required python libraries imported
import bnpy
import pandas as pd
import numpy as np
import os
import time
from glob import glob

# G == 0 SMOVB
# G == 1 SMOVB and Cohesion Function Mark I
# G == 2 SMOVB and Cohesion Funciton Mark II
# batch_size is the number of datapoint in a batch... five is the lowest it is test at
# window_size_in_batches is the number of batches in a window... five is the lowest it is tested at
# both batch_size and window_size_in_batches have only been test at multiples of 5 greater than 5
G=0
batch_size = 5
window_size_in_batches = 5
windows = []


# TRDPMAD is ready for multiprocessor (python MP) but for this demonstration file
# is run only in single processor mode. 
def TRDPMAD(mppack):
    # Get the argument as an MP job
    windows = mppack[0]
    batch_size = mppack[1]
    window_size_in_batches = mppack[2]
    data_set = mppack[3]
    G = mppack[4]

    # Initialize some parameters
    wds = len(windows)
    # bnpy hyper-parameters
    gamma = 1.0
    sF = 1.0
    # Initialize K component - this value places a max K the model can develop
    K = 25      
    # Number of iterations for bnpy
    nLap = 1
    # Accounting info for the bnpy coldstart/warmstart save to file function
    ds = data_set[0] + "." + data_set[1]

    # Init params provided by bnpy
    iname='randexamples'
    # Where to find the output from the coldstarted model
    opath = f'/tmp/bnp-anomaly/coldstart/{G}/{ds}/b0'  # Dynamic output path according to batch
    # Lists for out variables of interst 
    # ll == log likelihook of the posterio
    # a_all == approximated log likelihood 
    ll = [np.nan] * window_size_in_batches
    a_ll = [np.nan] * window_size_in_batches

    # output dataframes get written to the .csv
    data_df = pd.DataFrame()
    calc_df = pd.DataFrame()
    
    # for each window of data passed in
    for ii, window in enumerate(windows):
        # get data indexes
        df_index = window[0]
        # get bnpy xData object with windowed data
        xdata_data = window[1]

        # provide run status to stdout
        if ii % 5 == 0:
            print("XXX" + str(data_set)+ " " + str(ii)+"/"+str(wds))

        # run the bnpy model, initially with coldstart params, and then with warmstart params
        # include birth, merge, delete, shuffle component moves
        warm_start_model, warm_info_dict = bnpy.run(
            xdata_data, 'DPMixtureModel', 'DiagGauss', 'memoVB',
            output_path=opath,
            nLap=nLap, nTask=1, nBatch=window_size_in_batches, convergeThr=0.0001,
            gamma0=gamma, sF=sF, ECovMat='eye',
            K=K, 
            #moves='birth,merge,delete,shuffle',
            initname=iname,
            ts=True, debug=False, verbose=0, G=G)

        # Change the init params pointers after init coldstart run         
        iname=warm_info_dict['task_output_path']
        opath = f'/tmp/bnp-anomaly/warmstart/{G}/{ds}/b{ii +  1}'

        # From the window of batches, get the new batch and run stats on it
        # to check for anomaly
        batch = xdata_data.make_subset(list(range(batch_size * window_size_in_batches - batch_size, batch_size * window_size_in_batches)))
        LP = warm_start_model.calc_local_params(batch)
        SS = warm_start_model.get_global_suff_stats(batch, LP)
        # Get entropy of log likelihood
        LL = warm_start_model.calcLogLikCollapsedSamplerState(SS)
        ll.pop(0)
        ll.append(LL)
        ll_normed = [i/sum(ll) + 1e-10 for i in ll]
        entropy_ll = -sum([i*np.log(i) for i in ll_normed])

        # Calculate approximate log likelihood and get its entropy 
        approx_ll = warm_start_model.calc_evidence(batch)
        a_ll.pop(0)
        a_ll.append(approx_ll)
        a_ll_normed = [i/sum(a_ll) + 1e-10 for i in a_ll]
        entropy_a_ll = -sum([i*np.log(i) for i in a_ll_normed])
  
        # Put all the required info in a dataframe
        index = np.array(df_index.iloc[-1], dtype=int)
        x = df_index[len(df_index) - batch_size:]
        xx = df_index[-1:]
        data_df = data_df.append(pd.DataFrame({'x':x, 'y':batch.X.reshape((batch_size,))}), ignore_index=True)
        calc_df = calc_df.append(pd.DataFrame({'x':xx, 'll':LL, 'a_ll':approx_ll, 'e_ll':entropy_ll, 'e_a_ll':entropy_a_ll}), ignore_index=True)

    # Write the dataframes info to .csv - open it with pandas and get graphing
    ds = data_set[0] + "." + data_set[1]
    name = "data/test_output/" + ds + "_alg-" + str(G) +  "_bs-" + str(batch_size) \
                + "_wsib-" + str(window_size_in_batches) + ".csv"
    data_df.set_index('x', inplace=True)
    calc_df.set_index('x', inplace=True)
    pd.concat([data_df, calc_df],axis=1, sort=False).to_csv(name)
    return 0

# find the datasets in test - recommend doing one at a time using any of 
# [ds00,  ds02,  ds04, ds06, ds01, ds03, ds05, ds07] for file match
test_data_file = "data/test/ds00.csv"
test_data_names = [["ds00", "test"]]
data = [pd.read_csv(test_data_file)]

for d, df in enumerate(data):
    win = []
    i = 0
    while i * batch_size <= (len(df) - window_size_in_batches * batch_size):
        w= df[i * batch_size:i * batch_size + window_size_in_batches * batch_size]
        y = w.drop(columns=['Unnamed: 0'])
        yy = bnpy.data.XData.from_dataframe(y)
        x = w['Unnamed: 0']
        win.append([x, yy])
        i += 1
    windows.append([win, batch_size, window_size_in_batches, test_data_names[d], G])

for w in windows:
    TRDPMAD(w)
