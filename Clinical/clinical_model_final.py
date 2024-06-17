

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import pandas as pd
import numpy as np
import scipy

from para_opts import parse_opts 
opt = parse_opts(  ) 

# Load data
df = pd.read_csv('/Intern Programming/data/OPCdigits_333_split_t1good.csv') 
#df = pd.read_csv('./Data/OPCdigits_268_split_t2good.csv') 

df_train = df.loc[df['CVgroup_event'] == 'train'] 
df_test = df.loc[df['CVgroup_event'] == 'test'] 

# Define variables
time_vars = ['TIME_LR','TIME_RR','TIME_LRR','TIME_MET','TIME_TumorSpecificSurvival','TIME_OS','TIME_DFS','TIME_RFS']
event_vars = ['LR_code','RR_code','LRR_code','MET_code','TumorSpecificSurvival_code','OS_code','DFS_code','RFS_code']
covariates = ['AGE', 'GESLACHT_codes' ,'PACK_YEARS','TSTAD_codes_123VS4','NSTAD_codes_N01VSN2VSN3','P16_codes_newexperi','WHO_SCORE_codes_0VS123']

# multivariable
if opt.outcome == 0:
  sig_predictor  = ['WHO_SCORE_codes_0VS123','P16_codes_newexperi','TSTAD_codes_123VS4', 'PACK_YEARS'] 
if opt.outcome == 1:
  sig_predictor  = ['NSTAD_codes_N01VSN2VSN3', 'PACK_YEARS' ] 
if opt.outcome == 2:
  sig_predictor  = ['PACK_YEARS', 'NSTAD_codes_N01VSN2VSN3', 'WHO_SCORE_codes_0VS123']  
if opt.outcome == 3:
  sig_predictor  = ['NSTAD_codes_N01VSN2VSN3']  
if opt.outcome == 4:
  sig_predictor  = ['NSTAD_codes_N01VSN2VSN3', 'WHO_SCORE_codes_0VS123', 'TSTAD_codes_123VS4', 'PACK_YEARS']    
if opt.outcome == 5:
  sig_predictor  = ['P16_codes_newexperi' ,'TSTAD_codes_123VS4', 'WHO_SCORE_codes_0VS123', 'NSTAD_codes_N01VSN2VSN3']   
if opt.outcome == 6:
  sig_predictor  = ['NSTAD_codes_N01VSN2VSN3', 'P16_codes_newexperi', 'WHO_SCORE_codes_0VS123', 'TSTAD_codes_123VS4']  
if opt.outcome == 7:
  sig_predictor  = ['NSTAD_codes_N01VSN2VSN3', 'PACK_YEARS', 'P16_codes_newexperi' ]    

#for  i, sig_predictor in enumerate(sig_predictors):
if 1 == 1:    
    i = opt.outcome
    cph =  CoxPHFitter()
    cph.fit(df_train[sig_predictor + [time_vars[i], event_vars[i]]], duration_col=time_vars[i], event_col=event_vars[i])
    
    cph.print_summary() 
    clc_pred_train  = cph.predict_partial_hazard(df_train[sig_predictor])
    cindex  = concordance_index(list(df_train[time_vars[i]]), - clc_pred_train , event_observed= list(df_train[event_vars[i]]) ) 
    print ('Clinical model train c-index ', cindex)
    
    clc_pred_test  = cph.predict_partial_hazard(df_test[sig_predictor])
    cindex  = concordance_index(list(df_test[time_vars[i]]), - clc_pred_test , event_observed= list(df_test[event_vars[i]]) ) 
    print ('Clinical model test c-index ', cindex)     
    



