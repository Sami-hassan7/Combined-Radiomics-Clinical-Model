

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

df_train = df.loc[df['CVgroup_event'] == 'train'] # training set
df_test = df.loc[df['CVgroup_event'] == 'test']  # testing set

# Define variables
time_vars = ['TIME_LR','TIME_RR','TIME_LRR','TIME_MET','TIME_TumorSpecificSurvival','TIME_OS','TIME_DFS','TIME_RFS'] # time-to-event name
event_vars = ['LR_code','RR_code','LRR_code','MET_code','TumorSpecificSurvival_code','OS_code','DFS_code','RFS_code'] # event name
covariates = ['AGE', 'GESLACHT_codes' ,'PACK_YEARS','TSTAD_codes_123VS4','NSTAD_codes_N01VSN2VSN3','P16_codes_newexperi','WHO_SCORE_codes_0VS123'] # potential clinical predictors


# step 1: univariable analysis
sig_predictors  = []
for time_var, event_var in zip(time_vars, event_vars):
    
    sig_predictor  = []
    #print ('Univariable analysis for ', time_var, ' :') 
    # Fit Cox proportional hazards regression model
    cph = CoxPHFitter()
    for covariate in covariates:
        cph.fit(df_train[[covariate] + [time_var, event_var]], duration_col=time_var, event_col=event_var)

        # Print coefficients and p-values for each covariate
        #print(cph.summary.p)
        if  float(cph.summary.p) < 0.05:
            sig_predictor.append(covariate)
    sig_predictors.append(sig_predictor)               
print (sig_predictors)

# The result of step 1
sig_predictors = [['PACK_YEARS', 'TSTAD_codes_123VS4', 'P16_codes_newexperi', 'WHO_SCORE_codes_0VS123'],  # LR   
 ['PACK_YEARS', 'NSTAD_codes_N01VSN2VSN3'],                                                               # RR   
 ['PACK_YEARS', 'NSTAD_codes_N01VSN2VSN3', 'P16_codes_newexperi', 'WHO_SCORE_codes_0VS123'],              # LRR  
 ['NSTAD_codes_N01VSN2VSN3'],                                                                             # DMFS 
 ['GESLACHT_codes', 'PACK_YEARS', 'TSTAD_codes_123VS4', 'NSTAD_codes_N01VSN2VSN3', 'P16_codes_newexperi', 'WHO_SCORE_codes_0VS123'], # TSS 
 ['PACK_YEARS', 'TSTAD_codes_123VS4', 'NSTAD_codes_N01VSN2VSN3', 'P16_codes_newexperi', 'WHO_SCORE_codes_0VS123'],                   # OS  
 ['PACK_YEARS', 'TSTAD_codes_123VS4', 'NSTAD_codes_N01VSN2VSN3', 'P16_codes_newexperi', 'WHO_SCORE_codes_0VS123'],                   # DFS 
 ['PACK_YEARS', 'TSTAD_codes_123VS4', 'NSTAD_codes_N01VSN2VSN3', 'P16_codes_newexperi', 'WHO_SCORE_codes_0VS123']]                   # RFS   


# univaraible analysis result
if opt.outcome == 0:
  sig_predictor  = ['PACK_YEARS', 'TSTAD_codes_123VS4', 'P16_codes_newexperi', 'WHO_SCORE_codes_0VS123'] 
if opt.outcome == 1:
  sig_predictor  = ['PACK_YEARS', 'NSTAD_codes_N01VSN2VSN3'] 
if opt.outcome == 2:
  sig_predictor  = ['PACK_YEARS', 'NSTAD_codes_N01VSN2VSN3', 'P16_codes_newexperi', 'WHO_SCORE_codes_0VS123']  
if opt.outcome == 3:
  sig_predictor  = ['NSTAD_codes_N01VSN2VSN3']  
if opt.outcome == 4:
  sig_predictor  = ['GESLACHT_codes', 'PACK_YEARS', 'TSTAD_codes_123VS4', 'NSTAD_codes_N01VSN2VSN3', 'P16_codes_newexperi', 'WHO_SCORE_codes_0VS123']    
if opt.outcome == 5:
  sig_predictor  = ['PACK_YEARS', 'TSTAD_codes_123VS4', 'NSTAD_codes_N01VSN2VSN3', 'P16_codes_newexperi', 'WHO_SCORE_codes_0VS123']  
if opt.outcome == 6:
  sig_predictor  = ['PACK_YEARS', 'TSTAD_codes_123VS4', 'NSTAD_codes_N01VSN2VSN3', 'P16_codes_newexperi', 'WHO_SCORE_codes_0VS123']  
if opt.outcome == 7:
  sig_predictor  = ['PACK_YEARS', 'TSTAD_codes_123VS4', 'NSTAD_codes_N01VSN2VSN3', 'P16_codes_newexperi', 'WHO_SCORE_codes_0VS123']    


significant_vars = sig_predictor  
data = df_train  

#Step 2: Correlation and BIC-Based Variable Selection

# Compute the pairwise correlations between variables
correlations = data[significant_vars].corr()

# Find pairs of variables with correlation coefficient greater than 0.8
high_corr_pairs = []
for i in range(len(correlations.columns)):
    for j in range(i+1, len(correlations.columns)):
        if abs(correlations.iloc[i,j]) > 0.8:
            high_corr_pairs.append((correlations.columns[i], correlations.columns[j]))

print (significant_vars)
# For each pair of highly correlated variables, delete the one with higher BIC in the Cox model
for var1, var2 in high_corr_pairs:
    cph1 = CoxPHFitter()
    cph2 = CoxPHFitter()
    
    cph1.fit(data, duration_col=time_vars[opt.outcome], event_col=event_vars[opt.outcome], formula=var1)
    cph2.fit(data, duration_col=time_vars[opt.outcome], event_col=event_vars[opt.outcome], formula=var2)
    
    #bic1 = cph1.summary.iloc[ -1, 1 ] 
    #bic2 = cph2.summary.iloc[ -1, 1 ] 
    
    log_likelihood1  = cph1.log_likelihood_
    bic1  = -2 * log_likelihood1  + np.log(len(data)) * len(var1)
    
    log_likelihood2  = cph2.log_likelihood_
    bic2  = -2 * log_likelihood2  + np.log(len(data)) * len(var2)    
    
    print (bic1, bic2)
    
    if bic1 > bic2:
        var_to_drop = var1
    else:
        var_to_drop = var2
        
    if var_to_drop in significant_vars:   
       significant_vars.remove(var_to_drop)
       
print (significant_vars) # get the predictors without high correlation


def stepwise_bootstrap(X, significant_vars, criterion):
        # Generate a bootstrap sample
        boot_idx = np.random.choice(range(X.shape[0]), size=X.shape[0], replace=True)
        X_boot = X.iloc[boot_idx]

        # Initialize a cox model        
        cox_model = CoxPHFitter()    
        cox_model.fit(X_boot, duration_col=time_vars[opt.outcome], event_col=event_vars[opt.outcome], formula= ['GESLACHT_codes'] ) 
        
        # Perform stepwise selection
        selected_features = []
        i = 0
        while True:
            remaining_features = list(set([f for f in significant_vars if f not in selected_features]))
            if len(remaining_features) == 0:
                break
            candidate_scores = []
            for f in remaining_features:
                features = selected_features + [f]
                cox_model_candidate  = CoxPHFitter()    
                cox_model_candidate = cph.fit(X_boot, duration_col=time_vars[opt.outcome], event_col=event_vars[opt.outcome], formula= features)   # show_progress=True,
                
                # bic 
                log_likelihood  = cox_model_candidate.score( X_boot[features + [time_vars[opt.outcome] , event_vars[opt.outcome] ]  ])
                BIC  = -2 * log_likelihood  + np.log(len(X_boot)) *len(features)
                                                    
                candidate_scores.append(( BIC, f))
            #print (candidate_scores)        
            best_candidate = sorted(candidate_scores)[0]
            
            if selected_features != []:
               cox_model = CoxPHFitter() 
               cox_model.fit(X_boot, duration_col=time_vars[opt.outcome], event_col=event_vars[opt.outcome], formula=selected_features) # show_progress=True,
            
            # bic 
            if i> -1:
              #log_likelihood  = cox_model.score(X_boot[selected_features + [time_vars[opt.outcome] , event_vars[opt.outcome] ]])
              log_likelihood  = cox_model.log_likelihood_
              #print (log_likelihood)
              bic  = -2 * log_likelihood  + np.log(len(X_boot)) * len(selected_features)
              
              #print (bic, best_candidate)
               
              #calculate likelihood ratio Chi-Squared test statistic
              #LR_statistic = -2*(cox_model.score(X_boot[selected_features+ [time_vars[opt.outcome] , event_vars[opt.outcome] ]])- cox_model_candidate.score(X_boot[features+ [time_vars[opt.outcome] , event_vars[opt.outcome] ]]))

              LR_statistic = -2*(cox_model.log_likelihood_- cox_model_candidate.log_likelihood_)

              #calculate p-value of test statistic using 1 degrees of freedom
              p_val = scipy.stats.chi2.sf(LR_statistic, 1)
              #p_val = 1 - scipy.stats.chi2.cdf(LR_statistic, 1)
              #print ('p value of log-likelihood ratio test', p_val)
            else:
              p_val= 1
              bic = 10000
            
            #print (best_candidate)
            
            if best_candidate[0] < bic and p_val < 0.05:
            #if best_candidate[0] < bic:
                 print ('Comparing ')
                 selected_features.append(best_candidate[1])       
                 print ('current selected features:', selected_features)
            elif i == 0:
                 selected_features.append(best_candidate[1])       
                 print ('current selected features:', selected_features)
            else: 
                 break
            i +=1        
        return selected_features


Feature_freq_sorted = {}
# step 3, multi-variable selection
# set up data, time_var, event_var, and list of features

if 1 == 1:

    # then do multi-variable radiomcis feature selection       
    Feature_freq = {}.fromkeys(data[significant_vars].columns, 0)       
    n_bootstraps = 1000 
    scores = []
    selected_features_all = []
    for i in range(n_bootstraps):
       print ('Bootstrapping step ' , i)
       #try: 
       if 1 == 1:   
            selected_features = stepwise_bootstrap(data, significant_vars, criterion = 'bic')
            #print (selected_features )
            selected_features_all.append(selected_features) 
            #print (selected_features_all,)
            for f in list(selected_features):
                Feature_freq[f]=Feature_freq[f]+1
            Feature_freq_sorted = Feature_freq 
       #except:
            #continue                          
    print (Feature_freq_sorted)    

    #Feature_freq_sorted = pd.DataFrame.from_dict(Feature_freq_sorted)       
    Feature_freq_sorted = pd.DataFrame( Feature_freq_sorted.items(), columns=['Key', 'Value'] )      
    Feature_freq_sorted.to_csv('/Intern Programming/data/selected_clcpara_'  + str( event_vars[opt.outcome] )  + '_t1.csv', index = True, header=True) 