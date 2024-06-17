

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter 
from lifelines.plotting import add_at_risk_counts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
# from sklearn.linear_model import LinearRegression
from para_opts import parse_opts

def main():

  # load settings
  opt = parse_opts()
  opt.outcome = 5

  # Load data
  df = pd.read_csv('/Intern Programming/data/OPCdigits_t1t2_split.csv') 
  df_train = df.loc[df['CVgroup_event_t1t2'] == 'train'] 
  df_test = df.loc[df['CVgroup_event_t1t2'] == 'test'] 
   
  # Define variables
  time_vars = ['TIME_LR','TIME_RR','TIME_LRR','TIME_MET','TIME_TumorSpecificSurvival','TIME_OS','TIME_DFS','TIME_RFS']
  event_vars = ['LR_code','RR_code','LRR_code','MET_code','TumorSpecificSurvival_code','OS_code','DFS_code','RFS_code']
  covariates = ['AGE', 'GESLACHT_codes' ,'PACK_YEARS','TSTAD_codes_123VS4','NSTAD_codes_N01VSN2VSN3','P16_codes_newexperi','WHO_SCORE_codes_0VS123']
  
  real_event_names_short = ['LC', 'RC', 'LRC', 'DMFS', 'TSS', 'OS', 'DFS', 'RFS']
  
  sig_predictors = [ ] 
  
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

  ################### radiomics part
  radiodata = pd.read_csv('/Intern Programming/data/output/stabled_radiomics_features_t2.csv')   
  radiomics_feature_list = radiodata.columns.tolist()[2:]

  # Merge the two dataframes based on the patient ID column (for check)
#   merged_data = pd.merge(df,radiodata, on='PatientID') 
#   merged_data.to_csv('/home1/p303924/MRI_umcg_new/radiomics/' + 'umcg_data_t1_GTVpt_' + event_vars[opt.outcome] + '.csv') 

  merged_data = pd.merge(df,radiodata, on='PatientID') 
  merged_data.to_csv('/Intern Programming/data/output/radiomics/' + 'stabled_radiomics_features_t2_GTVPT_' + event_vars[opt.outcome] + '.csv') 
  
  df_train = pd.merge(df_train,radiodata, on='PatientID') 
  df_test =  pd.merge(df_test,radiodata, on='PatientID') 

  #step 0: select stable features (), calculate correlation of radiomics features before and after erosion of GTV

  #Step 1: Univariate Analysis
  
  # Load your dataset into a Pandas dataframe
  #data = traindata 
  data = df_train 
  
  # Instantiate a CoxPHFitter object
  cph = CoxPHFitter()
  
  # Fit the Cox model with each variable separately
  significant_vars = []
  
  for var in radiomics_feature_list: 
      
      print ('var:' , var)
       
      try: 
        cph.fit(data,  duration_col=time_vars[opt.outcome], event_col=event_vars[opt.outcome], formula=var)
        # Print the hazard ratios and confidence intervals for each variable
        # Get the p-values for each variable
      
        p_values = cph.summary['p']
        significant_var = list(p_values[p_values < 0.05].index)
      
        significant_vars.append(significant_var[0])
      except:
        continue
      
  significant_vars = list(significant_vars)

  print (significant_vars)
  
  #Step 2: Correlation and BIC-Based Variable Selection
  
  # Compute the pairwise correlations between variables
  correlations = data[significant_vars].corr()
  
  # Find pairs of variables with correlation coefficient greater than 0.8
  high_corr_pairs = []
  for i in range(len(correlations.columns)):
      for j in range(i+1, len(correlations.columns)):
          if abs(correlations.iloc[i,j]) > 0.8:
              high_corr_pairs.append((correlations.columns[i], correlations.columns[j]))
  
  #print (high_corr_pairs)
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
         
      print (significant_vars)
  
  def stepwise_bootstrap(X, significant_vars, criterion):
          # Generate a bootstrap sample
          boot_idx = np.random.choice(range(X.shape[0]), size=X.shape[0], replace=True)
          X_boot = X.iloc[boot_idx]
  
          # Initialize a cox model        
          cox_model = CoxPHFitter()    
          cox_model.fit(X_boot, duration_col=time_vars[opt.outcome], event_col=event_vars[opt.outcome], formula= [significant_vars[0]])
          
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
                print ('p value of log-likelihood ratio test', p_val)    
              else:
                p_val= 1
                bic = 10000
              
              print (best_candidate)
              
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
         try:   
              selected_features = stepwise_bootstrap(data, significant_vars, criterion = 'bic')
              #print (selected_features )
              selected_features_all.append(selected_features)
              #print (selected_features_all,)
              for f in list(selected_features):
                  Feature_freq[f]=Feature_freq[f]+1
              Feature_freq_sorted = Feature_freq
         except:
              continue                          
      print (Feature_freq_sorted)    
      
      Feature_freq_sorted = pd.DataFrame( Feature_freq_sorted.items(), columns=['Key', 'Value'] )      
      Feature_freq_sorted.to_csv('/Intern Programming/data/output/radiomics/radiomics_sorted_freq_t2_gtvpt_' + event_vars[opt.outcome] + '.csv', index = True, header=True) 
      
  '''
      Feature_freq_sorted = pd.DataFrame.from_dict(Feature_freq_sorted)       
      Feature_freq_sorted.to_csv('selected_radio_'  + str(opt.outcome)  + '.csv', index = True, header=True)        
  '''
  
def conf_cindex(test_predictions, ground_truth_y,ground_truth_e, bootstrap=1000, seed=None,  confint=0.95):
    """Takes as input test predictions, ground truth, number of bootstraps, seed, and confidence interval"""
    #inspired by https://stackoverflow.com/questions/19124239/scikit-learn-roc-curve-with-confidence-intervals by ogrisel
    bootstrapped_scores = []
    rng = np.random.RandomState(seed)
    if confint>1:
        confint=confint/100
    for i in range(bootstrap):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(test_predictions) - 1, len(test_predictions))
      
        if len(np.unique(ground_truth_y[indices])) < 2:
            continue
      
        #score = metrics.roc_auc_score(ground_truth[indices], test_predictions[indices])
        try:
           # For RC, sometimes no event selected, so mistake happens
           score = concordance_index(ground_truth_y[indices], test_predictions[indices], ground_truth_e[indices])
           bootstrapped_scores.append(score)
        except:
           continue
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    lower_bound=(1-confint)/2
    upper_bound=1-lower_bound
    confidence_lower = sorted_scores[int(lower_bound * len(sorted_scores))]
    confidence_upper = sorted_scores[int(upper_bound * len(sorted_scores))]
    auc = concordance_index(ground_truth_y, test_predictions, ground_truth_e)
    print("{:0.0f}% confidence interval for the score: [{:0.3f} - {:0.3}] and your cindex is: {:0.3f}".format(confint*100, confidence_lower, confidence_upper, auc))
    confidence_interval = (confidence_lower, auc, confidence_upper)
    return confidence_interval, sorted_scores  
  
from scipy.stats import chi2

def HosmerLemeshow(obseved ,expected,bins = 5, strategy = "quantile") :
    pihat=obseved
    Y = expected
    pihatcat=pd.cut(pihat, np.percentile(pihat,[0,20,40,60,80,100]),labels = False,include_lowest=True) #here we've chosen only 4 groups
    
    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, bins + 1)
        pihatcat = np.percentile(obseved, quantiles * 100)
    elif strategy == "uniform":
        pihatcat = np.linspace(0.0, 1.0, bins + 1)
    pihatcat = np.searchsorted(pihatcat[1:-1], obseved)

    meanprobs =[0]*bins 
    expevents =[0]*bins
    obsevents =[0]*bins 
    meanprobs2=[0]*bins 
    expevents2=[0]*bins
    obsevents2=[0]*bins 
    #points
    expprobs =[0]*bins
    obsprobs =[0]*bins 
    

    for i in range(bins):
       meanprobs[i]=np.mean(pihat[pihatcat==i])
       expevents[i]=np.sum(pihatcat==i)*np.array(meanprobs[i])
       obsevents[i]=np.sum(Y[pihatcat==i])
       meanprobs2[i]=np.mean(1-pihat[pihatcat==i])
       expevents2[i]=np.sum(pihatcat==i)*np.array(meanprobs2[i])
       obsevents2[i]=np.sum(1-Y[pihatcat==i]) 
       
       expprobs[i] = np.sum(Y[pihatcat==i]) / len(Y[pihatcat==i])
       obsprobs[i] = np.mean(pihat[pihatcat==i])

    data1={'meanprobs':meanprobs,'meanprobs2':meanprobs2}
    data2={'expevents':expevents,'expevents2':expevents2}
    data3={'obsevents':obsevents,'obsevents2':obsevents2}
    m=pd.DataFrame(data1)
    e=pd.DataFrame(data2)
    o=pd.DataFrame(data3)
    
    # The statistic for the test, which follows, under the null hypothesis,
    # The chi-squared distribution with degrees of freedom equal to amount of groups - 2. Thus 4 - 2 = 2
    tt=sum(sum((np.array(o)-np.array(e))**2/np.array(e))) 
    pvalue=1-chi2.cdf(tt,int(bins) - 2)

    return pd.DataFrame([[chi2.cdf(tt,2).round(2), pvalue.round(2)]],
                        columns = ["Chi2", "p - value"]), expprobs, obsprobs #expevents,  obsevents
                        
if __name__ == '__main__':
	main()