

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter 
from lifelines.plotting import add_at_risk_counts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.linear_model import LinearRegression
from para_opts import parse_opts

def main():

  # load settings
  opt = parse_opts()

  opt.outcome = 5


  
  # Load data
  df = pd.read_csv('/Intern Programming/data//OPCdigits_t1t2_split.csv')  
  
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
  radiodata = pd.read_csv('/Intern Programming/data/output/umcg_radiomicsfeature_t1_GTVPT.csv', index_col = 'Unnamed: 0') 
  radiomics_feature_list = radiodata.columns.tolist()[2:]

# Other erosion data 1mm, 3mm and 5mm 
  """
  radiodata = pd.read_csv('/Intern Programming/data/output/umcg_radiomicsfeature_t1_GTVPTtot_1mm.csv', index_col = 'Unnamed: 0') 
  radiomics_feature_list = radiodata.columns.tolist()[37:]

  radiodata = pd.read_csv('/Intern Programming/data/output/umcg_radiomicsfeature_t1_GTVPTtot_3mm.csv', index_col = 'Unnamed: 0') 
  radiomics_feature_list = radiodata.columns.tolist()[37:]

  radiodata = pd.read_csv('/Intern Programming/data/output/umcg_radiomicsfeature_t1_GTVPTtot_5mm.csv', index_col = 'Unnamed: 0') 
  radiomics_feature_list = radiodata.columns.tolist()[37:]
  
  """


  # Merge the two dataframes based on the patient ID column 
  merged_data = pd.merge(df,radiodata, on='PatientID') 
  merged_data.to_csv('/Intern Programming/data/output/radiomics/' + 'umcg_MRI_merged_data_t2_GTVpt_' + event_vars[opt.outcome] + '.csv') 
  
  df_train = pd.merge(df_train,radiodata, on='PatientID') 
  df_test =  pd.merge(df_test,radiodata, on='PatientID') 
  
  # multivariable
 
  if opt.outcome == 0:
    sig_predictor  = ['WHO_SCORE_codes_0VS123'] 
  if opt.outcome == 1:
     sig_predictor  = ['NSTAD_codes_N01VSN2VSN3' ] 
  if opt.outcome == 2:
     sig_predictor  = ['NSTAD_codes_N01VSN2VSN3', 'WHO_SCORE_codes_0VS123',  'PACK_YEARS' ]    
  if opt.outcome == 3:
    sig_predictor  = ['NSTAD_codes_N01VSN2VSN3' ]  
  if opt.outcome == 4:
    sig_predictor  = ['WHO_SCORE_codes_0VS123', 'TSTAD_codes_123VS4', 'P16_codes_newexperi', 'NSTAD_codes_N01VSN2VSN3' ]    
  if opt.outcome == 5:
    sig_predictor  = ['P16_codes_newexperi', 'WHO_SCORE_codes_0VS123', 'TSTAD_codes_123VS4' ,'NSTAD_codes_N01VSN2VSN3']     
  if opt.outcome == 6:
    sig_predictor  = ['P16_codes_newexperi','NSTAD_codes_N01VSN2VSN3','WHO_SCORE_codes_0VS123' ]  
  if opt.outcome == 7:
    sig_predictor  = ['NSTAD_codes_N01VSN2VSN3', 'P16_codes_newexperi']
  
  #for  i, sig_predictor in enumerate(sig_predictors): 
  if 1 == 1:    
      i = opt.outcome
      cph =  CoxPHFitter()
      cph.fit(df_train[sig_predictor + [time_vars[i], event_vars[i]]], duration_col=time_vars[i], event_col=event_vars[i])
      
      cph.print_summary() 
      clc_pred_train  = cph.predict_partial_hazard(df_train[sig_predictor])
      cindex  = concordance_index(list(df_train[time_vars[i]]), - clc_pred_train , event_observed= list(df_train[event_vars[i]]) ) 
      print ('clinical train c-index ', cindex)
      
      clc_pred_test  = cph.predict_partial_hazard(df_test[sig_predictor])
      cindex  = concordance_index(list(df_test[time_vars[i]]), - clc_pred_test , event_observed= list(df_test[event_vars[i]]) ) 
      print ('clinical test c-index ', cindex) 
  
  
          
## FOr T1


  if opt.outcome == 0:
      radio_predictor = [ 'original_glszm_ZoneEntropy']  
  if opt.outcome == 1:
      radio_predictor = [ 'original_firstorder_Energy']    
  if opt.outcome == 2:
      radio_predictor = [ 'original_glszm_ZoneEntropy' ]    
  if opt.outcome == 3:
      radio_predictor = ['original_glszm_ZoneEntropy']  
  if opt.outcome == 4:
      radio_predictor = ['original_glrlm_RunEntropy', 'original_firstorder_Kurtosis' ] 
  if opt.outcome == 5:
      radio_predictor = ['original_glszm_ZoneEntropy']  
  if opt.outcome == 6:
      radio_predictor = ['original_gldm_SmallDependenceEmphasis', 'original_firstorder_Kurtosis' ]        
  if opt.outcome == 7:
      radio_predictor = ['original_firstorder_Range' ] 

# ## for T2
#   if opt.outcome == 0:
#       radio_predictor = [ 'original_glszm_ZoneEntropy']  
#   if opt.outcome == 1:
#       radio_predictor = [ 'original_firstorder_RootMeanSquared']    
#   if opt.outcome == 2:
#       radio_predictor = [ 'original_firstorder_Range' ]    
#   if opt.outcome == 3:
#       radio_predictor = ['original_glszm_ZoneEntropy' ]  
#   if opt.outcome == 4:
#       radio_predictor = ['original_glrlm_RunEntropy', 'original_firstorder_Kurtosis' ] 
#   if opt.outcome == 5:
#       radio_predictor = ['original_glszm_ZoneEntropy']  
#   if opt.outcome == 6:
#       radio_predictor = ['original_gldm_SmallDependenceEmphasis', 'original_firstorder_Kurtosis' ]        
#   if opt.outcome == 7:
#       radio_predictor = ['original_firstorder_Range' ] 


  
      

  cph =  CoxPHFitter( )
  
  cph.fit(df_train[radio_predictor + [time_vars[i], event_vars[i]]], duration_col=time_vars[i], event_col=event_vars[i])
      
  cph.print_summary() 
  
  com_pred_train  = cph.predict_partial_hazard(df_train[radio_predictor])
  cindex  = concordance_index(list(df_train[time_vars[i]]), - com_pred_train , event_observed= list(df_train[event_vars[i]]) ) 
  print ('radiomics train c-index ', cindex)    
  
  com_pred_test  = cph.predict_partial_hazard(df_test[radio_predictor])
  cindex  = concordance_index(list(df_test[time_vars[i]]), - com_pred_test , event_observed= list(df_test[event_vars[i]]) ) 
  print ('radiomics test c-index ', cindex)      
  
  
  
      
  # add clc_pred and radiomics_pred to a cox model
  sig_predictor = sig_predictor + radio_predictor
  
  cph =  CoxPHFitter( )
  
  cph.fit(df_train[sig_predictor + [time_vars[i], event_vars[i]]], duration_col=time_vars[i], event_col=event_vars[i])
      
  cph.print_summary() 
  
  com_pred_train  = cph.predict_partial_hazard(df_train[sig_predictor])
  cindex  = concordance_index(list(df_train[time_vars[i]]), - com_pred_train , event_observed= list(df_train[event_vars[i]]) ) 
  print ('combined train c-index ', cindex)
      
  com_pred_test  = cph.predict_partial_hazard(df_test[sig_predictor])
  cindex  = concordance_index(list(df_test[time_vars[i]]), - com_pred_test , event_observed= list(df_test[event_vars[i]]) ) 
  print ('combined test c-index ', cindex)       
  
  p_values = cph.summary['p']
  
  print ( p_values )
  

  
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