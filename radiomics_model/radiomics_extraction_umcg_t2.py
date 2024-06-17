import os
import SimpleITK as sitk
import pickle
import pandas as pd
import numpy as np

# Define the path to the directory containing the CT images and GTV masks
path_to_data = "/scratch/hb-umcg_opc_outpred/UMCGOPC_MRI_nii_zscoreinHNmask"

# clinical data path 
clc_data_path  = '/home1/p303924/MRI_umcg/Data/OPCdigits_268_split_t2good.csv' 
clc_data = pd.read_csv( clc_data_path ) 

# Get a list of the CT images and GTV masks in the directory
patients_number = 268
files = clc_data[ 'PatientID' ]
IDs = range(patients_number)

# Load each CT image and GTV mask into a list
ct_images = [os.path.join(path_to_data, x + '_t2.nii.gz') for x in files]
gtv_masks = [os.path.join(path_to_data, x + '_gtvtottot2.nii.gz') for x in files]

from radiomics import featureextractor

# Create a PyRadiomics feature extractor object
extractor = featureextractor.RadiomicsFeatureExtractor()

# Specify the features you want to extract
extractor.disableAllFeatures()
extractor.enableFeatureClassByName('shape')
extractor.enableFeatureClassByName('firstorder')
extractor.enableFeatureClassByName('glcm')
extractor.enableFeatureClassByName('gldm')
extractor.enableFeatureClassByName('glrlm')
extractor.enableFeatureClassByName('glszm')
extractor.enableFeatureClassByName('ngtdm')

# Specify the discretization of the image intensities
extractor.settings["binWidth"] = 5 
extractor.settings["resampledPixelSpacing"] = (0.5, 0.5, 3.0)
 
# Specify the resampling of the image and mask
extractor.settings["interpolator"] = "sitkLinear"
extractor.settings["resamplingInterpolator"] = "sitkNearestNeighbor"
extractor.settings["correctMask"] = True

# Extract radiomics features for each CT image and GTV mask pair
all_features = pd.DataFrame()
all_features['PatientID']  = files
all_features.loc['PatientID'] ='' 
for i, ct_image, gtv_mask in zip(IDs, ct_images, gtv_masks):

  try:
    ct_image = sitk.ReadImage(ct_image)    
    
    print ( ct_image.GetSpacing() )
    
    gtv_mask = sitk.ReadImage(gtv_mask)
    
    features = extractor.execute(ct_image, gtv_mask)
 
    for feature_name in sorted(features.keys()):
    
        try: 
          all_features.loc[i,feature_name ] = features[feature_name]
        except:
          continue
  except:
    continue
              
print (all_features)
all_features.to_csv('/home1/p303924/MRI_umcg/radiomics/' + 'umcg_radiomicsfeature_t2_GTVtot.csv' ) 

'''
# Print the extracted features for each CT image and GTV mask pair
for i, features in enumerate(all_features):
    print(f"Features for CT image {i+1}:")
    for feature_name in sorted(features.keys()):
        print(f"{feature_name}: {features[feature_name]}")
    print (len(features))
'''
 
