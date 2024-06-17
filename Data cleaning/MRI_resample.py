
import numpy as np
import SimpleITK as sitk
import os

def resample_image(itk_image, out_spacing=[1.0, 1.0, 1.0], is_label=False):
    original_spacing = itk_image.GetSpacing()
    
    original_size = itk_image.GetSize()
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]
  
    '''
    # keep the same size with Spacing_function in MONAI.  
    out_size = [
        int(original_size[0] * (original_spacing[0] / out_spacing[0])),
        int(original_size[1] * (original_spacing[1] / out_spacing[1])),
        int(original_size[2] * (original_spacing[2] / out_spacing[2]))
    ]
    '''
    
    print ( out_size )

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    #resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        resample.SetDefaultPixelValue( 0 )
    else:
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetDefaultPixelValue( np.min( sitk.GetArrayFromImage(itk_image)) )

    return resample.Execute(itk_image) 



# for all patients together for t1 and t2
# load path
path_to_sitk_image = "/Intern Programming/data/UMCGOPC_MRI_nii_zscoreinHNmask"
# save path
path_to_save  = r'E:\Intern Programming\data\output\UMCGOPC_MRI_nii_zscoreinHNmask_1mm'
image_names = os.listdir( path_to_sitk_image )

image_names =  sorted( image_names )

for image_name in image_names:
    image_name_long = os.path.join( path_to_sitk_image, image_name )
    image =  sitk.ReadImage( image_name_long )
    newimage  = resample_image( image, out_spacing = [1.0, 1.0, 1.0], is_label = False )
    sitk.WriteImage( newimage, os.path.join( path_to_save, image_name[ 0 : 15] + 'r.nii.gz'  ) )   



# for all patients together for t1_gtv and t2_gtv
# load path
path_to_sitk_image = "/Intern Programming/data/t1t2_gtv"
path_to_save  = r'E:\Intern Programming\data\output\t1t2gtv_1mm'
image_names = os.listdir( path_to_sitk_image )   

image_names =  sorted( image_names )   
#image_names = [ 'UMCG-2116597_gtvpttot1.nii.gz' ] 

for image_name in image_names:   
    
    image_name_long = os.path.join( path_to_sitk_image, image_name )   
    image =  sitk.ReadImage( image_name_long )   
    newimage  = resample_image( image, out_spacing = [1.0, 1.0, 1.0], is_label = True )   
    
    name_part, extension_part = image_name.split('.', 1)   
    newimage[ newimage > 0 ] = 1   
    
    print ( sitk.GetArrayFromImage(newimage).shape )   
    sitk.WriteImage( newimage, os.path.join( path_to_save, name_part + 'r.nii.gz'  ) )   

    