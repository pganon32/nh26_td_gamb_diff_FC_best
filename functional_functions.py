"""
functions for functional team 
cii to correlation matrix functions
"""

#requires: 
import numpy as np
from cloudpathlib import S3Path, S3Client
import nibabel as nib
import neuropythy as ny


def ciis_to_giis(patient_num):
    """
    splits cii files for both LR and RL 
    returns a tuple of one LH and one RH timeseries
    (concatenated LR and RL in the time dimension) 
    """
    # The base HCP path:
    base_path = S3Path(
    's3://hcp-openaccess/HCP_1200/',
    client=S3Client(profile_name='hcp'))

    # load in the patient data 
    # both LR and RL filepaths 
    RL_dirpath = base_path / str(patient_num) / 'MNINonLinear' / 'Results' / 'tfMRI_GAMBLING_RL' 
    RL_filepath = RL_dirpath / 'tfMRI_GAMBLING_RL_Atlas_MSMAll.dtseries.nii'
    
    LR_dirpath = base_path / str(patient_num) / 'MNINonLinear' / 'Results' / 'tfMRI_GAMBLING_LR' 
    LR_filepath = LR_dirpath / 'tfMRI_GAMBLING_LR_Atlas_MSMAll.dtseries.nii'

    # cifti object:
    RL_cii = nib.load(RL_filepath.fspath)
    LR_cii = nib.load(LR_filepath.fspath)

    #split both into R and L hemisphere data 
    (RL_lh, RL_rh, subdat1) = ny.hcp.cifti_split(RL_cii)
    (LR_lh, LR_rh, subdat2) = ny.hcp.cifti_split(LR_cii)

    # concatenate RL and LR data in the time (0th) dimension
    # left hemisphere
    all_left_hemi = np.concatenate((RL_lh, LR_lh), axis=0)
    
    # right hemisphere 
    all_right_hemi = np.concatenate((RL_rh, LR_rh), axis=0)

    #return tuple 
    return (all_left_hemi, all_right_hemi)





    
    