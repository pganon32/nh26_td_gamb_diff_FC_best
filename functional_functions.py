"""
functions for functional team 
cii to correlation matrix functions
"""

#requires: 
import numpy as np
import pandas as pd
from cloudpathlib import S3Path, S3Client
import nibabel as nib
import neuropythy as ny
import hcp_utils as hcp
from nilearn.surface import InMemoryMesh, PolyMesh
from nilearn.surface import SurfaceImage

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

def create_mmp_mesh():
    ''' Creates a mesh based on the hcp_mmp atlas and 
        splits data for left and right hemishphere and
        returns labels_image 
    ''' 

    # define the mesh for the left and right hemispheres
    left_coords, left_faces = hcp.mesh["pial_left"]
    right_coords, right_faces = hcp.mesh["pial_right"]
    
    mesh = PolyMesh(
        left=InMemoryMesh(left_coords, left_faces),
        right=InMemoryMesh(right_coords, right_faces),
    )

    # Split the mesh 
    parc = hcp.mmp
    right_data = hcp.right_cortex_data(parc.map_all)
    left_data = hcp.left_cortex_data(parc.map_all)
    
    data = {
        "left": left_data,
        "right": right_data,
    }
    
    labels_image = SurfaceImage(mesh=mesh, data=data)

    return labels_image

def create_mmp_lookup():
    ''' Creates and returns a lookup table for the hcp_mmp atlas
    '''
    parc = hcp.mmp
    
    lut = pd.DataFrame(parc.labels.values(), index=parc.labels.keys(), columns=["name"])
    lut.iloc[0] = "Background"
    lut['color'] = parc.rgba.values()
    lut['index'] = lut.index

    return lut





    
    