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

TR_LEN = 0.72
VALENCE_LIST = ("win", "loss")
DIRECTION_LIST = ("RL", "LR")

# -------------------------------------------------------------------------------- # 
def split_time_series(roi_time_series, patient_num, hcp_base_path, tr_len=TR_LEN,):
    """
    Split an RL-then-LR concatenated gambling task time series into separate dfs for win and loss time series.

    Parameters
    ----------
    roi_time_series : 
        Array with shape (total_timepoints, n_rois).
        direction: RL, LR.

    tr_len : float (TR = 0.72) 

    Returns
    -------
    win_time_series : np.ndarray
        Win-block timepoints from RL and LR.

    loss_time_series : np.ndarray
        Loss-block timepoints from RL and LR.
    """
    

    # coerce func data time series to array and ensure TR*ROI shape 
    roi_time_series = np.asarray(roi_time_series)
    if roi_time_series.ndim != 2:
        raise ValueError("roi_time_series must have shape (timepoints, ROIs).")

    # check that nTRs is correct and calculate nTRs/run
    n_total_trs = roi_time_series.shape[0]
    if n_total_trs % 2 != 0:
        raise ValueError(
            f"Expected an even number of timepoints, "
            f"but found {n_total_trs}."
        )
    n_trs_per_run = n_total_trs // 2

    # ciis_to_giis() concatenates RL first and LR second.
    # split into RL and LR time series 
    run_time_series = {
        "RL": roi_time_series[:n_trs_per_run, :],
        "LR": roi_time_series[n_trs_per_run:, :],
    }

    # instantiate valence blocks for split of time series by valance
    valence_blocks = {"win": [], "loss": [],}

        
    for direction in DIRECTION_LIST: 
        direction_data = run_time_series[direction] # for given direction

        # get time (s) @ start of TR 
        tr_times = np.arange(n_trs_per_run) * tr_len

        # get block onsets for given valence
        for valence in VALENCE_LIST:

            ev_path = (
                hcp_base_path
                / str(patient_num)
                / "MNINonLinear"
                / "Results"
                / f"tfMRI_GAMBLING_{direction}"
                / "EVs"
                / f"{valence}.txt"
            )

            with ev_path.open("r") as ev_file:
                onset_df = pd.read_csv(
                    ev_file,
                    sep=r"\s+",
                    header=None,
                    names=[
                        "onset",
                        "duration",
                        "amplitude",
                    ],
                )

            for block in onset_df.itertuples(index=False):
                # annotate block start and end 
                block_start = block.onset
                block_end = block.onset + block.duration

                block_mask = (
                    (tr_times >= block_start)
                    & (tr_times < block_end)
                )

                # split data by task epoch (win or loss) 
                block_data = direction_data[block_mask, :]

                if block_data.shape[0] == 0:
                    raise ValueError(
                        f"No TRs selected for participant "
                        f"{patient_num}, direction={direction}, "
                        f"valence={valence}, "
                        f"onset={block_start}, "
                        f"duration={block.duration}."
                    )

                valence_blocks[valence].append(block_data)
                
    # concatenate win and loss blocks into one df /valence & return
    win_time_series = np.vstack(valence_blocks["win"])
    loss_time_series = np.vstack(valence_blocks["loss"])
    return win_time_series, loss_time_series
# -------------------------------------------------------------------------------- # 


def restore_corr_matrices(vectorized_data_with_ids):
    ''' 
    Restore full symmetric correlation matrices from a vectorized upper triangle.
    
    Parameters
    ----------
    vectorized_data_with_ids : list of 1D arrays
        First element of an array = participant ID, remaining elements = upper-triangular
        values of the correlation matrix
    
    Returns
    -------
    participant_id : the ID (first value)
    corr_matrix : the reconstructed (n x n) symmetric matrix with 1s on diagonal
    '''
    
    subject_ids = []
    data = []

    for row in vectorized_data_with_ids:
        subject_ids.append(str(int(row[0])))
        temp = vec_to_sym_matrix(row[1:])
        data.append(temp)

    return subject_ids, corr_matrices




    
    