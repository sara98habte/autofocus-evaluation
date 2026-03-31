"""
Read PSF data 

Author: Sara Habte - Imperial College London 
Date: December 2024
"""
import numpy as np
from skimage import io as imio

def read_zstack(zstack, ref_z_range, sym_crop):
    
    z_calib = imio.imread(zstack)[:, :, :]
    
    if sym_crop is None:
        sym_crop = 0   
    
    start_idx = sym_crop // 2
    end_idx = z_calib.shape[0] - sym_crop // 2 
    z_calib = z_calib[start_idx:end_idx, :, :]
    
    center_ = z_calib.shape[1] // 2
    num_frames = z_calib.shape[0]  
    ref_z_positions, step = np.linspace(-(ref_z_range/2), (ref_z_range/2), 
                                         num=num_frames, retstep=True)
    
    return z_calib, center_, ref_z_positions, step
def read_AstigTL(PSF_timelapse,binby2=False):

    sourcepath_timelapse=PSF_timelapse
    if binby2:
        PSF_TL=imio.imread(sourcepath_timelapse)[:,::2,::2]
    else:   
        PSF_TL=imio.imread(sourcepath_timelapse)[:,:,:]
    return PSF_TL


