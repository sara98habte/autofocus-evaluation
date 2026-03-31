"""
Astigmatic 2D Autocorrelation Analysis


Author: Sara Habte - Imperial College London 
Date: December 2024
"""

from scipy import signal, ndimage, interpolate, fft 
from scipy.optimize import curve_fit
import numpy as np
import os
import numpy as np
from scipy.optimize import curve_fit


def stdACFProj(new_image, center, shift):
 
   
    acf=np.fft.fftshift(np.fft.irfft2(np.abs(np.fft.rfft2(new_image)) ** 2))
    center__=center
    shift=shift
    row_crop,col_crop=center__-shift,center__+1+shift
    acf=acf[row_crop:col_crop,row_crop:col_crop]
    ACF_FOV=np.shape(acf)
   
    y_max, x_max = np.unravel_index(np.argmax(acf), acf.shape)
    

        # Mask the spike
    acf_masked = acf.copy()
    acf_masked[y_max, x_max] = np.nan

    # # Fit a spline to a small region around the peak, excluding the centre
    r = 10  # use a slightly larger region for a stable fit
    y_slice = slice(max(0, y_max - r), min(acf.shape[0], y_max + r + 1))
    x_slice = slice(max(0, x_max - r), min(acf.shape[1], x_max + r + 1))

    region = acf_masked[y_slice, x_slice]
    ys = np.arange(region.shape[0])
    xs = np.arange(region.shape[1])

    # Use only non-nan points
    valid = ~np.isnan(region)
    from scipy.interpolate import griddata
    points = np.array([(y, x) for y in ys for x in xs if valid[y, x]])
    values = np.array([region[y, x] for y, x in points])

    # Interpolate at the centre point
    centre = np.array([[r, r]])  # centre of the slice
    interpolated = griddata(points, values, centre, method='cubic')
    acf[y_max, x_max] = interpolated[0]

    # acf[cy, :] = np.nan  # zero-frequency row (y=0)
    # acf[:, cx] = np.nan  # zero-frequency column (x=0)
    # acf = np.delete(acf, cy, axis=0)  # remove centre row
    # acf = np.delete(acf, cx, axis=1)  # remove centre column

    acf_meanx=np.mean(acf,axis=0)
    acf_meany=np.mean(acf,axis=1)


    std_acf0=np.std(acf_meanx)
    std_acf1=np.std(acf_meany)
    return acf, acf_meanx, acf_meany, std_acf0, std_acf1
def sigmaGaussACFProj(acf_meanx, acf_meany, std_acf0, std_acf1):
    error = False
    xy_acfsize = np.arange(len(acf_meanx))

    # Remove NaN positions before fitting
    mask_x = ~np.isnan(acf_meanx)
    mask_y = ~np.isnan(acf_meany)
    x_clean = xy_acfsize[mask_x]
    y_clean = xy_acfsize[mask_y]
    meanx_clean = acf_meanx[mask_x]
    meany_clean = acf_meany[mask_y]

    def gaussian_1d(x, A, mu, sigma,offset):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) +offset

    pedestal_x = np.nanmean(acf_meanx[-10:])  # mean of last 10 points
    pedestal_y = np.nanmean(acf_meany[-10:])

    acf_meanx_sub = acf_meanx - pedestal_x
    acf_meany_sub = acf_meany - pedestal_y

    try:
        p0 = [
            np.nanmax(acf_meanx),
            x_clean[np.argmax(meanx_clean)],
            (x_clean[-1] - x_clean[0]) / 10,
            np.nanmin(acf_meanx)
        ]
        p1= [
            np.nanmax(acf_meany),
            y_clean[np.argmax(meany_clean)],
            (y_clean[-1] - y_clean[0]) / 10,
            np.nanmin(acf_meany)
        ]

        popt_0, _ = curve_fit(gaussian_1d, x_clean, meanx_clean, p0=p0, maxfev=2000)
        popt_1, _ = curve_fit(gaussian_1d, y_clean, meany_clean, p0=p1, maxfev=2000)

        x_fit = np.linspace(min(x_clean), max(x_clean), 100)
        y_fit =np.linspace(min(y_clean), max(y_clean), 100)
        y_fit_0 = gaussian_1d(x_fit, *popt_0)
        y_fit_1 = gaussian_1d(y_fit, *popt_1)

        Gsigma0 = popt_0[2]
        Gsigma1 = popt_1[2]
        Gstd0 = np.std(y_fit_0)
        Gstd1 = np.std(y_fit_1)

    except RuntimeError as e:
        error = True
        Gsigma0, Gsigma1, Gstd0, Gstd1 = 0, 0, 0, 0
        print(f"Error in curve fitting: {e}")

  
    return Gsigma0, Gsigma1, Gstd0, Gstd1, error,y_fit_0,y_fit_1



def ACF_Metrics(images, center_,save_folder=None):
    """
    Analyse complete frames to extract ACF metrics at each z-position.
    
    Parameters
    ----------
    images : np.ndarray
        3D image stack (z, y, x)
    center_ : int
        Center coordinate for ACF cropping
    
    Returns
    -------
    acf_stack : list
        List of 2D ACF arrays for each z-slice
    std_acf0Z : list
        Standard deviations of x-projections
    std_acf1Z : list
        Standard deviations of y-projections
    Gsigma0Z : list
        Gaussian widths in x-direction
    Gsigma1Z : list
        Gaussian widths in y-direction
    RGsigma0 : list
        Ratios sigma_x / sigma_y
    RGsigma1 : list
        Ratios sigma_y / sigma_x
    """
    # Initialize storage
    acf_stack = []
    std_acf0Z, std_acf1Z = [], []
    Gsigma0Z, Gsigma1Z = [], []
    Gstd0Z, Gstd1Z = [], []
    RGsigma0, RGsigma1 = [], []
    RmaxZ = []
    GfitX,GfitY=[],[]

    for slice_index in range(len(images)):
        image_i = images[slice_index]
        

        acf, acf_meanx, acf_meany, std_acf0, std_acf1 = stdACFProj(
            image_i, center=center_, shift=20
        )
        # if slice_index==285:
        #     plt.imshow(acf,cmap='gray')
        acf_stack.append(acf)

        Gsigma0, Gsigma1, Gstd0, Gstd1, error,y_fit_0,y_fit_1 = sigmaGaussACFProj(
            acf_meanx, acf_meany, std_acf0, std_acf1
        )
        
        if error:
            print(f"Error in sigmaGaussACFProj for slice {slice_index}")
            break
        

        Gsigma0Z.append(Gsigma0)
        Gsigma1Z.append(Gsigma1)

        GfitX.append(y_fit_0)
        GfitY.append(y_fit_1)        


        RGsigma0.append(Gsigma0 / Gsigma1)
        RGsigma1.append(Gsigma1 / Gsigma0)
    if save_folder:

        os.makedirs(save_folder,exist_ok=True)
        np.save(os.path.join(save_folder,'acf_stack.npy'),acf_stack)
        np.save(os.path.join(save_folder,'GfitX.npy'),GfitX)
        np.save(os.path.join(save_folder,'GfitY.npy'),GfitY)

    
    return  Gsigma0Z, Gsigma1Z, RGsigma0, RGsigma1


def DefocusModel_AstigACF(ref_z_range, Gsigma0Z, Gsigma1Z):
    """
    Build polynomial defocus model and high-resolution interpolation.
    
    This function:
    1. Fits 6th-degree polynomials to sigma_x(z) and sigma_y(z)
    2. Finds focal plane (where sigma_x = sigma_y)
    3. Adjusts z-positions relative to focal plane
    4. Creates high-resolution interpolated calibration curves
    
    Parameters
    ----------
    ref_z_range : np.ndarray
        Array of z-positions (microns)
    Gsigma0Z : np.ndarray or list
        Measured PSF widths in x-direction
    Gsigma1Z : np.ndarray or list
        Measured PSF widths in y-direction
    
    Returns
    -------
    adjusted_z_positions : np.ndarray
        Z-positions adjusted relative to focal plane
    Gsigma0Z_fit : np.ndarray
        Polynomial fit of sigma_x at adjusted z-positions
    Gsigma1Z_fit : np.ndarray
        Polynomial fit of sigma_y at adjusted z-positions
    Gsigma0Z_fit_interp : np.ndarray
        High-resolution interpolation of sigma_x
    Gsigma1Z_fit_interp : np.ndarray
        High-resolution interpolation of sigma_y
    z_new : np.ndarray
        High-resolution z-positions for interpolation
    """
    
    def find_intersections(x, y1, y2):
        """Find intersection points between two curves."""
        diff = y1 - y2
        sign_change_indices = np.where(np.diff(np.signbit(diff)))[0]
        
        intersections = []
        for i in sign_change_indices:
            # Linear interpolation at sign change
            x_interp = x[i] + (x[i+1] - x[i]) * (0 - diff[i]) / (diff[i+1] - diff[i])
            y_interp = np.interp(x_interp, x, y1)
            intersections.append((x_interp, y_interp))
        
        return intersections
    
    # Fit 6th-degree polynomials
    cx = np.polyfit(ref_z_range, Gsigma0Z, 6, full=False)
    cy = np.polyfit(ref_z_range, Gsigma1Z, 6, full=False)
    
    Gsigma0Z_fit = np.polyval(cx, ref_z_range)
    Gsigma1Z_fit = np.polyval(cy, ref_z_range)
    
    # Find focal plane (intersection of sigma_x and sigma_y)
    intersections = find_intersections(
        ref_z_range, 
        np.polyval(cx, ref_z_range), 
        np.polyval(cy, ref_z_range)
    )
    
    # Adjust z-positions relative to focal plane
    focal_plane_z = intersections[0][0]
    adjusted_z_positions = ref_z_range - focal_plane_z
    
    print(f"Focal plane offset: {focal_plane_z:.4f} μm")
    print(f"Adjusted z-range: [{adjusted_z_positions[0]:.3f}, {adjusted_z_positions[-1]:.3f}] μm")
    
    # Create high-resolution interpolation
    z_new = np.linspace(adjusted_z_positions[0], adjusted_z_positions[-1], num=60100)
    
    Gsigma0Z_fit_cubic = interpolate.interp1d(
        adjusted_z_positions, Gsigma0Z_fit, 
        kind='cubic', fill_value="extrapolate"
    )
    Gsigma1Z_fit_cubic = interpolate.interp1d(
        adjusted_z_positions, Gsigma1Z_fit, 
        kind='cubic', fill_value="extrapolate"
    )
    
    Gsigma0Z_fit_interp = Gsigma0Z_fit_cubic(z_new)
    Gsigma1Z_fit_interp = Gsigma1Z_fit_cubic(z_new)
    
    # Diagnostic output
    mid_idx = len(z_new) // 2
    RGsigma0_mid = Gsigma0Z_fit_interp[mid_idx] / Gsigma1Z_fit_interp[mid_idx]
    print(f"Sigma ratio at z=0: {RGsigma0_mid:.4f}")
    print(f"Interpolation: {len(z_new)} points over {len(adjusted_z_positions)} measurements")
    
    return (adjusted_z_positions, Gsigma0Z_fit, Gsigma1Z_fit, 
            Gsigma0Z_fit_interp, Gsigma1Z_fit_interp, z_new)



def find_nearest(array, value):
    return (np.abs(array - value)).argmin()

def Astig_ACF_pred(RGsigma0_TL, PSF_TL, Gsigma0Z_fit_interp, Gsigma1Z_fit_interp, 
                   z_new, LU_range, timestamps_TL_matched_minutes, npy_folder=None):
    
    # LU_range = np.array(LU_range).flatten()  # ← uncomment this
    print(len(LU_range), LU_range)
    
    startz, stopz = LU_range[0], LU_range[1]
    startzidx = find_nearest(z_new, startz)
    stopzidx = find_nearest(z_new, stopz)
    
    RGsigma0_fit_interp =   Gsigma0Z_fit_interp/Gsigma1Z_fit_interp
    RGsigma1_fit_interp =   Gsigma1Z_fit_interp/Gsigma0Z_fit_interp
    RGsigma1_LU = np.asarray(RGsigma0_fit_interp)[startzidx:stopzidx+1]
    z_new_lim = np.asarray(z_new)[startzidx:stopzidx+1]
    
    expected_frames_TL = len(PSF_TL)
    timestamps_TL_matched_minutes = np.array(timestamps_TL_matched_minutes).flatten()
    expected_frames_timematched = len(timestamps_TL_matched_minutes)

    pred_defocus_ = np.zeros(expected_frames_TL)
    pred_error_ = np.zeros(expected_frames_TL)
    expected_frames = min(expected_frames_TL, expected_frames_timematched)
    

    for TL_idx in range(expected_frames):
        min_idx = find_nearest(RGsigma1_LU, RGsigma0_TL[TL_idx])
        pred_defocus_[TL_idx] = z_new_lim[min_idx]
        pred_error_[TL_idx] = RGsigma0_TL[TL_idx] - RGsigma1_LU[min_idx]
    
    if npy_folder:
        os.makedirs(npy_folder, exist_ok=True)
        np.save(os.path.join(npy_folder, 'astig_defocus_matched_minutes.npy'), pred_defocus_)
    




 