"""
Autofocus Data Loading and Timestamp Alignment

Functions for loading autofocus meta data data, extracting timestamps, and 
synchronizing time stamps for performance evaluation.


Author: Sara Habte - Imperial College London 
Date: December 2024
"""


import numpy as np
import pandas as pd
import os
from datetime import datetime
from tifffile import TiffFile

 
def time_to_seconds(time_str):
    """
    Convert time string to seconds since midnight.
    
    Handles multiple time formats:
    - '%Y-%m-%d %H:%M:%S.%f %z' (full datetime with timezone)
    - '%H:%M:%S.%f' (time with microseconds)
    - '%H:%M:%S' (time with seconds)
    - '%H:%M' (time with jud minutes)
    
    Parameters
    ----------
    time_str : str
        Time string to convert
    
    Returns
    -------
    float
        Time in secons since midnight
    """
    time_str = time_str.strip()
    import re
    time_str = re.sub(r'\s+([+-]\d{4})$', r'\1', time_str)
    
    formats = [
        '%Y-%m-%d %H:%M:%S.%f%z',   
        '%Y-%m-%d %H:%M:%S.%f %z',  
        '%H:%M:%S.%f',
        '%H:%M:%S',
        '%H:%M'
    ]
    for fmt in formats:
        try:
            t = datetime.strptime(time_str, fmt)
            return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
        except ValueError:
            continue
    raise ValueError(f"Time string '{time_str}' does not match any known format")


def load_zlist_data(zlist_txt):
    """
    Load z-list data from CSV file.
    
    Parameters
    ----------
    zlist_txt : str
        Path to Zlist CSV file
    
    Returns
    -------
    timestamps : np.ndarray
        Array of timestamp strings
    af_correction : np.ndarray
        AF correction values (defocus predictions in nm)
    stage_pos : np.ndarray
        Stage z-positions
    """
    df = pd.read_csv(zlist_txt, header=None)
    
    # Extract and format timestamps
    timestamps = pd.to_datetime(df[1], format='ISO8601').dt.strftime('%H:%M:%S.%f').values[:-1]
    
    # Extract numeric data and clean parentheses
    af_correction = df[2].values[:-1]
    stage_pos = df[3].values[:-1]
    
    # Convert to float, removing any parentheses
    af_correction = np.array([float(str(x).replace('(', '').replace(')', '')) for x in af_correction])
    stage_pos = np.array([float(str(x).replace('(', '').replace(')', '')) for x in stage_pos])
    
    return timestamps, af_correction, stage_pos


def load_af_metrics_data(af_metrics_txt):
    """
    Load autofocus metrics from CSV file.
    
    Parameters
    ----------
    af_metrics_txt : str
        Path to AF metrics CSV file
    
    Returns
    -------
    timestamps : np.ndarray
        Array of timestamp strings
    af_metric0 : np.ndarray
        AF metric in x-direction
    af_metric1 : np.ndarray
        AF metric in y-direction
    af_metric_avg : np.ndarray
        Average AF metri
    """
    df = pd.read_csv(af_metrics_txt, header=None)
    
    # Extract timestamps
    timestamps = pd.to_datetime(df[3], format='mixed').dt.strftime('%H:%M:%S.%f').values[:-1]
    
    # Extract metrics
    af_metric0 = df[0].values[:-1]
    af_metric1 = df[1].values[:-1]
    af_metric_avg = df[2].values[:-1]
    
    # Convert to float, removing any parentheses
    af_metric0 = np.array([float(str(x).replace('(', '').replace(')', '')) for x in af_metric0])
    af_metric1 = np.array([float(str(x).replace('(', '').replace(')', '')) for x in af_metric1])
    af_metric_avg = np.array([float(str(x).replace('(', '').replace(')', '')) for x in af_metric_avg])
    
    return timestamps, af_metric0, af_metric1, af_metric_avg

def extract_timelapse_timestamps(tiff_paths):
    """
    Extract timestamps from time-lapse TIFF file(s) metadata.
    Parameters
    ----------
    tiff_paths : str or list of str
        Path(s) to time-lapse TIFF file(s)
    Returns
    -------
    timestamps : list
        List of timestamp strings from TIFF metadata
    """
    timestamps = []
    for tiff_path in tiff_paths:
        if isinstance(tiff_path, str):
            tiff_path = [tiff_path]
        
        for file_idx, tiff_file in enumerate(tiff_path):
            print(f"Reading TIFF {file_idx+1}/{len(tiff_path)}: {os.path.basename(tiff_file)}")
            try:
                with TiffFile(tiff_file) as tif:
                    for page in tif.pages:

                        try:
                            metadata = page.tags['MicroManagerMetadata'].value
                            
                            timestamps.append(metadata['ReceivedTime'])
                        except KeyError:
                            print(f"  Warning: No metadata for frame in {os.path.basename(tiff_file)}")
            except Exception as e:
                print(f"  Error processing metadata: {e}")
            except Exception as e:
                print(f"  Error opening file: {e}")
            continue
        if not timestamps:
            raise ValueError("No timestamps found in TIFF file(s)")
        print(f"✓ Extracted {len(timestamps)} timestamps, range: {timestamps[0]} → {timestamps[-1]}")
    return timestamps

def match_time_windows(timestamps_reference, timestamps_target, data_arrays):
    """
    Match target timestamps to reference time window and extract corresponding data.
    
    Parameters
    ----------
    timestamps_reference : np.ndarray
        Reference timestamps (in seconds) defining the time window
    timestamps_target : np.ndarray
        Target timestamps (in seconds) to be matched
    data_arrays : list of np.ndarray
        Data arrays to extract using matched indices
    
    Returns
    -------
    matched_timestamps : np.ndarray
        Target timestamps within reference window
    matched_data : list of np.ndarray
        Extracted data arrays corresponding to matched timestamps
    """
    start_time = timestamps_reference[0]
    end_time = timestamps_reference[-1]
    
    # Find matching indices
    idx_start = np.argmin(np.abs(timestamps_target - start_time))
    idx_end = np.argmin(np.abs(timestamps_target - end_time)) + 1
    
    # Extract matched data
    matched_timestamps = timestamps_target[idx_start:idx_end]
    matched_data = [arr[idx_start:idx_end] for arr in data_arrays]
    
    return matched_timestamps, matched_data


def convert_to_relative_minutes(timestamps_seconds, reference_time):
    """
    Convert timestamps to minutes relative to a reference time.
    
    Parameters
    ----------
    timestamps_seconds : np.ndarray
        Timestamps in seconds since midnight
    reference_time : float
        Reference time in seconds (typically the start of acquisition)
    
    Returns
    -------
    np.ndarray
        Timestamps in minutes relative to reference time
    """
    return (timestamps_seconds - reference_time) / 60.0


def OAF_Pred(PSF_timelapse, Zlist_txt, AF_metrics_txt,npy_folder=None):
    """
     time-align autofocus data for performance evaluation.
    
    This function does:
    1. Loads z-list and AF metrics from CSV files
    2. Extracts timestamps from time-lapse TIFF files
    3. Matches all timelines to a common time window
    4. Converts timestamps to relative minutes for plotting
    
    Parameters
    ----------
    time_lapse_LED_paths : str or list of str
        Path(s) to time-lapse TIFF file(s)
    Zlist_txt : str
        Path to Zlist CSV file (timestamps, AF corrections, stage positions)
    AF_metrics_txt : str
        Path to AF metrics CSV file (timestamps, x/y/avg metrics)
    
    Returns
    -------
    af_correction : np.ndarray
        AF correction values (nm) over matched time window
    stage_pos : np.ndarray
        Stage z-positions over matched time window
    timestamps_zlist_minutes : np.ndarray
        Zlist timestamps in relative minutes
    af_metric_avg_matched : np.ndarray
        Average AF metrics over matched time window
    timestamps_af_metrics_minutes : np.ndarray
        AF metrics timestamps in relative minutes
    timestamps_timelapse_minutes : np.ndarray
        Time-lapse frame timestamps in relative minutes
    """
    print("\n" + "="*70)
    print("PROCESSING AUTOFOCUS DATA")
    print("="*70 + "\n")
    
    # ========================================
    # 1. Load data files
    # ========================================
    timestamps_zlist, af_correction, stage_pos = load_zlist_data(Zlist_txt)
    timestamps_af_metrics, af_metric0, af_metric1, af_metric_avg = load_af_metrics_data(AF_metrics_txt)
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        timestamps_timelapse = extract_timelapse_timestamps(PSF_timelapse)
    
    # timestamps_timelapse = extract_timelapse_timestamps(PSF_timelapse)
    
    # ========================================
    # 2. Convert all timestamps to seconds
    # ========================================
    timestamps_zlist_sec = np.array([time_to_seconds(t) for t in timestamps_zlist])
    timestamps_af_metrics_sec = np.array([time_to_seconds(t) for t in timestamps_af_metrics])
    timestamps_timelapse_sec = np.array([time_to_seconds(t) for t in timestamps_timelapse])
    
    print(f"timestamps_timelapse_sec shape: {timestamps_timelapse_sec[0].shape}")

    # ========================================
    # 3. Match time-lapse to Zlist window
    # ========================================
    print("\nMatching time windows...")
    timestamps_tl_matched_sec, _ = match_time_windows(
        timestamps_zlist_sec,
        timestamps_timelapse_sec,
        []
    )
    
    timestamps_tl_matched_sec = np.array(timestamps_tl_matched_sec).flatten()
    
    print(f"timestamps_tl_matched_sec shape: {np.array(timestamps_tl_matched_sec).shape}")
    print(f"  Time-lapse matched: {len(timestamps_tl_matched_sec)} frames")
    
    # ========================================
    # 4. Match AF metrics to Zlist window
    # ========================================
    timestamps_af_matched_sec, [af_metric_avg_matched] = match_time_windows(
        timestamps_zlist_sec,
        timestamps_af_metrics_sec,
        [af_metric_avg]
    )
    
    print(f"  AF metrics matched: {len(timestamps_af_matched_sec)} points")
    
    # ========================================
    # 5. Convert to relative minutes
    # ========================================
    reference_time = timestamps_tl_matched_sec[0]  # Use TL start as reference
    
    timestamps_zlist_minutes = convert_to_relative_minutes(timestamps_zlist_sec, reference_time)
    timestamps_af_metrics_minutes = convert_to_relative_minutes(timestamps_af_matched_sec, reference_time)
    timestamps_timelapse_minutes = convert_to_relative_minutes(timestamps_tl_matched_sec, reference_time)
    
    print(f"\n✓ Time alignment complete")
    print(f"  Zlist: {timestamps_zlist_minutes[0]:.2f} → {timestamps_zlist_minutes[-1]:.2f} min")
    print(f"  AF metrics: {timestamps_af_metrics_minutes[0]:.2f} → {timestamps_af_metrics_minutes[-1]:.2f} min")
    print(f"  Time-lapse: {timestamps_timelapse_minutes[0]:.2f} → {timestamps_timelapse_minutes[-1]:.2f} min")
    print("="*70 + "\n")

    if npy_folder:
        os.makedirs(npy_folder,exist_ok=True)
        np.save(os.path.join(npy_folder,'af_correction_matched_minutes.npy'),af_correction)
        np.save(os.path.join(npy_folder,'stage_pos_matched_minutes.npy'),stage_pos)
        np.save(os.path.join(npy_folder,'timestamps_zlist_matched_minutes.npy'),timestamps_zlist_minutes)
        np.save(os.path.join(npy_folder,'af_metric_avg_matched_minutes.npy'),af_metric_avg_matched)
        np.save(os.path.join(npy_folder,'timestamps_af_metrics_matched_minutes.npy'),timestamps_af_metrics_minutes)
        np.save(os.path.join(npy_folder,'timestamps_TL_matched_minutes.npy'),timestamps_timelapse_minutes)
        

    

    return (af_correction, stage_pos, timestamps_zlist_minutes,
            af_metric_avg_matched, timestamps_af_metrics_minutes,
            timestamps_timelapse_minutes)








    
 
# if __name__ == "__main__":
#     # Load and process autofocus data
#     af_correction, stage_pos, timestamps_zlist_minutes, \
#     af_metric_avg_matched, timestamps_af_metrics_minutes, \
#     timestamps_timelapse_minutes = OAF_Pred(
#         time_lapse_LED_paths='path/to/timelapse.tif',
#         Zlist_txt='path/to/Zlist.csv',
#         AF_metrics_txt='path/to/AF_metrics.csv'
#     )
    
#     # Now ready for plottin
#     # AstigAF4plots(DisabledZ_time, black_vline, cropped_lasttime, pred_defocus_LURS,
#     #               timestamps_timelapse_minutes, af_correction, stage_pos, 
#     #               timestamps_zlist_minutes, af_metric_avg_matched, 
#     #               timestamps_af_metrics_minutes, save_folder, title=AF_title)