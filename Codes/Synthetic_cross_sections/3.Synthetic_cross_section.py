import os
import sys
import glob
import pickle
import numpy as np
import math
from geopy.distance import geodesic, distance
from geopy import Point
from matplotlib import pyplot as plt
import argparse



def fresnel_half_width(r, vel, T=11):
    """
    Calculate Fresnel half-width for a given depth, velocity, and period.
    Parameters:
        r (float or np.ndarray): Depth (km)
        vel (float): Shear wave velocity (km/s)
        T (float): Period (s), default 11
    Returns:
        np.ndarray: Fresnel half-width (km)
    """
    return np.sqrt(np.maximum((((T * vel) / 2) + r) ** 2 - r ** 2, 0))

def weight(distance, fresnel, factor):
    """
    Compute weighting based on distance and Fresnel zone.
    Parameters:
        distance (float or np.ndarray): Distance from grid point (km)
        fresnel (float or np.ndarray): Fresnel half-width (km)
        factor (float): Smoothing factor
    Returns:
        np.ndarray: Weight values
    """
    delta_bar = distance / (fresnel * factor)

    # np.clip avoids overflow for large delta_bar
    exp_arg = np.clip(8 * (delta_bar - 1), -700, 700)

    return 1 / (1 + np.exp(exp_arg))


def load_grid(grid_file, depth_max=125.0, depth_step=0.1):
    """
    Load cross-section grid points and create depth and distance arrays.
    Parameters:
        grid_file (str): Path to grid file (lon, lat columns)
        depth_max (float): Maximum depth (km)
        depth_step (float): Depth step (km)
    Returns:
        tuple: (grid_lon, grid_lat, grid_depth, grid_dist)
    """
    grid_data = np.loadtxt(grid_file)
    grid_lon, grid_lat = grid_data[:, 0], grid_data[:, 1]
    grid_depth = np.arange(0, depth_max, depth_step)
    grid_dist = np.arange(len(grid_lat)) * 1    # Assuming 1 km spacing between grid points (as used in script 1b)

    return grid_lon, grid_lat, grid_depth, grid_dist

def process_rf_files_ps(RF_files, grid_lon, grid_lat, grid_depth, stacking_smooth_fresnel, gauss_direct):
    """
    Process all receiver function (RF) files and stack amplitudes onto the cross-section grid.
    Args:
        RF_files (list): List of RF .pkl file paths
        grid_lon, grid_lat, grid_depth, grid_dist: Grid arrays
        stacking_smooth_fresnel (float): Fresnel smoothing factor
        gauss_direct (float): Gaussian width
    Returns:
        tuple: (vol, vol_weight, vol_num, SD_vol_running, Mn_running, Sn_running, Wn_running) stacked arrays
    """
    import pandas as pd
    stations = pd.read_csv('/raid2/cg812/Stations_to_use.csv', header=None)
    print('hello')
    
    shape = (len(grid_lon), len(grid_depth), 4)
    vol = np.zeros(shape)
    vol_weight = np.zeros(shape)
    vol_num = np.zeros(shape)
    SD_vol_running = np.zeros((len(grid_lon), len(grid_depth), 4))
    Mn_running = np.zeros((len(grid_lon), len(grid_depth), 2, 4))
    Sn_running = np.zeros((len(grid_lon), len(grid_depth), 2, 4))
    Wn_running = np.zeros((len(grid_lon), len(grid_depth), 2, 4))
      
    for count, rf_file in enumerate(RF_files, start=1):
        print(f"Processing Ps phase in RF {count}/{len(RF_files)}: {rf_file}")

        trace_list= glob.glob('/raid2/cg812/All_together/Gauss_6.0/*[!.png]')
        print(len(trace_list))
        station_current= trace_list[count-1]


        import obspy

        
        try:
            current_trace= obspy.read(station_current)
        except:
            print('theres gonna be an extra one here')
        BAZ= current_trace[0].stats.baz

        stat_lat = current_trace[0].stats.stla
        stat_lon = current_trace[0].stats.stlo
        sta_id = current_trace[0].stats.station
        print(stat_lon)

        
        with open(rf_file, 'rb') as pick_file:
            data = pickle.load(pick_file)

        try:
            filter = gauss_direct
            stats = data[0].stats['conversions'][filter]
            RF_depths, RF_amps_Ps, RF_H_Ps = stats['depth_Ps'], stats['amp_Ps'], stats['Hdist_Ps']
        
        except KeyError:
            print("No conversions found")
            continue

    
        grid_points = np.column_stack((grid_lat, grid_lon))
        stat_point = (stat_lat, stat_lon)
        dists = np.array([geodesic(stat_point, (lat, lon)).kilometers for lat, lon in grid_points])
        valid_idx = np.where(dists <= 100)[0]
        
        for la in valid_idx:
            current_grid_lat = grid_lat[la]
            current_grid_lon = grid_lon[la]

            pierce_points = [distance(kilometers=RF_H_Ps[d]).destination(Point(stat_lat, stat_lon), BAZ) for d in range(len(grid_depth))]
            pierce_lats = np.array([p.latitude for p in pierce_points])
            pierce_lons = np.array([p.longitude for p in pierce_points])
            distPs = np.array([distance((pierce_lats[d], pierce_lons[d]), (current_grid_lat, current_grid_lon)).km for d in range(len(grid_depth))])
            
            Vs = 6.4 / 1.78
            FZHW_Vs = fresnel_half_width(grid_depth, Vs)
            weight_Ps = np.where(distPs < (FZHW_Vs * stacking_smooth_fresnel), weight(distPs, FZHW_Vs, stacking_smooth_fresnel), 0)
            
            for d in range(len(grid_depth)):
                current_RF_amp_Ps = RF_amps_Ps[d]
                w = weight_Ps[d]
                vol[la, d, 0] += w * current_RF_amp_Ps
                vol_weight[la, d, 0] += w
                if w > 0:
                    vol_num[la, d, 0] += 1
                    current_x = w * current_RF_amp_Ps
                    if vol_num[la, d, 0] == 1:
                        Mn_running[la, d, 1, 0] = current_x
                        Sn_running[la, d, 1, 0] = 0
                        Wn_running[la, d, 1, 0] = w
                    elif vol_num[la, d, 0] > 1:
                        Wn_running[la, d, 0, 0] = Wn_running[la, d, 1, 0]
                        Wn_running[la, d, 1, 0] += w
                        Mn_running[la, d, 0, 0] = Mn_running[la, d, 1, 0]
                        Mn_running[la, d, 1, 0] += (w / Wn_running[la, d, 1, 0]) * (current_x - Mn_running[la, d, 0, 0])
                        Sn_running[la, d, 0, 0] = Sn_running[la, d, 1, 0]
                        Sn_running[la, d, 1, 0] += w * ((current_x - Mn_running[la, d, 0, 0]) * (current_x - Mn_running[la, d, 1, 0]))
                        if Wn_running[la, d, 0, 0] * vol_num[la, d, 0] * (vol_num[la, d, 0] - 1) > 0:
                            SD_vol_running[la, d, 0] = (
                                vol_num[la, d, 0] * Sn_running[la, d, 1, 0] /
                                (Wn_running[la, d, 0, 0] * vol_num[la, d, 0] * (vol_num[la, d, 0] - 1))
                            ) ** 0.5
        
            

    return vol, vol_weight, vol_num, SD_vol_running, Mn_running, Sn_running, Wn_running


def process_rf_files_multiples(RF_files, grid_lon, grid_lat, grid_depth, stacking_smooth_fresnel, gauss_mult, vol, vol_weight, vol_num, SD_vol_running, Mn_running, Sn_running, Wn_running):
    """
    Process all multiples in receiver function (RF) files and stack amplitudes onto the cross-section grid.
    Parameters:
        RF_files (list): List of RF .pkl file paths
        grid_lon, grid_lat, grid_depth, grid_dist: Grid arrays
        stacking_smooth_fresnel (float): Fresnel smoothing factor
        gauss_mult (float): Gaussian width
        vol, vol_weight, vol_num, SD_vol_running, Mn_running, Sn_running, Wn_running: output arrays from process_rf_files_ps
    Returns:
        tuple: (vol, vol_weight, vol_num, SD_vol_running) stacked arrays
    """
    import pandas as pd
    stations = pd.read_csv('/raid2/cg812/Stations_to_use.csv', header=None)

    for count, rf_file in enumerate(RF_files, start=1):
        print(f"Processing Multiples in RF {count}/{len(RF_files)}: {rf_file}")
        with open(rf_file, 'rb') as pick_file:
            data = pickle.load(pick_file)
        try:
            filter = gauss_mult
            stats = data[0].stats['conversions'][filter]
            RF_depths, RF_amps_PPs, RF_H_PPs = stats['depth_PPs'], stats['amp_PPs'], stats['Hdist_PPs']
            RF_amps_PSs, RF_H_PSs = stats['amp_PSs'], stats['Hdist_PSs']
        except KeyError:
            print("No conversions found")
            continue



        trace_list= glob.glob('/raid2/cg812/All_together/Gauss_2.0/*[!.png]')
        print(len(trace_list))
        station_current= trace_list[count-1]
        import obspy

        
        try:
            current_trace= obspy.read(station_current)
        except:
            print('theres gonna be an extra one here')
        BAZ= current_trace[0].stats.baz

        stat_lat = current_trace[0].stats.stla
        stat_lon = current_trace[0].stats.stlo
        sta_id = current_trace[0].stats.station

    

        grid_points = np.column_stack((grid_lat, grid_lon))
        stat_point = (stat_lat, stat_lon)
        dists = np.array([geodesic(stat_point, (lat, lon)).kilometers for lat, lon in grid_points])
        valid_idx = np.where(dists <= 100)[0]



        for la in valid_idx:
            current_grid_lat = grid_lat[la]
            current_grid_lon = grid_lon[la]

            # For the PPs Phase:
            n_pps = min(len(grid_depth), len(RF_H_PPs), len(RF_amps_PPs))
            pierce_points_PPs = [distance(kilometers=RF_H_PPs[d]).destination(Point(stat_lat, stat_lon), BAZ) for d in range(n_pps)]
            pierce_lats = np.array([p.latitude for p in pierce_points_PPs])
            pierce_lons = np.array([p.longitude for p in pierce_points_PPs])
            distPPs = np.array([distance((pierce_lats[d], pierce_lons[d]), (current_grid_lat, current_grid_lon)).km for d in range(n_pps)])

            Vs = 6.4 / 1.78
            FZHW_Vs = fresnel_half_width(grid_depth[:n_pps], Vs)
            weight_PPs = np.where(distPPs < (FZHW_Vs * stacking_smooth_fresnel), weight(distPPs, FZHW_Vs, stacking_smooth_fresnel), 0)

            for d in range(n_pps):
                current_RF_amp_PPs = RF_amps_PPs[d]

                w = weight_PPs[d]
                vol[la, d, 1] += w * current_RF_amp_PPs
                vol_weight[la, d, 1] += w
                if w > 0:
                    vol_num[la, d, 1] += 1
                    current_x = w * current_RF_amp_PPs
                    if vol_num[la, d, 1] == 1:
                        Mn_running[la, d, 1, 1] = current_x
                        Sn_running[la, d, 1, 1] = 0
                        Wn_running[la, d, 1, 1] = w
                    elif vol_num[la, d, 1] > 1:
                        Wn_running[la, d, 0, 1] = Wn_running[la, d, 1, 1]
                        Wn_running[la, d, 1, 1] += w
                        Mn_running[la, d, 0, 1] = Mn_running[la, d, 1, 1]
                        Mn_running[la, d, 1, 1] += (w / Wn_running[la, d, 1, 1]) * (current_x - Mn_running[la, d, 0, 1])
                        Sn_running[la, d, 0, 1] = Sn_running[la, d, 1, 1]
                        Sn_running[la, d, 1, 1] += w * ((current_x - Mn_running[la, d, 0, 1]) * (current_x - Mn_running[la, d, 1, 1]))
                        if Wn_running[la, d, 0, 1] * vol_num[la, d, 1] * (vol_num[la, d, 1] - 1) > 0:
                            SD_vol_running[la, d, 1] = (
                                vol_num[la, d, 1] * Sn_running[la, d, 1, 1] /
                                (Wn_running[la, d, 0, 1] * vol_num[la, d, 1] * (vol_num[la, d, 1] - 1))
                            ) ** 0.5

            # For the PSs Phase:
            n_pss = min(len(grid_depth), len(RF_H_PSs), len(RF_amps_PSs))
            

            pierce_points_PSs = [distance(kilometers=RF_H_PSs[d]).destination(Point(stat_lat, stat_lon), BAZ) for d in range(n_pss)]
            pierce_lats = np.array([p.latitude for p in pierce_points_PSs])
            pierce_lons = np.array([p.longitude for p in pierce_points_PSs])
            distPSs = np.array([distance((pierce_lats[d], pierce_lons[d]), (current_grid_lat, current_grid_lon)).km for d in range(n_pss)])

            Vs = 6.4 / 1.78
            FZHW_Vs = fresnel_half_width(grid_depth[:n_pss], Vs)
            weight_PSs = np.where(distPSs < (FZHW_Vs * stacking_smooth_fresnel), weight(distPSs, FZHW_Vs, stacking_smooth_fresnel), 0)

            for d in range(n_pss):
                current_RF_amp_PSs = RF_amps_PSs[d]

                w = weight_PSs[d]
                vol[la, d, 2] += w * current_RF_amp_PSs
                vol_weight[la, d, 2] += w
                if w > 0:
                    vol_num[la, d, 2] += 1
                    current_x = w * current_RF_amp_PSs
                    if vol_num[la, d, 2] == 1:
                        Mn_running[la, d, 1, 2] = current_x
                        Sn_running[la, d, 1, 2] = 0
                        Wn_running[la, d, 1, 2] = w
                    elif vol_num[la, d, 2] > 1:
                        Wn_running[la, d, 0, 2] = Wn_running[la, d, 1, 2]
                        Wn_running[la, d, 1, 2] += w
                        Mn_running[la, d, 0, 2] = Mn_running[la, d, 1, 2]
                        Mn_running[la, d, 1, 2] += (w / Wn_running[la, d, 1, 2]) * (current_x - Mn_running[la, d, 0, 2])
                        Sn_running[la, d, 0, 2] = Sn_running[la, d, 1, 2]
                        Sn_running[la, d, 1, 2] += w * ((current_x - Mn_running[la, d, 0, 2]) * (current_x - Mn_running[la, d, 1, 2]))
                        if Wn_running[la, d, 0, 2] * vol_num[la, d, 2] * (vol_num[la, d, 2] - 1) > 0:
                            SD_vol_running[la, d, 2] = (
                                vol_num[la, d, 2] * Sn_running[la, d, 1, 2] /
                                (Wn_running[la, d, 0, 2] * vol_num[la, d, 2] * (vol_num[la, d, 2] - 1))
                            ) ** 0.5
                        
            
    return vol, vol_weight, vol_num, SD_vol_running

def normalize_and_mask(vol, vol_weight, vol_num, SD_vol_running, grid_lat, grid_depth, roll_av, mask_out_val=0.1):
    """
    Normalize stacked amplitudes by weight and count, calculate standard deviation arrays, and apply masking.
    Parameters:
        vol (np.ndarray): Stacked amplitudes
        vol_weight (np.ndarray): Stacking weights
        vol_num (np.ndarray): Number of contributing RFs
        SD_vol_running (np.ndarray): Standard deviation array
        grid_lat, grid_depth (np.ndarray): Grid information
        roll_avg: Rolling average
        mask_out_val (float): Weight threshold for masking, default 0.1
    Returns:
        tuple: (weight_norm_vol, num_norm_vol, SD_vol_minus2SD_running, SD_vol_plus2SD_running, Masked_volume, Masked_volume_abs, SD_vol_abv_2SD_running
        vol_merge_SD_plus2SE, vol_merge_SD_minus2SE, vol_plot_merge)
    """

    # Calculate sign of stack
    vol_sign = np.zeros_like(vol)
    vol_sign[:,:,0]=np.sign(vol[:,:,0])
    vol_sign[:,:,1]=np.sign(vol[:,:,1])
    vol_sign[:,:,2]=np.sign(vol[:,:,2])

    # Normalize and calculate standard deviation
    weight_norm_vol = np.where(vol_weight > 0, vol / vol_weight, 0)
    num_norm_vol = np.where(vol_num > 0, vol / vol_num, 0)

    SD_vol_minus2SD_running = np.where(vol_weight > 1, num_norm_vol - 2 * SD_vol_running, 0)
    SD_vol_plus2SD_running = np.where(vol_weight > 1, num_norm_vol + 2 * SD_vol_running, 0)

    # Mask out low-weight data
    Masked_volume = np.where(vol_weight < mask_out_val, 0, weight_norm_vol)
    Masked_volume_abs = np.where(vol_weight < mask_out_val, 0, 1)

    # Mask standard deviation outside the threshold
    SD_vol_abv_2SD_running = np.zeros_like(vol_weight)
    SD_vol_abv_2SD_running = np.where(vol_weight < mask_out_val, 0, SD_vol_abv_2SD_running)
    SD_vol_abv_2SD_running = np.where(SD_vol_plus2SD_running < 0, -1, SD_vol_abv_2SD_running)

    # Combine phases in stack based on sign criteria
    mask = (vol_sign[..., 0] == vol_sign[..., 1]) & (vol_sign[..., 0] == -vol_sign[..., 2]) &  (vol_weight[...,0] > mask_out_val) & (vol_weight[...,1] > mask_out_val) & (vol_weight[...,1] > mask_out_val) 
    weight_norm_vol[..., 3] = np.where(mask, weight_norm_vol[..., 0] + weight_norm_vol[..., 1] - weight_norm_vol[..., 2], weight_norm_vol[..., 3])

    # Smooth merged grid
    vol_merge = weight_norm_vol[:, :, 3]
    vol_merge_SD = np.sqrt(np.sum(SD_vol_running[:, :, :3] ** 2, axis=2))

    # Initialize arrays for plotting
    vol_plot_merge = np.zeros((len(grid_lat), len(grid_depth)))
    vol_plot_merge_SD = np.zeros((len(grid_lat), len(grid_depth)))

    #WHY IS THIS ROLLING AVERAGE HERE?
    # Apply rolling average
    for lat_cur in range(len(grid_lat)):
        for dep_index in range(roll_av, len(grid_depth) - roll_av):
            # Sum over the rolling window
            window = vol_merge[lat_cur, dep_index - roll_av:dep_index + roll_av + 1]
            window_SD = vol_merge_SD[lat_cur, dep_index - roll_av:dep_index + roll_av + 1]
            
            # Compute the mean for the current depth index
            vol_plot_merge[lat_cur, dep_index] = np.mean(window)
            vol_plot_merge_SD[lat_cur, dep_index] = np.mean(window_SD)

    # Calculate the standard deviation limits
    vol_merge_SD_plus2SE = vol_plot_merge + 2 * vol_plot_merge_SD
    vol_merge_SD_minus2SE = vol_plot_merge - 2 * vol_plot_merge_SD

    return (weight_norm_vol, SD_vol_minus2SD_running, SD_vol_plus2SD_running, Masked_volume, Masked_volume_abs, SD_vol_abv_2SD_running, vol_merge_SD_plus2SE, vol_merge_SD_minus2SE, vol_plot_merge)



def pick_peaks(weight_norm_vol, vol_weight, grid_lat, grid_lon, grid_dist, grid_depth, vol_plot_merge, vol_merge_SD_minus2SE, vol_merge_SD_plus2SE, SD_vol_minus2SD_running, SD_vol_plus2SD_running, roll_av=50, min_amp=0.04, mask_out_val=0.1):
    """
    Detect positive and negative amplitude peaks in the stacked section using rolling window and scipy.signal.find_peaks.
    Parameters:
        weight_norm_vol (np.ndarray): Normalized stacked amplitudes
        vol_weight (np.ndarray): Stacking weights
        grid_lat, grid_lon, grid_dist, grid_depth: Grid arrays
        roll_av (int): Rolling window size for peak detection
        min_amp (float): Minimum amplitude threshold
        mask_out_val (float): Weight threshold for masking
    Returns:
        tuple: (dist_peaks, lat_peaks, lon_peaks, dep_peaks, amp_peaks, ... for negative peaks)
    """
    dist_peaks_PS, lat_peaks_PS, lon_peaks_PS, dep_peaks_PS, amp_peaks_PS = [], [], [], [], []
    dist_peaks_neg_PS, lat_peaks_neg_PS, lon_peaks_neg_PS, dep_peaks_neg_PS, amp_peaks_neg_PS = [], [], [], [], []

    dist_peaks_PPS, lat_peaks_PPS, lon_peaks_PPS, dep_peaks_PPS, amp_peaks_PPS = [], [], [], [], []
    dist_peaks_neg_PPS, lat_peaks_neg_PPS, lon_peaks_neg_PPS, dep_peaks_neg_PPS, amp_peaks_neg_PPS = [], [], [], [], []

    dist_peaks_PSS, lat_peaks_PSS, lon_peaks_PSS, dep_peaks_PSS, amp_peaks_PSS = [], [], [], [], []
    dist_peaks_neg_PSS, lat_peaks_neg_PSS, lon_peaks_neg_PSS, dep_peaks_neg_PSS, amp_peaks_neg_PSS = [], [], [], [], []
    
    dist_peaks_merge, lat_peaks_merge, lon_peaks_merge, dep_peaks_merge, amp_peaks_merge = [], [], [], [], []
    dist_peaks_neg_merge, lat_peaks_neg_merge, lon_peaks_neg_merge, dep_peaks_neg_merge, amp_peaks_neg_merge = [], [], [], [], []
    dist_peaks_merge_max, lat_peaks_merge_max, lon_peaks_merge_max, dep_peaks_merge_max, amp_peaks_merge_max = [], [], [], [], []

    # Rolling window for peak picking
    roll_array_prev = np.asarray(range(-(roll_av // 2) - 1, (roll_av // 2)))
    roll_array = np.asarray(range(-(roll_av // 2), (roll_av // 2) + 1))
    roll_array_next = np.asarray(range(-(roll_av // 2) + 1, (roll_av // 2) + 2))

    for i in range(len(grid_lat)):
        amps = weight_norm_vol[i, :]
        lat_cur = grid_lat[i]
        dist_cur = grid_dist[i]
        lon_cur = grid_lon[i]
        weights = vol_weight[i, :]
        #I do want to include shallow structure, maybe not 12km though-
        index_max = np.argmax(vol_plot_merge[i,20:])
        dist_peaks_merge_max.append(dist_cur)
        lat_peaks_merge_max.append(lat_cur)
        lon_peaks_merge_max.append(lon_cur)
        dep_peaks_merge_max.append(grid_depth[index_max+20]) 
        amp_peaks_merge_max.append(vol_plot_merge[i,index_max+20])
            
        for dep_index in range(roll_av, (len(grid_depth) - 1) - roll_av):
            dep_cur = grid_depth[dep_index]
            prev_amp_PS = sum(weight_norm_vol[[i]*len(roll_array),dep_index+roll_array_prev,0])/roll_av
            cur_amp_PS = sum(weight_norm_vol[[i]*len(roll_array),dep_index+roll_array,0])/roll_av
            next_amp_PS = sum(weight_norm_vol[[i]*len(roll_array),dep_index+roll_array_next,0])/roll_av
            weight_cur = float(vol_weight[i,dep_index,0])
            weights_prev = vol_weight[[i]*len(roll_array),dep_index+roll_array_prev,0]
            no_zeros = (weights_prev == 0).sum()
            SD_cur_PS = SD_vol_minus2SD_running[i,dep_index,0]
            SD_cur_plus_PS = SD_vol_plus2SD_running[i,dep_index,0]
            true_amp_PS = weight_norm_vol[i,dep_index,0]

            prev_amp_PPS = sum(weight_norm_vol[[i]*len(roll_array),dep_index+roll_array_prev,1])/roll_av
            cur_amp_PPS = sum(weight_norm_vol[[i]*len(roll_array),dep_index+roll_array,1])/roll_av
            next_amp_PPS = sum(weight_norm_vol[[i]*len(roll_array),dep_index+roll_array_next,1])/roll_av
            SD_cur_PPS = SD_vol_minus2SD_running[i,dep_index,1]
            SD_cur_plus_PPS = SD_vol_plus2SD_running[i,dep_index,1]
            true_amp_PPS = weight_norm_vol[i,dep_index,1]

            prev_amp_PSS = sum(weight_norm_vol[[i]*len(roll_array),dep_index+roll_array_prev,2])/roll_av
            cur_amp_PSS = sum(weight_norm_vol[[i]*len(roll_array),dep_index+roll_array,2])/roll_av
            next_amp_PSS = sum(weight_norm_vol[[i]*len(roll_array),dep_index+roll_array_next,2])/roll_av
            SD_cur_PSS = SD_vol_minus2SD_running[i,dep_index,2]
            SD_cur_plus_PSS = SD_vol_plus2SD_running[i,dep_index,2]
            true_amp_PSS = weight_norm_vol[i,dep_index,2]

            prev_amp_merge = sum(vol_plot_merge[[i]*len(roll_array),dep_index+roll_array_prev])/roll_av
            cur_amp_merge = sum(vol_plot_merge[[i]*len(roll_array),dep_index+roll_array])/roll_av
            next_amp_merge = sum(vol_plot_merge[[i]*len(roll_array),dep_index+roll_array_next])/roll_av
            SD_cur_merge = vol_merge_SD_minus2SE[i,dep_index]
            SD_cur_plus_merge = vol_merge_SD_plus2SE[i,dep_index]
            true_amp_merge = vol_plot_merge[i,dep_index]

            # Peak conditions for each phase
            if (cur_amp_PS > prev_amp_PS) and (cur_amp_PS > next_amp_PS) and (cur_amp_PS>0) and (prev_amp_PS>0)and (next_amp_PS>0)and (cur_amp_PS > min_amp) and (weight_cur > mask_out_val) and (SD_cur_PS >0) and (no_zeros == 0):
                dist_peaks_PS.append(dist_cur)
                lat_peaks_PS.append(lat_cur)
                lon_peaks_PS.append(lon_cur)
                dep_peaks_PS.append(dep_cur) 
                amp_peaks_PS.append(true_amp_PS)

            if (cur_amp_PS < prev_amp_PS) and (cur_amp_PS < next_amp_PS) and (cur_amp_PS<0) and (prev_amp_PS<0)and (next_amp_PS<0)and (cur_amp_PS < -min_amp) and (weight_cur > mask_out_val) and (SD_cur_plus_PS <0) and (no_zeros == 0):
                dist_peaks_neg_PS.append(dist_cur)
                lat_peaks_neg_PS.append(lat_cur)
                lon_peaks_neg_PS.append(lon_cur)
                dep_peaks_neg_PS.append(dep_cur)  
                amp_peaks_neg_PS.append(true_amp_PS) 

            if (cur_amp_PPS > prev_amp_PPS) and (cur_amp_PPS > next_amp_PPS) and (cur_amp_PPS>0) and (prev_amp_PPS>0)and (next_amp_PPS>0)and (cur_amp_PPS > min_amp) and (weight_cur > mask_out_val) and (SD_cur_PPS >0) and (no_zeros == 0):
                dist_peaks_PPS.append(dist_cur)
                lat_peaks_PPS.append(lat_cur)
                lon_peaks_PPS.append(lon_cur)
                dep_peaks_PPS.append(dep_cur) 
                amp_peaks_PPS.append(true_amp_PPS)

            if (cur_amp_PPS < prev_amp_PPS) and (cur_amp_PPS < next_amp_PPS) and (cur_amp_PPS<0) and (prev_amp_PPS<0)and (next_amp_PPS<0)and (cur_amp_PPS < -min_amp) and (weight_cur > mask_out_val) and (SD_cur_plus_PPS <0) and (no_zeros == 0):
                dist_peaks_neg_PPS.append(dist_cur)
                lat_peaks_neg_PPS.append(lat_cur)
                lon_peaks_neg_PPS.append(lon_cur)
                dep_peaks_neg_PPS.append(dep_cur)  
                amp_peaks_neg_PPS.append(true_amp_PPS) 

            if (cur_amp_PSS > prev_amp_PSS) and (cur_amp_PSS > next_amp_PSS) and (cur_amp_PSS>0) and (prev_amp_PSS>0)and (next_amp_PSS>0)and (cur_amp_PSS > min_amp) and (weight_cur > mask_out_val) and (SD_cur_PSS >0) and (no_zeros == 0):
                dist_peaks_PSS.append(dist_cur)
                lat_peaks_PSS.append(lat_cur)
                lon_peaks_PSS.append(lon_cur)
                dep_peaks_PSS.append(dep_cur) 
                amp_peaks_PSS.append(true_amp_PSS)
            
            if (cur_amp_PSS < prev_amp_PSS) and (cur_amp_PSS < next_amp_PSS) and (cur_amp_PSS<0) and (prev_amp_PSS<0)and (next_amp_PSS<0)and (cur_amp_PSS < -min_amp) and (weight_cur > mask_out_val) and (SD_cur_plus_PSS <0) and (no_zeros == 0):
                dist_peaks_neg_PSS.append(dist_cur)
                lat_peaks_neg_PSS.append(lat_cur)
                lon_peaks_neg_PSS.append(lon_cur)
                dep_peaks_neg_PSS.append(dep_cur)  
                amp_peaks_neg_PSS.append(true_amp_PSS) 

            if (cur_amp_merge > prev_amp_merge) and (cur_amp_merge > next_amp_merge) and (cur_amp_merge>0) and (prev_amp_merge>0)and (next_amp_merge>0)and (cur_amp_merge > min_amp) and (weight_cur > mask_out_val) and (SD_cur_merge >0) and (no_zeros == 0):
                dist_peaks_merge.append(dist_cur)
                lat_peaks_merge.append(lat_cur)
                lon_peaks_merge.append(lon_cur)
                dep_peaks_merge.append(dep_cur) 
                amp_peaks_merge.append(true_amp_merge)

            if (cur_amp_merge < prev_amp_merge) and (cur_amp_merge < next_amp_merge) and (cur_amp_merge<0) and (prev_amp_merge<0)and (next_amp_merge<0)and (cur_amp_merge < -min_amp) and (weight_cur > mask_out_val) and (SD_cur_plus_merge <0) and (no_zeros == 0):
                dist_peaks_neg_merge.append(dist_cur)
                lat_peaks_neg_merge.append(lat_cur)
                lon_peaks_neg_merge.append(lon_cur)
                dep_peaks_neg_merge.append(dep_cur)  
                amp_peaks_neg_merge.append(true_amp_merge)

    return (dist_peaks_PS, lat_peaks_PS, lon_peaks_PS, dep_peaks_PS, amp_peaks_PS,
    dist_peaks_neg_PS, lat_peaks_neg_PS, lon_peaks_neg_PS, dep_peaks_neg_PS, amp_peaks_neg_PS,
    dist_peaks_PPS, lat_peaks_PPS, lon_peaks_PPS, dep_peaks_PPS, amp_peaks_PPS,
    dist_peaks_neg_PPS, lat_peaks_neg_PPS, lon_peaks_neg_PPS, dep_peaks_neg_PPS, amp_peaks_neg_PPS,
    dist_peaks_PSS, lat_peaks_PSS, lon_peaks_PSS, dep_peaks_PSS, amp_peaks_PSS,
    dist_peaks_neg_PSS, lat_peaks_neg_PSS, lon_peaks_neg_PSS, dep_peaks_neg_PSS, amp_peaks_neg_PSS,
    dist_peaks_merge, lat_peaks_merge, lon_peaks_merge, dep_peaks_merge, amp_peaks_merge,
    dist_peaks_neg_merge, lat_peaks_neg_merge, lon_peaks_neg_merge, dep_peaks_neg_merge, amp_peaks_neg_merge,
    dist_peaks_merge_max, lat_peaks_merge_max, lon_peaks_merge_max, dep_peaks_merge_max, amp_peaks_merge_max)


def plot_cross_section(output_path, grid_dist, grid_depth, weight_norm_vol, maskdist, maskdep, 
    startlat, startlon, endlat, endlon, stacking_smooth_fresnel, gauss_direct, gauss_mult, vol_plot_merge,
    dist_peaks_PS, dep_peaks_PS, dist_peaks_neg_PS, dep_peaks_neg_PS,
    dist_peaks_PPS, dep_peaks_PPS, dist_peaks_neg_PPS, dep_peaks_neg_PPS,
    dist_peaks_PSS, dep_peaks_PSS, dist_peaks_neg_PSS, dep_peaks_neg_PSS,
    dist_peaks_merge_max, dep_peaks_merge_max, dist_peaks_neg_merge, dep_peaks_neg_merge,
    color_saturate=0.07):
    """
    Plot the CCP stacked cross-section with peaks and masked points.
    Parameters:
        grid_dist (np.ndarray): Cross-section distances
        grid_depth (np.ndarray): Depths
        weight_norm_vol (np.ndarray): Normalized stacked amplitudes
        maskdist, maskdep: Masked points for plotting
        dist_peaks, dep_peaks: Positive peak locations
        dist_peaks_neg, dep_peaks_neg: Negative peak locations
        color_saturate (float): Color scale saturation
    Returns:
        void
    """
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # PS phase
    pc1 = axes[0].pcolormesh(grid_dist, grid_depth, weight_norm_vol[:,:,0].T, vmin=-color_saturate, vmax=color_saturate, cmap='RdYlBu_r')
    axes[0].plot(maskdist, maskdep, 'ws', markeredgewidth=0.0)
    axes[0].plot(dist_peaks_PS, dep_peaks_PS, 'ro')
    axes[0].plot(dist_peaks_neg_PS, dep_peaks_neg_PS, 'bo')
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Depth (km)")
    axes[0].set_title("PS phase")

    # PPS phase
    pc2 = axes[1].pcolormesh(grid_dist, grid_depth, weight_norm_vol[:,:,1].T, vmin=-color_saturate, vmax=color_saturate, cmap='RdYlBu_r')
    axes[1].plot(maskdist, maskdep, 'ws', markeredgewidth=0.0)
    axes[1].plot(dist_peaks_PPS, dep_peaks_PPS, 'ro')
    axes[1].plot(dist_peaks_neg_PPS, dep_peaks_neg_PPS, 'bo')
    axes[1].invert_yaxis()
    axes[1].set_ylabel("Depth (km)")
    axes[1].set_title("PPS phase")

    # PSS phase
    pc3 = axes[2].pcolormesh(grid_dist, grid_depth, weight_norm_vol[:,:,2].T, vmin=-color_saturate, vmax=color_saturate, cmap='RdYlBu')
    axes[2].plot(maskdist, maskdep, 'ws', markeredgewidth=0.0)
    axes[2].plot(dist_peaks_PSS, dep_peaks_PSS, 'ro')
    axes[2].plot(dist_peaks_neg_PSS, dep_peaks_neg_PSS, 'bo')

    axes[2].invert_yaxis()
    axes[2].set_ylabel("Depth (km)")
    axes[2].set_title("PSS phase")

    # Merged phase
    pc4 = axes[3].pcolormesh(grid_dist, grid_depth, vol_plot_merge.T, vmin=-color_saturate, vmax=color_saturate, cmap='RdYlBu_r')
    axes[3].plot(maskdist, maskdep, 'ws', markeredgewidth=0.0)
    axes[3].plot(dist_peaks_merge_max, dep_peaks_merge_max, 'ro')
    axes[3].plot(dist_peaks_neg_merge, dep_peaks_neg_merge, 'bo')
    axes[3].set_xlabel("Distance (km)")
    axes[3].set_ylabel("Depth (km)")
    axes[3].invert_yaxis()
    axes[3].set_title("Multiphase Stack")

    plt.tight_layout()
    plt.savefig(f"{output_path}/ccp_Multiphase_cross-section_{startlat}_{startlon}_{endlat}_{endlon}_fres_{stacking_smooth_fresnel}_Gauss{gauss_direct}_{gauss_mult}_smooth_crust_narrow_window.png", dpi=300, bbox_inches="tight")
    plt.close()



def save_outputs(
    output_path, startlat, startlon, endlat, endlon, stacking_smooth_fresnel, gauss_direct, gauss_mult,
    grid_lat, grid_lon, grid_dist, grid_depth, weight_norm_vol,
    dist_peaks_PS, lat_peaks_PS, lon_peaks_PS, dep_peaks_PS, amp_peaks_PS,
    dist_peaks_neg_PS, lat_peaks_neg_PS, lon_peaks_neg_PS, dep_peaks_neg_PS, amp_peaks_neg_PS,
    dist_peaks_PPS, lat_peaks_PPS, lon_peaks_PPS, dep_peaks_PPS, amp_peaks_PPS,
    dist_peaks_neg_PPS, lat_peaks_neg_PPS, lon_peaks_neg_PPS, dep_peaks_neg_PPS, amp_peaks_neg_PPS,
    dist_peaks_PSS, lat_peaks_PSS, lon_peaks_PSS, dep_peaks_PSS, amp_peaks_PSS,
    dist_peaks_neg_PSS, lat_peaks_neg_PSS, lon_peaks_neg_PSS, dep_peaks_neg_PSS, amp_peaks_neg_PSS,
    dist_peaks_merge, lat_peaks_merge, lon_peaks_merge, dep_peaks_merge, amp_peaks_merge,
    dist_peaks_neg_merge, lat_peaks_neg_merge, lon_peaks_neg_merge, dep_peaks_neg_merge, amp_peaks_neg_merge,
    dist_peaks_merge_max, lat_peaks_merge_max, lon_peaks_merge_max, dep_peaks_merge_max, amp_peaks_merge_max,
    masklat, masklon, maskdist, maskdep, mask_out_val, Masked_volume, Masked_volume_abs, 
    SD_vol_abv_2SD_running, vol, vol_weight, SD_vol_minus2SD_running, SD_vol_plus2SD_running, SD_vol_running, 
    vol_plot_merge, vol_merge_SD_minus2SE, vol_merge_SD_plus2SE):
    """
    Save cross-section, peaks, and mask information to text files.
    Parameters:
        output_path (str): Output directory
        startlat, startlon, endlat, endlon: Cross-section endpoints
        stacking_smooth_fresnel, gauss: Parameters
        grid_lat, grid_lon, grid_dist, grid_depth: Grid arrays
        weight_norm_vol (np.ndarray): Normalized stacked amplitudes
        dist_peaks, lat_peaks, lon_peaks, dep_peaks, amp_peaks: Positive peaks
        dist_peaks_neg, lat_peaks_neg, lon_peaks_neg, dep_peaks_neg, amp_peaks_neg: Negative peaks
        masklat, masklon, maskdist, maskdep: Masked points
    Returns:
        void
    """

    #stick in smoothed merged volume
    weight_norm_vol[:,:,3]=vol_plot_merge
    SD_vol_minus2SD_running[:,:,3] = vol_merge_SD_minus2SE
    SD_vol_plus2SD_running[:,:,3] = vol_merge_SD_plus2SE

    # Save cross-section to a text file
    outfile = f"{output_path}/CROSS-SECTION_Multiphase_{startlat}_{startlon}_{endlat}_{endlon}_fres_{stacking_smooth_fresnel}_Gauss{gauss_direct}_{gauss_mult}_slice_smooth_crust_narrow_window.txt"
    with open(outfile, "w") as text_file:
        for i in range(len(grid_lat)):
            for j in range(len(grid_depth)):
                text_file.write(f"{grid_lon[i]} {grid_lat[i]} {grid_dist[i]} {grid_depth[j]} {vol_plot_merge[i, j]}\n")

    # Save picked peaks to text files
    outfile_pos = f"{output_path}/CROSS-SECTION_Multiphase_{startlat}_{startlon}_{endlat}_{endlon}_fres_{stacking_smooth_fresnel}_Gauss{gauss_direct}_{gauss_mult}_peakspos_smooth_crust_narrow_window.txt"
    with open(outfile_pos, "w") as text_file:
        for x in range(len(lat_peaks_merge_max)):
            text_file.write(f"{lon_peaks_merge_max[x]} {lat_peaks_merge_max[x]} {dist_peaks_merge_max[x]} {dep_peaks_merge_max[x]} {amp_peaks_merge_max[x]}\n")

    outfile_neg = f"{output_path}/CROSS-SECTION_Multiphase_{startlat}_{startlon}_{endlat}_{endlon}_fres_{stacking_smooth_fresnel}_Gauss{gauss_direct}_{gauss_mult}_peaksneg_smooth_crust_narrow_window.txt"
    with open(outfile_neg, "w") as text_file_neg:
        for x in range(len(lat_peaks_neg_merge)):
            text_file_neg.write(f"{lon_peaks_neg_merge[x]} {lat_peaks_neg_merge[x]} {dist_peaks_neg_merge[x]} {dep_peaks_neg_merge[x]} {amp_peaks_neg_merge[x]}\n")

    # Save mask information to text file
    outfile_mask = f"{output_path}/CROSS-SECTION_Multiphase_{startlat}_{startlon}_{endlat}_{endlon}_fres_{stacking_smooth_fresnel}_Gauss{gauss_direct}_{gauss_mult}_smooth_crust_narrow_window.txt"
    with open(outfile_mask, "w") as text_file:
        for x in range(len(maskdist)):
            text_file.write(f"{masklon[x]} {masklat[x]} {maskdist[x]} {maskdep[x]}\n")

    # Put together one volume of lat,lon,depth containing all relevent infomation
    final_output = np.zeros([len(grid_lat), len(grid_depth), 4, 9])
    for la in range(len(grid_lat)):
        for d in range(len(grid_depth)):
            final_output[la,d,:,0]=vol[la,d,:]
            final_output[la,d,:,1]=vol_weight[la,d,:]
            final_output[la,d,:,2]=weight_norm_vol[la,d,:]
            final_output[la,d,:,3]=SD_vol_running[la,d,:]
            final_output[la,d,:,4]=SD_vol_minus2SD_running[la,d,:]
            final_output[la,d,:,5]=SD_vol_plus2SD_running[la,d,:]
            final_output[la,d,:,6]=SD_vol_abv_2SD_running[la,d,:]
            final_output[la,d,:,7]=Masked_volume[la,d,:]
            final_output[la,d,:,8]=Masked_volume_abs[la,d,:]

    outputdict={}
    outputdict['grid']=final_output
    outputdict['peaks'] = {'pos': {}, 'neg': {}}
    outputdict['peaks']['pos']['lat']={} 
    outputdict['peaks']['pos']['lon']={} 
    outputdict['peaks']['pos']['dist']={} 
    outputdict['peaks']['pos']['dep']={} 
    outputdict['peaks']['pos']['amp']={} 
    outputdict['peaks']['neg']['lat']={} 
    outputdict['peaks']['neg']['lon']={} 
    outputdict['peaks']['neg']['dist']={} 
    outputdict['peaks']['neg']['dep']={} 
    outputdict['peaks']['neg']['amp']={} 

    # Save all peak arrays correctly (positive and negative, for all phases)
    outputdict['peaks']['pos']['lat']['PS'] = lat_peaks_PS
    outputdict['peaks']['pos']['lon']['PS'] = lon_peaks_PS
    outputdict['peaks']['pos']['dist']['PS'] = dist_peaks_PS
    outputdict['peaks']['pos']['dep']['PS'] = dep_peaks_PS
    outputdict['peaks']['pos']['amp']['PS'] = amp_peaks_PS
    outputdict['peaks']['neg']['lat']['PS'] = lat_peaks_neg_PS
    outputdict['peaks']['neg']['lon']['PS'] = lon_peaks_neg_PS
    outputdict['peaks']['neg']['dist']['PS'] = dist_peaks_neg_PS
    outputdict['peaks']['neg']['dep']['PS'] = dep_peaks_neg_PS
    outputdict['peaks']['neg']['amp']['PS'] = amp_peaks_neg_PS

    outputdict['peaks']['pos']['lat']['PPS'] = lat_peaks_PPS
    outputdict['peaks']['pos']['lon']['PPS'] = lon_peaks_PPS
    outputdict['peaks']['pos']['dist']['PPS'] = dist_peaks_PPS
    outputdict['peaks']['pos']['dep']['PPS'] = dep_peaks_PPS
    outputdict['peaks']['pos']['amp']['PPS'] = amp_peaks_PPS
    outputdict['peaks']['neg']['lat']['PPS'] = lat_peaks_neg_PPS
    outputdict['peaks']['neg']['lon']['PPS'] = lon_peaks_neg_PPS
    outputdict['peaks']['neg']['dist']['PPS'] = dist_peaks_neg_PPS
    outputdict['peaks']['neg']['dep']['PPS'] = dep_peaks_neg_PPS
    outputdict['peaks']['neg']['amp']['PPS'] = amp_peaks_neg_PPS

    outputdict['peaks']['pos']['lat']['PSS'] = lat_peaks_PSS
    outputdict['peaks']['pos']['lon']['PSS'] = lon_peaks_PSS
    outputdict['peaks']['pos']['dist']['PSS'] = dist_peaks_PSS
    outputdict['peaks']['pos']['dep']['PSS'] = dep_peaks_PSS
    outputdict['peaks']['pos']['amp']['PSS'] = amp_peaks_PSS
    outputdict['peaks']['neg']['lat']['PSS'] = lat_peaks_neg_PSS
    outputdict['peaks']['neg']['lon']['PSS'] = lon_peaks_neg_PSS
    outputdict['peaks']['neg']['dist']['PSS'] = dist_peaks_neg_PSS
    outputdict['peaks']['neg']['dep']['PSS'] = dep_peaks_neg_PSS
    outputdict['peaks']['neg']['amp']['PSS'] = amp_peaks_neg_PSS

    outputdict['peaks']['pos']['lat']['merge'] = lat_peaks_merge
    outputdict['peaks']['pos']['lon']['merge'] = lon_peaks_merge
    outputdict['peaks']['pos']['dist']['merge'] = dist_peaks_merge
    outputdict['peaks']['pos']['dep']['merge'] = dep_peaks_merge
    outputdict['peaks']['pos']['amp']['merge'] = amp_peaks_merge
    outputdict['peaks']['neg']['lat']['merge'] = lat_peaks_neg_merge
    outputdict['peaks']['neg']['lon']['merge'] = lon_peaks_neg_merge
    outputdict['peaks']['neg']['dist']['merge'] = dist_peaks_neg_merge
    outputdict['peaks']['neg']['dep']['merge'] = dep_peaks_neg_merge
    outputdict['peaks']['neg']['amp']['merge'] = amp_peaks_neg_merge

    outputdict['peaks']['pos']['lat']['merge_max'] = lat_peaks_merge_max
    outputdict['peaks']['pos']['lon']['merge_max'] = lon_peaks_merge_max
    outputdict['peaks']['pos']['dist']['merge_max'] = dist_peaks_merge_max
    outputdict['peaks']['pos']['dep']['merge_max'] = dep_peaks_merge_max
    outputdict['peaks']['pos']['amp']['merge_max'] = amp_peaks_merge_max

    outputdict['mask'] = {'lat': masklat, 'lon': masklon, 'dep': maskdep, 'dist': maskdist, 'val': mask_out_val}
    outputdict['axis'] = {'lat': grid_lat, 'lon': grid_lon, 'dist': grid_dist, 'dep': grid_depth}
    outputdict['gauss'] = {'Ps': gauss_direct, 'mult': gauss_mult}
    outputdict['fres'] = stacking_smooth_fresnel

    output_file = f"{output_path}/CROSS-SECTION_Multiphase_{startlat}_{startlon}_{endlat}_{endlon}_fres_{stacking_smooth_fresnel}_Gauss{gauss_direct}_{gauss_mult}_smooth_crust_narrow_window.pkl"
    with open(output_file, 'wb') as out_put_write:
        pickle.dump(outputdict, out_put_write)


   
gauss_direct = '_6'
gauss_mult = '_2' #because i'm a fucking idiot
stacking_smooth_fresnel = 0.2

output_path = '/raid2/cg812/Synthetic_RF_with_two_chambers'        # Path to save the results (output png and text files).      # Path to RF data that will be processed for multiples.
mask_out_val = 0.01                                                 # Masking threshold.   # Start and end points of the cross-section.
roll_av = 5                                                        # Rolling average for smoothing and peak detection. Is 2km onwards 
min_amp = 0.04                                                      # Amplitude threshold for peak detection.
startlat, startlon, endlat, endlon = 65.025, -17.14, 65.025, -16.14

grid_lon, grid_lat, grid_depth, grid_dist = load_grid(f'/raid2/cg812/Grids/cross-section_EW_C.lonlat', 60)


RF_files_direct = ['/raid2/cg812/Synthetic_RF_with_two_chambers/Gauss_6_crust_gradient.pkl']*len(glob.glob('/raid2/cg812/All_together/Gauss_6.0/*[!.png]'))
RF_files_mult = ['/raid2/cg812/Synthetic_RF_with_two_chambers/Gauss_2_crust_gradient.pkl']*len(glob.glob('/raid2/cg812/All_together/Gauss_2.0/*[!.png]'))

# Call function for processing the RF files for the PS phase.
vol, vol_weight, vol_num, SD_vol_running, Mn_running, Sn_running, Wn_running = process_rf_files_ps(RF_files_direct, grid_lon, grid_lat, grid_depth, stacking_smooth_fresnel, gauss_direct)
# Call function for processing the RF files for the multiples (PPS and PSS).
vol, vol_weight, vol_num, SD_vol_running = process_rf_files_multiples(RF_files_mult, grid_lon, grid_lat, grid_depth, stacking_smooth_fresnel, gauss_mult, vol, vol_weight, vol_num, SD_vol_running, Mn_running, Sn_running, Wn_running)
# Call function for normalising and masking the output from process_rf_files and process_rf_files_multiples.
weight_norm_vol, SD_vol_minus2SD_running, SD_vol_plus2SD_running, Masked_volume, Masked_volume_abs, SD_vol_abv_2SD_running, vol_merge_SD_plus2SE, vol_merge_SD_minus2SE, vol_plot_merge = normalize_and_mask(vol, vol_weight, vol_num, SD_vol_running, grid_lat, grid_depth, roll_av, mask_out_val)

# Masking for plotting
dists, deps = np.where(vol_weight[:,:,0] < mask_out_val)
maskdist = grid_dist[dists]
masklat = grid_lat[dists]
masklon = grid_lon[dists]
maskdep = grid_depth[deps]

# Call function for picking postitive and negative amplitude peaks.
(dist_peaks_PS, lat_peaks_PS, lon_peaks_PS, dep_peaks_PS, amp_peaks_PS,
dist_peaks_neg_PS, lat_peaks_neg_PS, lon_peaks_neg_PS, dep_peaks_neg_PS, amp_peaks_neg_PS,
dist_peaks_PPS, lat_peaks_PPS, lon_peaks_PPS, dep_peaks_PPS, amp_peaks_PPS,
dist_peaks_neg_PPS, lat_peaks_neg_PPS, lon_peaks_neg_PPS, dep_peaks_neg_PPS, amp_peaks_neg_PPS,
dist_peaks_PSS, lat_peaks_PSS, lon_peaks_PSS, dep_peaks_PSS, amp_peaks_PSS,
dist_peaks_neg_PSS, lat_peaks_neg_PSS, lon_peaks_neg_PSS, dep_peaks_neg_PSS, amp_peaks_neg_PSS,
dist_peaks_merge, lat_peaks_merge, lon_peaks_merge, dep_peaks_merge, amp_peaks_merge,
dist_peaks_neg_merge, lat_peaks_neg_merge, lon_peaks_neg_merge, dep_peaks_neg_merge, amp_peaks_neg_merge,
dist_peaks_merge_max, lat_peaks_merge_max, lon_peaks_merge_max, dep_peaks_merge_max, amp_peaks_merge_max) = pick_peaks(
    weight_norm_vol, vol_weight, grid_lat, grid_lon, grid_dist, grid_depth,
    vol_plot_merge, vol_merge_SD_minus2SE, vol_merge_SD_plus2SE, SD_vol_minus2SD_running, SD_vol_plus2SD_running,
    roll_av, min_amp, mask_out_val)

# Call function for plotting the cross-section.
plot_cross_section(output_path, grid_dist, grid_depth, weight_norm_vol, maskdist, maskdep, 
    startlat, startlon, endlat, endlon, stacking_smooth_fresnel, gauss_direct, gauss_mult, vol_plot_merge,
    dist_peaks_PS, dep_peaks_PS, dist_peaks_neg_PS, dep_peaks_neg_PS,
    dist_peaks_PPS, dep_peaks_PPS, dist_peaks_neg_PPS, dep_peaks_neg_PPS,
    dist_peaks_PSS, dep_peaks_PSS, dist_peaks_neg_PSS, dep_peaks_neg_PSS,
    dist_peaks_merge_max, dep_peaks_merge_max, dist_peaks_neg_merge, dep_peaks_neg_merge)

# Call function for saving the outputs.
save_outputs(output_path, startlat, startlon, endlat, endlon, stacking_smooth_fresnel, gauss_direct, gauss_mult,
    grid_lat, grid_lon, grid_dist, grid_depth, weight_norm_vol,
    dist_peaks_PS, lat_peaks_PS, lon_peaks_PS, dep_peaks_PS, amp_peaks_PS,
    dist_peaks_neg_PS, lat_peaks_neg_PS, lon_peaks_neg_PS, dep_peaks_neg_PS, amp_peaks_neg_PS,
    dist_peaks_PPS, lat_peaks_PPS, lon_peaks_PPS, dep_peaks_PPS, amp_peaks_PPS,
    dist_peaks_neg_PPS, lat_peaks_neg_PPS, lon_peaks_neg_PPS, dep_peaks_neg_PPS, amp_peaks_neg_PPS,
    dist_peaks_PSS, lat_peaks_PSS, lon_peaks_PSS, dep_peaks_PSS, amp_peaks_PSS,
    dist_peaks_neg_PSS, lat_peaks_neg_PSS, lon_peaks_neg_PSS, dep_peaks_neg_PSS, amp_peaks_neg_PSS,
    dist_peaks_merge, lat_peaks_merge, lon_peaks_merge, dep_peaks_merge, amp_peaks_merge,
    dist_peaks_neg_merge, lat_peaks_neg_merge, lon_peaks_neg_merge, dep_peaks_neg_merge, amp_peaks_neg_merge,
    dist_peaks_merge_max, lat_peaks_merge_max, lon_peaks_merge_max, dep_peaks_merge_max, amp_peaks_merge_max,
    masklat, masklon, maskdist, maskdep, mask_out_val, Masked_volume, Masked_volume_abs, 
    SD_vol_abv_2SD_running, vol, vol_weight, SD_vol_minus2SD_running, SD_vol_plus2SD_running, SD_vol_running, 
    vol_plot_merge, vol_merge_SD_minus2SE, vol_merge_SD_plus2SE)

print("OUTPUTS saved to:", output_path)
