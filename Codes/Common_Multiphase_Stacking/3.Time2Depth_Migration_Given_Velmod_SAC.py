import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from obspy.taup import TauPyModel
from scipy.interpolate import interp1d
import pickle
import glob
import time
import sys
from geopy.distance import great_circle
import math
import multiprocessing
import os
import logging
import obspy
from joblib import parallel_config
from joblib import Parallel, delayed

def get_dist_km_for_phase(pierce_arr, dep, radian_dist):
    """
    Extract the horizontal distance (in km) from the receiver to the pierce point at a given depth for a seismic phase.
    Ensures the selected pierce point (from TauPy's "get_pierce_points" function) is on the receiver-side (after the turning point).

    Parameters:
        pierce_arr (np.ndarray): Array of pierce points for a phase.
        dep (float): Target depth (km) to extract the pierce point for.
        radian_dist (float): Total epicentral distance from source to receiver in radians.

    Returns:
        float: Horizontal distance (km) from the receiver to the pierce point at the specified depth.
    """
    
    """ Access the pierce[phase].pierce array columns by their name. """
    if hasattr(pierce_arr, 'dtype') and pierce_arr.dtype.names:
        if 'depth' in pierce_arr.dtype.names:
            depths_pierce = pierce_arr['depth']
        else:
            raise ValueError(f"Unknown depth field in pierce array: {pierce_arr.dtype.names}")
        if 'dist' in pierce_arr.dtype.names:
            dist_pierce = pierce_arr['dist']
        else:
            dist_pierce = pierce_arr[list(pierce_arr.dtype.names)[2]]
    else:
        if pierce_arr.ndim == 1:
            pierce_arr = pierce_arr.reshape(1, -1)
        depths_pierce = pierce_arr[:, 1]
        dist_pierce = pierce_arr[:, 2]
    
    """ Find all indices where depth is close to desired depth (within 1.1 km) """
    indices = np.where(np.isclose(depths_pierce, dep, atol=1.1))[0]

    """ Only consider pierce points strictly after the turning point (i.e., deepest depth)
    so we can be sure this is receiver-side. Then pick the closest index to the desired depth."""
    turning_idx = np.argmax(depths_pierce)
    receiver_side = indices[indices > turning_idx]
    if len(receiver_side) > 0:
        idx = receiver_side[np.argmin(np.abs(depths_pierce[receiver_side] - dep))]
    else:
        after_turning = np.arange(turning_idx + 1, len(depths_pierce))
        if len(after_turning) > 0:
            idx = after_turning[np.argmin(np.abs(depths_pierce[after_turning] - dep))]
        else:
            idx = np.argmin(np.abs(depths_pierce - dep))
    return (radian_dist - dist_pierce[idx]) * (180 / np.pi) * 111

def create_dep_time_mat(npz_file, epi, src_dep=30, dep_lim=141, plot=False, rf_file=None):
    """
    Calculate travel times and horizontal distances for various seismic phases at different depths.

    Parameters:
        npz_file (str): Path to velocity model file for the station being processed.
        epi (float): Epicentral distance of earthquake being processed.
        src_dep (float): Source depth of earthquake being processed.
        dep_lim (float): Maximum depth being analysed.
        plot (bool): Generate diagnostic plots.
        rf_file (str): RF file being processed.

    Returns:
        time (array): travel-times for each phase at each depth.
        dist (array): horizontal distances for each phase at each depth.
    """

    depths = np.zeros(dep_lim)
    model = TauPyModel(model=npz_file)
    #model = TauPyModel(model='/Users/rita_kounoudis/CuBES_CCP/FILES/prem_added_discon_taup.npz')
    time = np.zeros([dep_lim, 3])
    dist = np.zeros([dep_lim, 3])

    logging.info("Calculating travel times and horizontal distances for P, Ps, PpPs, PsPs")

    """ Calculate the travel-time of each phase (P, Ps, PpPs, PsPs), using specified input velocity model above.
    Then calculates the pierce points at the specified depth """
    for dep in range(1, dep_lim):
        logging.debug(f"Calculating pierce points at depth: {dep}km")
        ps = f'P{dep}s'
        pps = f'PPv{dep}s'
        pss = f'PSv{dep}s'
        arr = model.get_travel_times(source_depth_in_km=src_dep, distance_in_degree=epi, phase_list=('P', ps, pps, pss))

        print(arr)
        if len(arr) < 2:
            logging.warning(f"Skipping depth {dep} km, phases missing")
            continue
        logging.debug(f"Depth {dep} km has {len(arr)} phases")
        
        depths[dep] = deppierce = model.get_pierce_points(source_depth_in_km=src_dep, distance_in_degree=epi, phase_list=('P', ps, pps, pss), receiver_depth_in_km=0)
        

        radian_dist = (epi * np.pi) / 180
        dist_km_Ps = get_dist_km_for_phase(pierce[1].pierce, dep, radian_dist)
        dist_km_pps = get_dist_km_for_phase(pierce[2].pierce, dep, radian_dist)
        dist_km_pss = get_dist_km_for_phase(pierce[3].pierce, dep, radian_dist)

        """ If travel times exist for all 4 phases then calculate the travel-time of each phase from the Direct P arrival.
        Assign each phase's time and distance to the arrays created above."""
        if len(arr) == 4:
            time[dep, 0] = arr[1].time - arr[0].time
            time[dep, 1] = arr[2].time - arr[0].time
            time[dep, 2] = arr[3].time - arr[0].time
            dist[dep, 0] = dist_km_Ps
            dist[dep, 1] = dist_km_pps
            dist[dep, 2] = dist_km_pss
        elif len(arr) > 4: # Deal with multiple triplication arrivals, take each first phase arrival
            ps_arr_found = pps_arr_found = pss_arr_found = False
            for i in range(len(arr)):
                name = arr[i].name
                if name == ps and not ps_arr_found:
                    time[dep, 0] = arr[i].time - arr[0].time
                    dist[dep, 0] = get_dist_km_for_phase(pierce[i].pierce, dep, radian_dist)
                    ps_arr_found = True
                if name == pps and not pps_arr_found:
                    time[dep, 1] = arr[i].time - arr[0].time
                    dist[dep, 1] = get_dist_km_for_phase(pierce[i].pierce, dep, radian_dist)
                    pps_arr_found = True
                if name == pss and not pss_arr_found:
                    time[dep, 2] = arr[i].time - arr[0].time
                    dist[dep, 2] = get_dist_km_for_phase(pierce[i].pierce, dep, radian_dist)
                    pss_arr_found = True
    
    if plot:
        rf_base = os.path.splitext(os.path.basename(rf_file))[0]
        plt.figure()
        plt.plot(dist[:, 0], depths, 'r-', label='Ps')
        plt.plot(dist[:, 1], depths, 'g-', label='PpPs')
        plt.plot(dist[:, 2], depths, 'b-', label='PsPs')
        plt.gca().invert_yaxis()
        plt.xlabel('Distance (km)')
        plt.ylabel('Depth (km)')
        plt.legend()
        plt.savefig(f'distance_vs_depth_{rf_base}.png', dpi=300)
        plt.close()

        plt.figure()
        plt.plot(time[:, 0], depths, 'r-', label='Ps')
        plt.plot(time[:, 1], depths, 'g-', label='PpPs')
        plt.plot(time[:, 2], depths, 'b-', label='PsPs')
        plt.gca().invert_yaxis()
        plt.xlabel('Time (s)')
        plt.ylabel('Depth (km)')
        plt.legend()
        plt.savefig(f'time_vs_depth_{rf_base}.png', dpi=300)
        plt.close()

    return time, dist
    



def process_rf_file(rf_file, model_dir, depths_neg, depths, filter):
    """
    Process a single receiver function file, calculate time-depth conversions, and save the results.

    Parameters:
        rf_file (str): Path to the receiver function file.
        model_dir (str): Path to the velocity model files.
        depths_neg (array): Array of depths for negative depths.
        filter (str): The gaussian filter for RF events.

    Returns:
        void
    """
    print(rf_file)
    stream = obspy.read(rf_file)
    station = stream[0].stats.station
    output_dir = '/raid2/cg812/Migrated_to_depth_3D/5'
    savepath= os.path.join(output_dir, stream[0].stats.station + '_' + str(stream[0].stats.starttime)  + '.pkl')

    
    # Automatically find matching npz_file for this station by calling the previous function.
    

    if not os.path.exists(savepath):
        #npz_file= '/raid2/cg812/Velocity_models/1D_Vp_model_of_region.npz'
        try:
                model_dir= '/raid2/cg812/3D_velocity_models'
                pattern = os.path.join(model_dir, f"*{station}.npz")
                matches = glob.glob(pattern)
                npz_file= matches[0]
                logging.info(f"Station {station} using velocity model: {npz_file}")
        except FileNotFoundError:
            logging.warning(f"Skipping {rf_file}: No velocity model found for station {station}")
            return
        
        sampling_rate = stream[0].stats.sampling_rate
        npts = stream[0].stats.npts

        RF_time = np.arange(0, npts) /sampling_rate 
        amplitude_data = stream[0].data
        RF_amp = amplitude_data / np.max(np.abs(amplitude_data))

        dist = stream[0].stats.dist
        ev_dp = stream[0].stats.evdp
        epi_ang = round(dist)
        logging.info(f"Processing station: {station}, Epicentral angle: {epi_ang}")

        try:
            dep_time_matrix, dist_time_matrix = create_dep_time_mat(npz_file, epi_ang, src_dep=ev_dp, plot=False, rf_file=rf_file)
        except Exception as e:
            logging.error(f"Failed to calculate travel times for: {rf_file} ({e})")
            return

        # Extract travel time information
        time_tPs = dep_time_matrix[:, 0] +5
        time_tPPs = dep_time_matrix[:, 1] +5
        time_tPSs = dep_time_matrix[:, 2] + 5

        # Extract distance information
        Hdist_Ps = dist_time_matrix[:,0]
        Hdist_PPs = dist_time_matrix[:,1]
        Hdist_PSs = dist_time_matrix[:,2]

        # Define time windows for each phase (based on observation)
        TPs_min = np.argmin(np.abs(RF_time - (-4) - 5))  
        TPs_max = np.argmin(np.abs(RF_time - 15 - 5))
        TPPs_min = np.argmin(np.abs(RF_time - (-4)) - 5)
        TPPs_max = np.argmin(np.abs(RF_time - 45) - 5)
        TPSs_min = np.argmin(np.abs(RF_time - (-4)) - 5)
        TPSs_max = np.argmin(np.abs(RF_time - 60) - 5)

        # Account for negative arrival times
        neg_arr_times_tPs = []
        neg_arr_times_tPPs = []
        neg_arr_times_tPSs = []
        negdepstart = -40
        for j in range(len(np.arange(negdepstart, 0))):
            idx = (j * (-1)) + 40
            if idx >= 0 and idx < len(time_tPs):
                neg_arr_times_tPs.append(time_tPs[idx] * -1)
            if idx >= 0 and idx < len(time_tPPs):
                neg_arr_times_tPPs.append(time_tPPs[idx] * -1)
            if idx >= 0 and idx < len(time_tPSs):
                neg_arr_times_tPSs.append(time_tPSs[idx] * -1)
        
        # Prepend negative times to positive arrays
        time_tPs_full = np.array(neg_arr_times_tPs + list(time_tPs))
        time_tPPs_full = np.array(neg_arr_times_tPPs + list(time_tPPs))
        time_tPSs_full = np.array(neg_arr_times_tPSs + list(time_tPSs))

        # Ensure last time values meet minimum thresholds
        if time_tPs[-1]< 15:
            time_tPs[-1]=15
        if time_tPPs[-1]< 45:
            time_tPPs[-1]=45
        if time_tPSs[-1]< 60:
            time_tPSs[-1]=60

        # Interpolate and calculate depth from time
        ftPs = interp1d(time_tPs_full, depths_neg, kind='linear')
        depth_tPs = ftPs(RF_time[TPs_min:TPs_max])

        ftPPs = interp1d(time_tPPs_full, depths_neg, kind='linear')
        depth_tPPs = ftPPs(RF_time[TPPs_min:TPPs_max])

        ftPSs = interp1d(time_tPSs_full, depths_neg, kind='linear')
        depth_tPSs = ftPSs(RF_time[TPSs_min:TPSs_max])

        # Ensure valid depths
        depth_tPs[-1] = max(depth_tPs[-1], 125)
        depth_tPPs[-1] = max(depth_tPPs[-1], 80)
        depth_tPSs[-1] = max(depth_tPSs[-1], 80)


        # Interpolation of RF amplitudes
        fdPs = interp1d(depth_tPs, RF_amp[TPs_min:TPs_max], kind='cubic', fill_value='extrapolate')
        amp_aPs = fdPs(np.arange(0, 125, 0.1))
        amp_aPs_neg = fdPs(np.arange(-4, 125, 0.1))

        fdPPs = interp1d(depth_tPPs, RF_amp[TPPs_min:TPPs_max], kind='cubic', fill_value='extrapolate')
        amp_aPPs = fdPPs(np.arange(0, 80, 0.1))
        amp_aPPs_neg = fdPPs(np.arange(-4, 80, 0.1))

        fdPSs = interp1d(depth_tPSs, RF_amp[TPSs_min:TPSs_max], kind='cubic', fill_value='extrapolate')
        amp_aPSs = fdPSs(np.arange(0, 80, 0.1))
        amp_aPSs_neg = fdPSs(np.arange(-4, 80, 0.1))

        # Extract horizontal distance at each depth and corresponding RF amplitude
        # define function of Hdist=f(depth) and resample over discrete depth range
        fdPs = interp1d(depths, Hdist_Ps, kind='cubic', fill_value='extrapolate')
        H_dist_tPs=fdPs(np.arange(0, 125, 0.1))
        fdPPs = interp1d(depths, Hdist_PPs, kind='cubic', fill_value='extrapolate')
        H_dist_tPPs=fdPPs(np.arange(0, 125, 0.1))
        fdPSs = interp1d(depths, Hdist_PSs, kind='cubic', fill_value='extrapolate')
        H_dist_tPSs=fdPSs(np.arange(0, 80, 0.1))

        """ Work out lat,lon for given H dist. gc_forward(lat1,lon1,azim,dist):
        # Given a starting point, initial bearing, and a distance,
        # returns the end point on a great circle path """
        BAZ=stream[0].stats.baz
        stat_lat = float(stream[0].stats.stla)
        stat_lon = float(stream[0].stats.stlo)
        latlon_tPs=[]
        for curdist in H_dist_tPs:
            destination = great_circle(miles=curdist).destination((stat_lat, stat_lon), BAZ)
            latlon_tPs.append((destination.latitude, destination.longitude))

        latlon_tPPs=[]
        for curdist in H_dist_tPPs:
            destination = great_circle(miles=curdist).destination((stat_lat, stat_lon), BAZ)
            latlon_tPPs.append((destination.latitude, destination.longitude))

        latlon_tPSs=[]
        for curdist in H_dist_tPSs:
            destination = great_circle(miles=curdist).destination((stat_lat, stat_lon), BAZ)
            latlon_tPSs.append((destination.latitude, destination.longitude))



        
        # Add solutions to metadata directly to stream object
        if 'conversions' not in stream[0].stats:
            stream[0].stats['conversions'] = {}

        stream[0].stats['conversions'][filter] = {
            'depth_Ps_neg': np.arange(-4, 125, 0.1),
            'amp_Ps_neg': amp_aPs_neg,
            'depth_PPs_neg': np.arange(-4, 80, 0.1),
            'amp_PPs_neg': amp_aPPs_neg,
            'depth_PSs_neg': np.arange(-4, 80, 0.1),
            'amp_PSs_neg': amp_aPSs_neg,
            'depth_Ps': np.arange(0, 125, 0.1),
            'amp_Ps': amp_aPs,
            'Hdist_Ps': H_dist_tPs,
            'latlon_Ps': latlon_tPs,
            'depth_PPs': np.arange(0, 80, 0.1),
            'amp_PPs': amp_aPPs,
            'Hdist_PPs': H_dist_tPPs,
            'latlon_PPs': latlon_tPPs,
            'depth_PSs': np.arange(0, 80, 0.1),
            'amp_PSs': amp_aPSs,
            'Hdist_PSs': H_dist_tPSs,
            'latlon_PSs': latlon_tPSs
        }

        # Save results to PICKLE file
        stream.write(savepath, format= 'PICKLE')

    else:
        print('already done')

def process_all_rf_files(path, model_dir, depths_neg, depths, filter):
    """
    Process all receiver function files in a directory.
    
    Parameters:
        path (str): Path to the directory containing RF files.
        model_dir (str): Path to the velocity model files.
        depths_neg (array): Array of depths for negative depths.
        filter (str): A filter for RF events.

    Return:
        void
    """
    rf_files= list()
    

    for stations in path:
        for bins in glob.glob(stations + '/*[!.png]'):
            for file in glob.glob(bins + '/*[!.png]'):
                rf_files.append(file)


    """ Use multiprocessing for parallel execution (uses available CPUs to process RFs in parallel) """
    with parallel_config(backend= 'loky', n_jobs=4, verbose=5):
        Parallel()(delayed(process_rf_file)(file, model_dir, depths_neg, depths, filter) for file in rf_files)


    """ Uncomment the following for sequential processing of each RF and
    comment out the above multiprocessing commands """
    #for rf_file in rf_files:
    #    if 'rf' in rf_file:
    #        process_rf_file(rf_file, model_dir, depths_neg, filter)


filter= '5'
path = ['/raid2/cg812/2015_RFs/Gauss_5.0/VIFE', '/raid2/cg812/2015_RFs/Gauss_5.0/LOGR', '/raid2/cg812/2015_RFs/Gauss_5.0/NAUG', '/raid2/cg812/2015_RFs/Gauss_5.0/DREK', '/raid2/cg812/2015_RFs/Gauss_5.0/HOTT', '/raid2/cg812/2015_RFs/Gauss_5.0/DYSA', '/raid2/cg812/2015_RFs/Gauss_5.0/STOR', '/raid2/cg812/2015_RFs/Gauss_5.0/VIKS'] #Path to directory containing the receiver functions with certain gaussian widths.
model_dir = '/raid2/cg812/3D_velocity_models'   # Path to directory containing individual velocity model files.
depths_neg = np.arange(-40, 141, 1)     # Specify desired depth range for processing (use negative depths to account for negative arrivals).
depths = np.arange(0, 141, 1)           # Specify desired depth range for processing.

""" Sets up a log file for tracking output (faster than including print statements 
in multiprocessing). Specify the destination and name for the saved log file """
logging.basicConfig(
    filename="/raid2/cg812/Migrated_to_depth_3D/rf_processing.log",
    level=logging.INFO,
    format='%(asctime)s %(processName)s %(levelname)s: %(message)s',
    )

process_all_rf_files(path, model_dir, depths_neg, depths, filter)
