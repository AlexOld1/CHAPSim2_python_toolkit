# CHAPSim2_pp.py
# This script processes computes first order turbulence statistics from time and space averaged data for channel flow simulations.
# Input is currently in the form of .dat files, which are expected to be in a specified filepath structure (see below).

# import libraries ------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import math
import os

# import modules --------------------------------------------------------------------------------------------------------------------------------------

import operations as op
import utils as ut

# =====================================================================================================================================================
from config import *

# Input parameters in development

# 3D visualisation (under construction)
visualisation_on = False

#instant_ux_on = False
#instant_uy_on = False
#instant_uz_on = False
#instant_press_on = False
#instant_phi_on = False

# analytical input -- currently doesn't work
# Analytical_lam_mhd_on = False
# Analytical_lam_Ha = [4.0, 6.0, 8.0]
# Ana_Re_tau = 150

# =====================================================================================================================================================

# Define file paths -----------------------------------------------------------------------------------------------------------------------------------

# Channel flow reference data (isothermal, no mhd, Re_tau = 180) - 
# "DNS of Turbulent Channel Flow up to Re_tau = 590", R. D. Moser, J. Kim & N. N. Mansour, 1999 (Phys. Fluids, vol 11, pg 943-945).
ref_mkm180_means_path = 'Reference_Data/MKM180_profiles/chan180.means'
ref_mkm180_reystress_path = 'Reference_Data/MKM180_profiles/chan180.reystress'

# mhd channel flow reference data (isothermal, transverse magnetic field, Ha = 4, 6, Re_tau = 150) - thtlabs.jp
NK_ref_paths = {
    'ref_NK_Ha_6' : 'Reference_Data/Noguchi&Kasagi_mhd_ref_data/thtlabs_Ha_6_turb.txt',
    'ref_NK_Ha_4' : 'Reference_Data/Noguchi&Kasagi_mhd_ref_data/thtlabs_Ha_4_turb.txt',
    'ref_NK_uv_Ha_6' : 'Reference_Data/Noguchi&Kasagi_mhd_ref_data/thtlabs_Ha_6_uv_rms.txt',
    'ref_NK_uv_Ha_4' : 'Reference_Data/Noguchi&Kasagi_mhd_ref_data/thtlabs_Ha_4_uv_rms.txt',
}

# mhd channel flow reference data - XCompact3D (isothermal, transverse magnetic field, Ha = 4, 6, Re_tau = 150) - 
# "A high-order finite-difference solver for direct numerical simulations of magnetohydrodynamic turbulence", J. Fang, S. Laizet & A. Skillen (Comp. Phys Comms., 2025)
ref_XComp_Ha_6_path = 'Reference_Data/XCompact3D_mhd_validation/u_prime_sq.txt'

# =====================================================================================================================================================

# Load reference data ---------------------------------------------------------------------------------------------------------------------------------

if mkm180_ch_ref_on:
    ref_mkm180_means = np.loadtxt(ref_mkm180_means_path)
    ref_mkm180_reystress = np.loadtxt(ref_mkm180_reystress_path)
    print("mkm180 reference data loaded successfully.")
else:
    print("mkm180 reference is disabled or required data is missing.")

if mhd_NK_ref_on:
    NK_ref_data = {}
    for path in NK_ref_paths:
        NK_ref_data[path] = np.loadtxt(NK_ref_paths[path])
    print("Noguchi & Kasagi MHD channel reference data loaded successfully.")
else:
    print("NK mhd reference is disabled or required data is missing.")

if mhd_XCompact_ref_on:
    ref_XComp_Ha_6 = np.loadtxt(ref_XComp_Ha_6_path,delimiter=',',skiprows=1)
    print("XCompact mhd reference data loaded successfully")
else:
    print("XCompact mhd reference is disabled or required data is missing")

# Load and store case data ----------------------------------------------------------------------------------------------------------------------------

# time & space averaged data (.dat files)

ts_avg_data = {}
ts_avg_data.clear()  # Clear any existing keys

for case in cases:
    for timestep in timesteps:
        for quantity in quantities:
            key = (case, quantity, timestep)
            file_path = ut.data_filepath(folder_path, case, quantity, timestep)
            file_exists = os.path.isfile(file_path)
            print (f"Looking for files in: {file_path}")
            if file_exists:
                data = ut.load_ts_avg_data(file_path)
                if data is not None:
                    ts_avg_data[key] = data
                else:
                    print(f'.dat file is empty for {case}, {timestep}, {quantity}')
            else:
                print(f'No .dat file found for {case}, {timestep}, {quantity}')
print(ts_avg_data.keys)

# 3D visualisation data (.xdmf files)

visu_data = {}
visu_data.clear()
grid_info = {}

if visualisation_on:
    for case in cases:
        for timestep in timesteps:

            print(f"\n{'-'*60}")
            print(f"Processing: {case}, {timestep}")
            print(f"{'-'*60}")

            file_names = ut.visu_file_paths(folder_path, case, timestep)

            existing_files = [file for file in file_names if os.path.isfile(file)]
            missing_files = [file for file in file_names if not os.path.isfile(file)]
            for file in missing_files:
                print(f'No .xdmf file found for {file}')

            if existing_files:
                arrays, grid_info_cur = ut.read_xdmf_extract_numpy_arrays(file_names)

                if arrays:
                    # Store arrays with timestep prefix
                    key_arrays = {f"{key}": value for key, value in arrays.items()}
                    visu_data.update(key_arrays)
                    
                    # Store grid info (should be same for all timesteps)
                    if not grid_info and grid_info_cur:
                        grid_info = grid_info_cur
                    
                    print(f"\nSuccessfully extracted {len(arrays)} arrays from case {case}, timestep {timestep}")
                else:
                    print(f"No arrays extracted from timestep {timestep}")
            else:
                continue

if visu_data and visualisation_on:
    ut.reader_output_summary(visu_data)
        
    # Print grid information
    if grid_info:
        print(f"\n{'='*60}")
        print("GRID INFORMATION")
        print(f"{'='*60}")
        for key, value in grid_info.items():
             print(f"{key}: {value}")

    # PRint Array Extraction info
    print(f"\nTotal arrays extracted: {len(visu_data)}")                
else:
    print("No arrays were successfully extracted.")

# =====================================================================================================================================================

# dictionary for turbulence statistics ----------------------------------------------------------------------------------------------------------------

all_turb_stats = {}
all_turb_stats.clear()  # Clear any existing keys

# Read Velocity Profiles ------------------------------------------------------------------------------------------------------------------------------

ux_velocity = {}

if ux_velocity_on:
    for case in cases:
        for timestep in timesteps:
            key_ux = (case, 'ux', timestep)
            if key_ux in ts_avg_data:
                ux_data = ts_avg_data[key_ux]
                ux_velocity[(case, timestep)] = op.read_velocity_profile(ux_data)
                all_turb_stats['ux_velocity'] = ux_velocity
else:
    print("ux velocity calculation is disabled or required data is missing.")

# Compute First Order Turbulence Statistics -----------------------------------------------------------------------------------------------------------

u_prime_sq = {}

if u_prime_sq_on and 'uu' in quantities and 'ux' in quantities:
    for case in cases:
        for timestep in timesteps:
            key_uu = (case, 'uu', timestep)
            key_ux = (case, 'ux', timestep)
            if key_uu in ts_avg_data and key_ux in ts_avg_data:
                
                uu_data = ts_avg_data[key_uu]
                ux_data = ts_avg_data[key_ux]
                result = op.compute_u_prime_sq(ux_data, uu_data)
                u_prime_sq[(case, timestep)] = result
                all_turb_stats['u_prime_sq'] = u_prime_sq
else:
    print("u'u' calculation is disabled or required data is missing.")

u_prime_v_prime = {}

if u_prime_v_prime_on and 'uv' in quantities and 'ux' in quantities and 'uy' in quantities:
    for case in cases:
        for timestep in timesteps:
            key_uv = (case, 'uv', timestep)
            key_ux = (case, 'ux', timestep)
            key_uy = (case, 'uy', timestep)
            if key_uv in ts_avg_data and key_ux in ts_avg_data and key_uy in ts_avg_data:
                uv_data = ts_avg_data[key_uv]
                ux_data = ts_avg_data[key_ux]
                uy_data = ts_avg_data[key_uy]
                result = op.compute_u_prime_v_prime(ux_data, uy_data, uv_data)
                u_prime_v_prime[(case, timestep)] = result
                all_turb_stats['u_prime_v_prime'] = u_prime_v_prime
else:
    print("u'v' calculation is disabled or required data is missing.")

w_prime_sq = {}

if w_prime_sq_on and 'ww' in quantities and 'uz' in quantities:
    for case in cases:
        for timestep in timesteps:
            key_ww = (case, 'ww', timestep)
            key_uz = (case, 'uz', timestep)
            if key_ww in ts_avg_data and key_uz in ts_avg_data:
                ww_data = ts_avg_data[key_ww]
                uz_data = ts_avg_data[key_uz]
                result = op.compute_w_prime_sq(uz_data, ww_data)
                w_prime_sq[(case, timestep)] = result
                all_turb_stats['w_prime_sq'] = w_prime_sq
else:
    print("w'w' calculation is disabled or required data is missing.")

v_prime_sq = {}

if v_prime_sq_on and 'vv' in quantities and 'uy' in quantities:
    for case in cases:
        for timestep in timesteps:
            key_vv = (case, 'vv', timestep)
            key_uy = (case, 'uy', timestep)
            if key_vv in ts_avg_data and key_uy in ts_avg_data:
                vv_data = ts_avg_data[key_vv]
                uy_data = ts_avg_data[key_uy]
                result = op.compute_v_prime_sq(uy_data, vv_data)
                v_prime_sq[(case, timestep)] = result
                all_turb_stats['v_prime_sq'] = v_prime_sq
else:
    print("v'v' calculation is disabled or required data is missing.")

# Reference data processing ---------------------------------------------------------------------------------------------------------------------------

# mkm180

if mkm180_ch_ref_on:
    ref_mkm180_y = ref_mkm180_means[:, 0]
    ref_mkm180_y_plus = ref_mkm180_means[:, 1]

    mkm180_stats = {
        'ux_velocity' : ref_mkm180_means[:, 2],
        'u_prime_sq' : ref_mkm180_reystress[:, 2],
        'v_prime_sq' : ref_mkm180_reystress[:, 3],
        'w_prime_sq' : ref_mkm180_reystress[:, 4],
        'u_prime_v_prime' : ref_mkm180_reystress[:, 5]
    }

# mhd

if mhd_NK_ref_on:
    ref_y_H4 = NK_ref_data['ref_NK_Ha_4'][:, 1]
    ref_y_H6 = NK_ref_data['ref_NK_Ha_6'][:, 1]
    ref_y_uv_H4 = NK_ref_data['ref_NK_uv_Ha_4'][:, 1]
    ref_y_uv_H6 = NK_ref_data['ref_NK_uv_Ha_6'][:, 1]
    
    NK_H4_ref_stats = {
        'ux_velocity' : NK_ref_data['ref_NK_Ha_4'][:, 2] * 1.02169,
        'u_prime_sq' : np.square(NK_ref_data['ref_NK_Ha_4'][:, 3]),
        'u_prime_v_prime' : -1 * NK_ref_data['ref_NK_uv_Ha_4'][:, 2],
        'v_prime_sq' : np.square(NK_ref_data['ref_NK_Ha_4'][:, 4]),
        'w_prime_sq' : np.square(NK_ref_data['ref_NK_Ha_4'][:, 5])
    }
    
    NK_H6_ref_stats = {
        'ux_velocity' : NK_ref_data['ref_NK_Ha_6'][:, 2],
        'u_prime_sq' : np.square(NK_ref_data['ref_NK_Ha_6'][:, 3]),
        'u_prime_v_prime' : -1 * NK_ref_data['ref_NK_uv_Ha_6'][:, 2],
        'v_prime_sq' : np.square(NK_ref_data['ref_NK_Ha_6'][:, 4]),
        'w_prime_sq' : np.square(NK_ref_data['ref_NK_Ha_6'][:, 5])
    }

if mhd_XCompact_ref_on:
    ref_yplus_uu_H6 = ref_XComp_Ha_6[:, 0]
    xcomp_H6_stats = {
        'u_prime_sq' : ref_XComp_Ha_6[:, 1]
    }

# Analytical Velocity Profiles ---------------------------------------------------------------------------------------------------------------------------------

# not currently working

#Ana_lam_Ha_prof = {}
#Ana_lam_Ha_prof.clear()  # Clear any existing keys

#if Analytical_lam_mhd_on:
#    for case in Analytical_lam_Ha:
#
#        Re_bulk = op.get_Re(case, cases, Re)
#        Ana_lam_Ha_prof[case] = op.analytical_laminar_mhd_prof(case, Re_bulk, Ana_Re_tau)
#        print(f'Calculated analytical laminar profile for case = {case}')
#else:
#    print(f'Analytical MHD laminar profile calculation is disabled or required data is missing.')

# Normalise Quantities with respect to u_tau squared and average symmetrically ------------------------------------------------------------------------

turb_stats_norm = {}
turb_stats_norm.clear()  # Clear any existing keys

for turb_stat, stat_dict in all_turb_stats.items():
    
    temp_dict = {}
    temp_dict.clear()  # Clear any existing keys

    for (case, timestep), values in stat_dict.items():

        # Define key for ux data

        key_ux = (case, 'ux', timestep)
        if key_ux in ts_avg_data:
            ux_data = ts_avg_data[key_ux]
        else:
         print(f'Missing ux data for normalisation calc')

        cur_Re = op.get_Re(case, cases, Re)

        # Normalisation and symmetric averaging

        if norm_by_u_tau_sq:
            normed = op.norm_turb_stat_wrt_u_tau_sq(ux_data, values, cur_Re)
        else:
            normed = values

        if norm_ux_by_u_tau and turb_stat == 'ux_velocity':
            # Normalise ux velocity by u_tau
            normed = op.norm_ux_velocity_wrt_u_tau(ux_data, cur_Re)
            print(f'ux velocity normalised by u_tau for {case}, {timestep}')

        if symmetric_average_on and turb_stat != 'u_prime_v_prime':
            normed_avg = op.symmetric_average(normed)
            temp_dict[(case, timestep)] = normed_avg
            print(f'Symmetric averaged data for {case}, {timestep}')
        else:
            temp_dict[(case, timestep)] = normed[:(len(normed)//2)]
            print(f'First {len(normed)} values normalised for {case}, {timestep}')

        # Window averaging

        if window_average_on:
            key_low_bound = (case, quantity, f'{window_average_val_lower_bound}')
            if key_low_bound in ts_avg_data:
                data_t1 = ts_avg_data[key_low_bound]
                data_t1 = data_t1[:, 2]
                data_t1 = op.symmetric_average(data_t1)
                win_nor_avg = op.window_average(data_t1, normed, int(window_average_val_lower_bound), int(timestep), int(stat_start_timestep))
                temp_dict[(case, timestep)] = win_nor_avg
                print(f'Window averaged data for {case}, {timestep}')
            else:
                temp_dict[(case, timestep)] = normed_avg
                print(f'No data for window averaging for {case}, {timestep}')

        # Store the normalised and averaged data
        turb_stats_norm[turb_stat] = temp_dict

# =====================================================================================================================================================

# 2D Plot ---------------------------------------------------------------------------------------------------------------------------------------------

colours_1 = {
    'ux_velocity' : 'b',
    'u_prime_sq' : 'r',
    'u_prime_v_prime' : 'g',
    'w_prime_sq' : 'c',
    'v_prime_sq' : 'm',
}

colours_2 = {
    'ux_velocity' : 'r',
    'u_prime_sq' : 'orange',
    'u_prime_v_prime' : 'lime',
    'w_prime_sq' : 'b',
    'v_prime_sq' : 'purple',
}

colours_3 = {
    'ux_velocity' : 'indigo',
    'u_prime_sq' : 'maroon',
    'u_prime_v_prime' : 'olive',
    'w_prime_sq' : 'navy',
    'v_prime_sq' : 'deeppink',
}

colours_4 = {
    'ux_velocity' : 'magenta',
    'u_prime_sq' : 'peru',
    'u_prime_v_prime' : 'lightgreen',
    'w_prime_sq' : 'teal',
    'v_prime_sq' : 'indigo',
}

colours_blck = {
    'ux_velocity' : 'black',
    'u_prime_sq' : 'black',
    'u_prime_v_prime' : 'black',
    'w_prime_sq' : 'black',
    'v_prime_sq' : 'black',
}

colours = ( colours_blck,colours_1, colours_2, colours_3, colours_4)

stat_labels = {
    "ux_velocity" : "Streamwise Velocity",
    "u_prime_sq" : "<u'u'>",
    "u_prime_v_prime" : "<u'v'>",
    "v_prime_sq" : "<v'v'>",
    "w_prime_sq" : "<w'w'>",
}

if len(turb_stats_norm) == 1 or multi_plot == False: # creates a single plot for all turbulence statistics
    
    plt.figure(figsize=(10, 6))
    
    for stat_name, stat_dict in turb_stats_norm.items():
        for (case, timestep), values in stat_dict.items():

            # Get y coordinates from ux data

            key_ux = (case, 'ux', timestep)
            if key_ux in ts_avg_data:
                ux_data = ts_avg_data[key_ux]
                y = (ux_data[:len(ux_data)//2, 1] + 1)
                cur_Re = op.get_Re(case, cases, Re)
                y_plus = op.norm_y_to_y_plus(y, ux_data, cur_Re)

                # Plot current data
                label = f'{stat_labels[stat_name]}, {case.replace('_', ' = ')}'

                # Choose color based on case
                if len(cases) <= len(colours):
                    colour_set = op.get_col(case, cases, colours)
                    line_colour = colour_set[stat_name]
                else:
                    line_colour = list(np.random.choice(range(256), size=3))  # Default colour

                if linear_y_scale:
                    plt.plot(y_plus, values, label=label, linestyle='-', marker='', color=line_colour)

                    # plot reference data
                    if mhd_NK_ref_on and stat_name in NK_H4_ref_stats:
                        if case == 'Ha_4':
                            plt.plot(ref_y_H4, NK_H4_ref_stats[stat_name], linestyle='', 
                                   label=f'Ha = 4, Noguchi & Kasagi',
                                   marker='o', color=colours_1[stat_name], markevery=2)
                        elif case == 'Ha_6':
                            plt.plot(ref_y_H6, NK_H6_ref_stats[stat_name], linestyle='', 
                                   label=f'Ha = 6, Noguchi & Kasagi',
                                   marker='o', color=colours_2[stat_name], markevery=2)
                        
                    if mhd_XCompact_ref_on:
                        plt.okt(ref_yplus_uu_H6, xcomp_H6_stats[stat_name], linestyle='', label='XCompact3D, Ha = 6', marker='s')
                    
                    if mkm180_ch_ref_on:
                        plt.plot(ref_mkm180_y_plus, mkm180_stats[stat_name], linewidth=2, label=f'{stat_labels[stat_name]} MKM180', marker='o')
                
                elif log_y_scale:
                    
                    plt.semilogx(y_plus, values, label=label, linestyle='-', marker='', color=line_colour)

                    # plot reference data
                    if mhd_NK_ref_on and stat_name in NK_H4_ref_stats:
                        if case == 'Ha_4':
                            plt.semilogx(ref_y_H4, NK_H4_ref_stats[stat_name], linestyle='', 
                                   label=f'Ha = 4, Noguchi & Kasagi',
                                   marker='o', color=colours_1[stat_name], markevery=2)
                        elif case == 'Ha_6':
                            plt.semilogx(ref_y_H6, NK_H6_ref_stats[stat_name], linestyle='', 
                                   label=f'Ha = 6, Noguchi & Kasagi',
                                   marker='o', color=colours_2[stat_name], markevery=2)
                        
                    if mhd_XCompact_ref_on:
                        plt.semilogx(ref_yplus_uu_H6, xcomp_H6_stats[stat_name], linestyle='', label='XCompact3D, Ha = 6', marker='s')
                        
                    if mkm180_ch_ref_on:
                        plt.semilogx(ref_mkm180_y_plus, mkm180_stats[stat_name], linewidth=2, label=f'{stat_labels[stat_name]} MKM180', marker='o')

                    if stat_name == 'ux_velocity' and ux_velocity_log_ref_on:
                        plt.semilogx(y_plus[:15], y_plus[:15], '--', linewidth=1, label='$u^+ = y^+$', color='black', alpha=0.5)
                        u_plus_ref = 2.5 * np.log(y_plus) + 5.5
                        plt.plot(y_plus, u_plus_ref, '--', linewidth=1, label='$u^+ = 2.5ln(y^+) + 5.5$', color='black', alpha=0.5)
                        
                else:
                    print('Plotting input incorrectly defined')
            else:
                print(f'Missing ux data for plotting')

    plt.xlabel('$y^+$')

    if len(all_turb_stats) == 1:
        plt.ylabel(f"Normalised {stat_name.replace('_', ' ')}")
    elif len(all_turb_stats) > 1:
        plt.ylabel("Normalised Reynolds Stresses")
    plt.legend()
    plt.grid(True)

elif len(turb_stats_norm) > 1 and multi_plot == True: # creates a different plot for each turbulence statistic

    n_stats = len(turb_stats_norm)
    ncols = math.ceil(math.sqrt(n_stats))
    nrows = math.ceil(n_stats / ncols)

    fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(15, 10),
            constrained_layout=True
        )
    
    # Ensure axs is always a 2D array for consistent indexing
    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    elif nrows == 1 or ncols == 1:
        axs = axs.reshape(nrows, ncols)

    # Plot each statistic on its own subplot
    for i, (stat_name, stat_dict) in enumerate(turb_stats_norm.items()):
        row = i // ncols
        col = i % ncols
        ax = axs[row, col]
        
        for (case, timestep), values in stat_dict.items():

            # Get y coordinates from ux data
            key_ux = (case, 'ux', timestep)
            if key_ux in ts_avg_data:
                ux_data = ts_avg_data[key_ux]
                y = (ux_data[:len(ux_data)//2, 1] + 1)
                
                if set_y_plus_scaling:
                    y_plus = y * y_plus_scale_value
                else:
                    cur_Re = op.get_Re(case, cases, Re)
                    y_plus = op.norm_y_to_y_plus(y, ux_data, cur_Re)

                # Plot current data
                label = f'{case.replace("_", " = ")}'
                
                # Choose color based on case
                if len(cases) <= len(colours):
                    colour_set = op.get_col(case, cases, colours)
                    line_colour = colour_set[stat_name]
                else:
                    line_colour = list(np.random.choice(range(256), size=3))  # Default colour

                if linear_y_scale:
                    ax.plot(y_plus, values, label=label, linestyle='-', marker='', color=line_colour)

                    # plot reference data
                    if mhd_NK_ref_on and stat_name in NK_H4_ref_stats:
                        if case == 'Ha_4':
                            ax.plot(ref_y_H4, NK_H4_ref_stats[stat_name], linestyle='', 
                                   label=f'Ha = 4, Noguchi & Kasagi',
                                   marker='o', color=colours_1[stat_name], markevery=2)
                        elif case == 'Ha_6':
                            ax.plot(ref_y_H6, NK_H6_ref_stats[stat_name], linestyle='', 
                                   label=f'Ha = 6, Noguchi & Kasagi',
                                   marker='o', color=colours_2[stat_name], markevery=2)
                        
                    if mhd_XCompact_ref_on and stat_name in xcomp_H6_stats:
                        ax.plot(ref_yplus_uu_H6, xcomp_H6_stats[stat_name], linestyle='', 
                               label='XCompact3D, Ha = 6', marker='s')
                    
                    if mkm180_ch_ref_on and stat_name in mkm180_stats:
                        ax.plot(ref_mkm180_y_plus, mkm180_stats[stat_name], linewidth=2, 
                               label=f'MKM180', marker='o')

                elif log_y_scale:
                    
                    ax.semilogx(y_plus, values, label=label, linestyle='-', marker='', color=line_colour)

                    if mhd_NK_ref_on and stat_name in NK_H4_ref_stats:
                        if case == 'Ha_4':
                            ax.semilogx(ref_y_H4, NK_H4_ref_stats[stat_name], linestyle='', 
                                       label=f'Ha = 4, Noguchi & Kasagi',
                                       marker='o', color=colours_1[stat_name], markevery=2)
                        elif case == 'Ha_6':
                            ax.semilogx(ref_y_H6, NK_H6_ref_stats[stat_name], linestyle='', 
                                       label=f'Ha = 6, Noguchi & Kasagi',
                                       marker='o', color=colours_2[stat_name], markevery=2)
                        
                    if mhd_XCompact_ref_on and stat_name in xcomp_H6_stats:
                        ax.semilogx(ref_yplus_uu_H6, xcomp_H6_stats[stat_name], linestyle='', 
                                   label='XCompact3D, Ha = 6', marker='s')
                        
                    if mkm180_ch_ref_on and stat_name in mkm180_stats:
                        ax.semilogx(ref_mkm180_y_plus, mkm180_stats[stat_name], linewidth=2, 
                                   label=f'MKM180', marker='o')

                    if stat_name == 'ux_velocity' and ux_velocity_log_ref_on:
                        ax.semilogx(y_plus[:15], y_plus[:15], '--', linewidth=1, 
                                   label='$u^+ = y^+$', color='black', alpha=0.5)
                        u_plus_ref = 2.5 * np.log(y_plus) + 5.5
                        ax.plot(y_plus, u_plus_ref, '--', linewidth=1, 
                               label='$u^+ = 2.5ln(y^+) + 5.5$', color='black', alpha=0.5)
                else:
                    print('Plotting input incorrectly defined')
            else:
                print(f'Missing ux data for plotting')

        # Set subplot properties
        ax.set_title(f'{stat_labels[stat_name]}')
        ax.set_ylabel(f"Normalised {stat_name.replace('_', ' ')}")
        ax.grid(True)
        ax.legend(fontsize='small')
        
        # Add xlabel to bottom row subplots
        if row == nrows - 1:
            ax.set_xlabel('$y^+$')

    # Hide unused subplots
    for i in range(n_stats, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axs[row, col].set_visible(False)

if display_fig or save_fig:

    current_fig = plt.gcf()
    
    if display_fig and not save_fig:
        print('Displaying figure...')
        plt.show()

    elif save_fig and not display_fig:
        print('Saving figure as plot.png...')
        current_fig.savefig('plot.png',
           dpi=300,
           bbox_inches='tight',
           pad_inches=0.1,
           facecolor='white',
           edgecolor='none',
           transparent=True,
           orientation='landscape')
    
elif display_fig and save_fig:
    current_fig = plt.gcf()
    print('Displaying and saving figure as plot.png...')
    current_fig.savefig('plot.png',
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.1,
        facecolor='white',
        edgecolor='none',
        transparent=True,
        orientation='landscape') 
    plt.show
else:
    print('No 2D output option selected')