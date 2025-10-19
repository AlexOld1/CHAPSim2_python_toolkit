# import libraries ------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import os

# import modules --------------------------------------------------------------------------------------------------------------------------------------

import operations as op
import utils as ut
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

    # Print Array Extraction info
    print(f"\nTotal arrays extracted: {len(visu_data)}")                
else:
    print("No arrays were successfully extracted.")