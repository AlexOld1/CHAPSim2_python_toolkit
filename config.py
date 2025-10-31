# Define input cases ----------------------------------------------------------------------------------------------------------------------------------

# format: folder_path/case/1_data/quantity_timestep.dat
folder_path = '/home/alex/sim_results/mhd_channel_validation/CPG/'
cases = ['Ha_4','Ha_6'] # case names must match folder names exactly
timesteps = ['284000','290000','454000','552000']
quantities = ['uu', 'ux', 'uy', 'uv', 'uz', 'ww','vv','pr','T'] # for time & space averaged files

forcing = 'CPG' # 'CMF' or 'CPG'
Re = [2305, 2355] # indexing matches 'cases' if different Re used. Use reference value for CPG.

# Output ----------------------------------------------------------------------------------------------------------------------------------------------

# velocity profiles & first order statistics
ux_velocity_on = True
u_prime_sq_on = True
u_prime_v_prime_on = True
w_prime_sq_on = True
v_prime_sq_on = True
tke_on = True
temp_on = False

# Processing options ----------------------------------------------------------------------------------------------------------------------------------

# normalisation (1D data)
norm_by_u_tau_sq = True
norm_ux_by_u_tau = True
norm_y_to_y_plus = True

# Plotting options ------------------------------------------------------------------------------------------------------------------------------------

half_channel_plot = False
linear_y_scale = True
log_y_scale = False
multi_plot = True
display_fig = True
save_fig = False

# reference data options
ux_velocity_log_ref_on = True
mhd_NK_ref_on = False
mkm180_ch_ref_on = False

#====================================================================================================================================================
