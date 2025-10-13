# Define input cases ----------------------------------------------------------------------------------------------------------------------------------

folder_path = '/home/alex/sim_results/mhd_channel_validation/CPG/' # format: folder_path/case/1_data/quantity_timestep.dat
cases = ['Ha_4'] # case names must match folder names exactly
timesteps = ['214000']
quantities = ['uu', 'ux', 'uy', 'uv', 'uz', 'ww','vv','pr'] # for time & space averaged files

Re = ['2305'] # indexing matches 'cases' if different Re used

# Output ----------------------------------------------------------------------------------------------------------------------------------------------

# velocity profiles & first order statistics
ux_velocity_on = False
u_prime_sq_on = True
u_prime_v_prime_on = True
w_prime_sq_on = True
v_prime_sq_on = True

# Processing options ----------------------------------------------------------------------------------------------------------------------------------

# averaging (1D data)
symmetric_average_on = True
window_average_on = False
window_average_val_lower_bound = 180000 # this doesn't work right now
stat_start_timestep = 180000

# normalisation (1D data)
norm_by_u_tau_sq = True
norm_ux_by_u_tau = True

# Plotting options ------------------------------------------------------------------------------------------------------------------------------------

linear_y_scale = True
log_y_scale = False
set_y_plus_scaling = False
y_plus_scale_value = 153
multi_plot = False
display_fig = True
save_fig = False

# reference data options
ux_velocity_log_ref_on = True
mhd_NK_ref_on = True
mhd_XCompact_ref_on = False
mkm180_ch_ref_on = False

#====================================================================================================================================================