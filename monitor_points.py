import numpy as np
import matplotlib.pyplot as plt
import utils as ut

plt.rcParams['agg.path.chunksize'] = 10000 # Configure matplotlib for better performance with large datasets
plt.rcParams['path.simplify_threshold'] = 1.0

# ====================================================================================================================================================
# Input parameters
# ====================================================================================================================================================

path = '/home/alex/sim_results/mhd_heated_channel_validation/Ha_16/3_monitor/'
#path = '/home/alex/sim_results/mesh_con_Ha_30/30_pts/3_monitor/3_monitor/'
#path = '/home/alex/sim_results/mhd_channel_validation/CPG/Ha_6/3_monitor/'
files = ['domain1_monitor_pt1_flow.dat','domain1_monitor_pt3_flow.dat','domain1_monitor_pt5_flow.dat',
         'domain1_monitor_pt2_flow.dat','domain1_monitor_pt4_flow.dat']
clean_file = False
sample_factor = 100  # Plot every nth point to reduce data density
thermo_on = True

# ====================================================================================================================================================
 
for file in files:
    if clean_file:
        print(f'Cleaning dataset {file}...')
        expected_columns = 7 if thermo_on else 6
        data = ut.clean_dat_file(path+file, f'{file.replace('.dat','_clean')}', expected_columns)
    else:
        data = np.loadtxt(f'monitor_point_plots/{file.replace('.dat','_clean')}', skiprows=3)
        print(f'Loaded monitor_point_plots/{file.replace('.dat','_clean')} for plotting.')

    data = data[::sample_factor] # sample data for plotting
    print(f'Plotting {len(data)} points')

    time = data[:,0]
    u = data[:,1]
    v = data[:,2]
    w = data[:,3]
    p = data[:,4]
    phi = data[:,5]
    if thermo_on:
        T = data[:,6]

    plt.figure(figsize=(10,6))
    plt.plot(time, u, label='u-velocity', linewidth=0.5)
    plt.plot(time, v, label='v-velocity', linewidth=0.5)
    plt.plot(time, w, label='w-velocity', linewidth=0.5)
    plt.plot(time, p, label='pressure', linewidth=0.5)
    plt.plot(time, phi, label='press. corr.', linewidth=0.5)
    if thermo_on:
        plt.plot(time, T, label='temperature', linewidth=0.5)
    plt.xlabel('Time')
    plt.ylabel('Flow Variables')
    plt.title(f'{file}')
    plt.legend()
    plt.grid()
    plt.savefig(f'monitor_point_plots/{file.replace('.dat','_plot')}', dpi=300)

print('='*80)
print('All plots saved in monitor_point_plots/.')
print('='*80)
