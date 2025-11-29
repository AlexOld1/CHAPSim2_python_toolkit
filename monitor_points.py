import numpy as np
import matplotlib.pyplot as plt
import utils as ut

plt.rcParams['agg.path.chunksize'] = 10000 # Configure matplotlib for better performance with large datasets
plt.rcParams['path.simplify_threshold'] = 1.0

# ====================================================================================================================================================
# Input parameters
# ====================================================================================================================================================

#path = '/home/alex/sim_results/mhd_heated_channel_validation/Ha_16/3_monitor/'
#path = '/home/alex/sim_results/mesh_con_Ha_30/30_pts/3_monitor/3_monitor/'
#path = '/home/alex/sim_results/mhd_channel_validation/CPG/Ha_4/3_monitor/'
path = '/home/alex/sim_results/elev_modes/isothermal_base_cases/3_monitor/'
pt_files = ['domain1_monitor_pt1_flow.dat','domain1_monitor_pt3_flow.dat','domain1_monitor_pt5_flow.dat',
         'domain1_monitor_pt2_flow.dat','domain1_monitor_pt4_flow.dat']
blk_files = ['domain1_monitor_bulk_history.log', 'domain1_monitor_change_history.log']

plt_pts = True
plt_bulk = True
save_to_path = True

clean_file = False
sample_factor = 10  # Plot every nth point to reduce data density
thermo_on = False

# ====================================================================================================================================================
 
if plt_pts:
    for file in pt_files:
        if clean_file:
            print(f'Cleaning dataset {file}...')
            expected_columns = 7 if thermo_on else 6
            data = ut.clean_dat_file(path+file, f'{file.replace('domain1_monitor_','').replace('.dat','_clean')}', expected_columns)
        else:
            data = np.loadtxt(path+file,skiprows=3)
            #print(f'Loaded monitor_point_plots/{file.replace('.dat','_clean')} for plotting.')

        data = data[::sample_factor] # sample data for plotting
        print(f'Plotting {len(data)} points for {file}...')

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
        plt.savefig(f'monitor_point_plots/{file.replace('domain1_monitor_','').replace('.dat','_plot')}', dpi=300)
        if save_to_path:
            plt.savefig(f'{path}{file.replace('domain1_monitor_','').replace('.dat','_plot')}', dpi=300)
        print(f'Saved plot for {file}')

if plt_bulk:
    for file in blk_files:
        if clean_file:
            print(f'Cleaning dataset {file}...')
            expected_columns = 6 if thermo_on else 3
            blk_data = ut.clean_dat_file(path+file, f'{file.replace('domain1_monitor_','').replace('.log','_clean')}', expected_columns)
        else:
            blk_data = np.loadtxt(path+file, skiprows=2)
        
        blk_data = blk_data[::sample_factor] # sample data for plotting

        if file == 'domain1_monitor_bulk_history.log':
            time = blk_data[:,0]
            MKE = blk_data[:,1]
            qx = blk_data[:,2]
            if thermo_on:
                gx = blk_data[:,3]
                T = blk_data[:,4]
                h = blk_data[:,5]

            plt.figure(figsize=(10,6))
            plt.plot(time, MKE, label='Mean Kinetic Energy', linewidth=0.5)
            plt.plot(time, qx, label='Bulk Velocity', linewidth=0.5)
            if thermo_on:
                plt.plot(time, T, label='Bulk Temperature', linewidth=0.5)
                plt.plot(time, h, label='Bulk Enthalpy', linewidth=0.5)
                plt.plot(time, gx, label='Density * Bulk Velocity', linewidth=0.5)
            plt.xlabel('Time')
            plt.ylabel('Bulk Flow Variables')
            plt.title('Bulk Quantities')
            plt.legend()
            plt.grid()
            plt.savefig(f'monitor_point_plots/{file.replace('domain1_monitor_','').replace('.log','_plot')}', dpi=300)
            if save_to_path:
                plt.savefig(f'{path}{file.replace('domain1_monitor_','').replace('.log','_plot')}', dpi=300)
            print(f'Saved bulk history plot for {file}')
        
        if file == 'domain1_monitor_change_history.log':
            time = blk_data[:,0]
            mass_cons = blk_data[:,1]
            mass_chng_rt = blk_data[:,4]
            KE_chng_rt = blk_data[:,5]

            plt.figure(figsize=(10,6))
            plt.plot(time, mass_cons, label='Mass Conservation', linewidth=0.5)
            plt.plot(time, mass_chng_rt, label='Mass Change Rate', linewidth=0.5)
            plt.plot(time, KE_chng_rt, label='Kinetic Energy Change Rate', linewidth=0.5)
            plt.xlabel('Time')
            plt.ylabel('Change History Variables')
            plt.title('Change History')
            plt.legend()
            plt.grid()
            plt.savefig(f'monitor_point_plots/{file.replace('domain1_monitor_','').replace('.log','_plot')}', dpi=300)
            if save_to_path:
                plt.savefig(f'{path}{file.replace('domain1_monitor_','').replace('.log','_plot')}', dpi=300)
            print(f'Saved change history plot for {file}')

print('='*100)
if save_to_path:
    print(f'All plots saved in monitor_point_plots/ and {path}.')
else:
    print('All plots saved in monitor_point_plots/.')
print('='*100)
