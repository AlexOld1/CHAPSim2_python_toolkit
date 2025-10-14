import numpy as np
import matplotlib.pyplot as plt

# ====================================================================================================================================================
# Input parameters
# ====================================================================================================================================================

path = '/home/alex/sim_results/mesh_con_Ha_30/30_pts/3_monitor/3_monitor/'
#path = '/home/alex/sim_results/mhd_channel_validation/CPG/Ha_6/3_monitor/'
files = ['domain1_monitor_pt1_flow.dat','domain1_monitor_pt3_flow.dat','domain1_monitor_pt5_flow.dat',
         'domain1_monitor_pt2_flow.dat','domain1_monitor_pt4_flow.dat']

# ====================================================================================================================================================

def clean_dat_file(input_file, output_file, expected_cols=6):
    clean_data = []
    bad_lines = []
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                values = [float(x) for x in line.split()]
                if len(values) == expected_cols:
                    clean_data.append(values)
                else:
                    bad_lines.append((line_num, len(values), line.strip()))
                    
            except ValueError as e:
                bad_lines.append((line_num, 'ERROR', line.strip()))
      
    if bad_lines:
        print(f"Found {len(bad_lines)} problematic lines:")
        for line_num, cols, content in bad_lines:
            print(f"  Line {line_num} ({cols} cols): {content[:80]}...")
    
    np.savetxt(f'monitor_point_plots/{output_file}', clean_data, fmt='%.5E')  # Save clean data
    print(f"\nSaved {len(clean_data)} clean lines to {output_file}")
    
    return np.array(clean_data)

for file in files:
    print(f'Cleaning dataset {file}...')
    data = clean_dat_file(path+file, f'{file.replace('.dat','_clean')}', expected_cols=6)
    time = data[:,0]
    u = data[:,1]
    v = data[:,2]
    w = data[:,3]
    p = data[:,4]

    plt.figure(figsize=(10,6))
    plt.plot(time, u, label='u-velocity')
    plt.plot(time, v, label='v-velocity')
    plt.plot(time, w, label='w-velocity')
    plt.plot(time, p, label='pressure')
    plt.xlabel('Time')
    plt.ylabel('Flow Variables')
    plt.title(f'{file}')
    plt.legend()
    plt.grid()
    plt.savefig(f'monitor_point_plots/{file.replace('.dat','_plot')}', dpi=1000)
