import numpy as np

def get_Re(case, cases, Re):
    if not int and len(Re) > 1:
        cur_Re = Re[cases.index(case)]
    else:
        cur_Re = Re[0]
    return cur_Re

def read_velocity_profile(ux):
    ux = ux[:, 2]
    return ux

def compute_u_prime_sq(ux, uu):
    ux_col = ux[:, 2]
    uu_col = uu[:, 2]

    return uu_col - np.square(ux_col)

def compute_u_prime_v_prime(ux, uy, uv):
    ux_col = ux[:, 2]
    uv_col = uv[:, 2]
    uy_col = uy[:, 2]

    ux_uy = np.multiply(ux_col, uy_col)
    return uv_col - ux_uy

def compute_w_prime_sq(uz, ww):
    uz_col = uz[:, 2]
    ww_col = ww[:, 2]

    return ww_col - np.square(uz_col)

def compute_v_prime_sq(uy, vv):
    uy_col = uy[:, 2]
    vv_col = vv[:, 2]

    return vv_col - np.square(uy_col)

def norm_turb_stat_wrt_u_tau_sq(ux_data, turb_stat, Re_bulk):

    Re_bulk = int(Re_bulk)
    du = ux_data[0, 2] - ux_data[1, 2]
    dy = ux_data[0, 1] - ux_data[1, 1]
    dudy = du/dy
    tau_w = dudy/Re_bulk
    u_tau_sq = abs(tau_w)
    u_tau = np.sqrt(u_tau_sq)
    Re_tau = u_tau * Re_bulk
    print(f'u_tau = {u_tau}, tau_w = {tau_w}, Re_tau = {Re_tau}')
    print('-'*100)

    turb_stat = np.asarray(turb_stat)
    return np.divide(turb_stat, u_tau_sq)

def norm_ux_velocity_wrt_u_tau(ux_data, Re_bulk):
    
    Re_bulk = int(Re_bulk)
    du = ux_data[0, 2] - ux_data[1, 2]
    dy = ux_data[0, 1] - ux_data[1, 1]
    dudy = du/dy
    u_tau = np.sqrt(abs(dudy/Re_bulk))
    ux_velocity = ux_data[:, 2]
    ux_velocity = np.asarray(ux_velocity)
    return np.divide(ux_velocity, u_tau)

def norm_y_to_y_plus(y, ux_data, Re_bulk):

    Re_bulk = int(Re_bulk)
    du = ux_data[0, 2] - ux_data[1, 2]
    dy = ux_data[0, 1] - ux_data[1, 1]
    dudy = du/dy
    u_tau = np.sqrt(abs(dudy/Re_bulk))
    y_plus = y * u_tau * Re_bulk
    return y_plus

def symmetric_average(arr):
    n = len(arr)
    half = n // 2
    # If even length: average symmetric pairs
    if n % 2 == 0:
        return (arr[:half] + arr[::-1][:half]) / 2
    else:
        # Odd length: do same, keep middle element as is
        symmetric_avg = (arr[:half] + arr[::-1][:half]) / 2
        middle = np.array([(arr[half])])  # keep center value
        return np.concatenate((symmetric_avg, middle))

def window_average(data_t1, data_t2, t1, t2, stat_start_timestep):

    stat_t2 = t2 - stat_start_timestep
    stat_t1 = t1 - stat_start_timestep
    t_diff = stat_t2 - stat_t1

    if t_diff == 0:
        return data_t2  # If no difference, return the second timestep data directly
    else:
        return (stat_t2 * data_t2 - stat_t1 * data_t1) / t_diff

def analytical_laminar_mhd_prof(case, Re_bulk, Re_tau): # U. Müller, L. Bühler, Analytical Solutions for MHD Channel Flow, 2001.
        u_tau = Re_tau / Re_bulk
        y = np.linspace(0, 1, 100) * Re_tau
        prof = (((Re_tau * u_tau)/(case * np.tanh(case)))*((1 - np.cosh(case * (1 - y)))/np.cosh(case)) + 1.225)
        return prof

def compute_vel_fluctuation(inst_data, time_avg_data):

    vel_fluc = inst_data - time_avg_data

    return vel_fluc

def compute_vorticity_omega_x(uy, uz, y, z): # test vorticity functions

    duzdy = np.gradient(uz, y)
    duydz = np.gradient(uy, z)
    omega_x = duzdy - duydz

    return omega_x

def compute_vorticity_omega_y(ux, uz, x, z):

    duxdz = np.gradient(ux, z)
    duzdx = np.gradient(uz, x)
    omega_y = duxdz- duzdx

    return omega_y

def compute_vorticity_omega_z(uy, ux, x, y):

    duydx = np.gradient(uy, x)
    duxdy = np.gradient(ux, y)
    omega_z = duydx - duxdy

    return omega_z

def get_col(case, cases, colours):
    if len(cases) > 1:
        colour = colours[cases.index(case)]
    else:
        colour = colours[0]
    return colour

# lamda2 and q criterion next