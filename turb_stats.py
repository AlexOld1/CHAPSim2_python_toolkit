# turb_stats.py
# This script processes computes first order turbulence statistics from time and space averaged data for channel flow simulations.

# import libraries ------------------------------------------------------------------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import os

# import modules --------------------------------------------------------------------------------------------------------------------------------------
import operations as op
import utils as ut

# =====================================================================================================================================================
# CONFIGURATION CLASSES
# =====================================================================================================================================================

@dataclass
class Config:
    """Configuration wrapper for all settings"""
    # File paths and cases
    folder_path: str
    cases: List[str]
    timesteps: List[str]
    quantities: List[str]
    forcing: str
    Re: List[str]

    # Output options
    ux_velocity_on: bool
    u_prime_sq_on: bool
    u_prime_v_prime_on: bool
    w_prime_sq_on: bool
    v_prime_sq_on: bool
    tke_on: bool
    temp_on: bool

    # Processing options
    norm_by_u_tau_sq: bool
    norm_ux_by_u_tau: bool
    norm_y_to_y_plus: bool

    # Plotting options
    half_channel_plot: bool
    linear_y_scale: bool
    log_y_scale: bool
    multi_plot: bool
    display_fig: bool
    save_fig: bool

    # Reference data options
    ux_velocity_log_ref_on: bool
    mhd_NK_ref_on: bool
    mkm180_ch_ref_on: bool

    @classmethod
    def from_module(cls, config_module):
        """Create Config from imported config module"""
        return cls(
            folder_path=config_module.folder_path,
            cases=config_module.cases,
            timesteps=config_module.timesteps,
            quantities=config_module.quantities,
            forcing=config_module.forcing,
            Re=config_module.Re,
            ux_velocity_on=config_module.ux_velocity_on,
            u_prime_sq_on=config_module.u_prime_sq_on,
            u_prime_v_prime_on=config_module.u_prime_v_prime_on,
            w_prime_sq_on=config_module.w_prime_sq_on,
            v_prime_sq_on=config_module.v_prime_sq_on,
            tke_on=config_module.tke_on,
            temp_on=config_module.temp_on,
            norm_by_u_tau_sq=config_module.norm_by_u_tau_sq,
            norm_ux_by_u_tau=config_module.norm_ux_by_u_tau,
            norm_y_to_y_plus=config_module.norm_y_to_y_plus,
            half_channel_plot=config_module.half_channel_plot,
            linear_y_scale=config_module.linear_y_scale,
            log_y_scale=config_module.log_y_scale,
            multi_plot=config_module.multi_plot,
            display_fig=config_module.display_fig,
            save_fig=config_module.save_fig,
            ux_velocity_log_ref_on=config_module.ux_velocity_log_ref_on,
            mhd_NK_ref_on=config_module.mhd_NK_ref_on,
            mkm180_ch_ref_on=config_module.mkm180_ch_ref_on,
        )


@dataclass
class PlotConfig:
    """Configuration for plotting aesthetics"""

    colors_1: Dict[str, str] = None
    colors_2: Dict[str, str] = None
    colors_3: Dict[str, str] = None
    colors_4: Dict[str, str] = None
    colors_blck: Dict[str, str] = None
    stat_labels: Dict[str, str] = None

    def __post_init__(self):
        if self.colors_1 is None:
            self.colors_1 = {
                'ux_velocity': 'b',
                'u_prime_sq': 'r',
                'u_prime_v_prime': 'g',
                'w_prime_sq': 'c',
                'v_prime_sq': 'm',
            }

        if self.colors_2 is None:
            self.colors_2 = {
                'ux_velocity': 'r',
                'u_prime_sq': 'orange',
                'u_prime_v_prime': 'lime',
                'w_prime_sq': 'b',
                'v_prime_sq': 'purple',
            }

        if self.colors_3 is None:
            self.colors_3 = {
                'ux_velocity': 'indigo',
                'u_prime_sq': 'maroon',
                'u_prime_v_prime': 'olive',
                'w_prime_sq': 'navy',
                'v_prime_sq': 'deeppink',
            }

        if self.colors_4 is None:
            self.colors_4 = {
                'ux_velocity': 'magenta',
                'u_prime_sq': 'peru',
                'u_prime_v_prime': 'lightgreen',
                'w_prime_sq': 'teal',
                'v_prime_sq': 'indigo',
            }

        if self.colors_blck is None:
            self.colors_blck = {
                'ux_velocity': 'black',
                'u_prime_sq': 'black',
                'u_prime_v_prime': 'black',
                'w_prime_sq': 'black',
                'v_prime_sq': 'black',
            }

        if self.stat_labels is None:
            self.stat_labels = {
                "ux_velocity": "Streamwise Velocity",
                "u_prime_sq": "<u'u'>",
                "u_prime_v_prime": "<u'v'>",
                "v_prime_sq": "<v'v'>",
                "w_prime_sq": "<w'w'>",
            }

    @property
    def colours(self):
        """Returns tuple of all color schemes"""
        return (self.colors_1, self.colors_2, self.colors_3, self.colors_4, self.colors_blck)
    def colours_ref(self):
        """Returns tuple of all color schemes with black first for plotting a reference"""
        return (self.colors_blck, self.colors_1, self.colors_2, self.colors_3, self.colors_4)


# =====================================================================================================================================================
# DATA MANAGEMENT CLASSES
# =====================================================================================================================================================

class TurbulenceData:
    """Manages time-space averaged data loading and access"""

    def __init__(self, folder_path: str, cases: List[str], timesteps: List[str], quantities: List[str]):
        self.folder_path = folder_path
        self.cases = cases
        self.timesteps = timesteps
        self.quantities = quantities
        self.data: Dict[Tuple[str, str, str], np.ndarray] = {}

    def load_all(self) -> None:
        """Load all time-space averaged data files"""
        for case in self.cases:
            for timestep in self.timesteps:
                for quantity in self.quantities:
                    self._load_single(case, quantity, timestep)

    def _load_single(self, case: str, quantity: str, timestep: str) -> None:
        """Load a single data file"""
        key = (case, quantity, timestep)
        file_path = ut.data_filepath(self.folder_path, case, quantity, timestep)

        print(f"Looking for files in: {file_path}")

        if os.path.isfile(file_path):
            data = ut.load_ts_avg_data(file_path)
            if data is not None:
                self.data[key] = data
            else:
                print(f'.dat file is empty for {case}, {timestep}, {quantity}')
        else:
            print(f'No .dat file found for {case}, {timestep}, {quantity}')

    def get(self, case: str, quantity: str, timestep: str) -> Optional[np.ndarray]:
        """Get specific data array"""
        return self.data.get((case, quantity, timestep))

    def has(self, case: str, quantity: str, timestep: str) -> bool:
        """Check if data exists"""
        return (case, quantity, timestep) in self.data

    def keys(self):
        """Return all data keys"""
        return self.data.keys()


class ReferenceData:
    """Manages all reference datasets"""

    # Reference data file paths
    REF_MKM180_MEANS_PATH = 'Reference_Data/MKM180_profiles/chan180.means'
    REF_MKM180_REYSTRESS_PATH = 'Reference_Data/MKM180_profiles/chan180.reystress'

    NK_REF_PATHS = {
        'ref_NK_Ha_6': 'Reference_Data/Noguchi&Kasagi_mhd_ref_data/thtlabs_Ha_6_turb.txt',
        'ref_NK_Ha_4': 'Reference_Data/Noguchi&Kasagi_mhd_ref_data/thtlabs_Ha_4_turb.txt',
        'ref_NK_uv_Ha_6': 'Reference_Data/Noguchi&Kasagi_mhd_ref_data/thtlabs_Ha_6_uv_rms.txt',
        'ref_NK_uv_Ha_4': 'Reference_Data/Noguchi&Kasagi_mhd_ref_data/thtlabs_Ha_4_uv_rms.txt',
    }

    #REF_XCOMP_HA_6_PATH = 'Reference_Data/XCompact3D_mhd_validation/u_prime_sq.txt'

    def __init__(self, config: Config):
        self.config = config
        self.mkm180_stats: Optional[Dict[str, np.ndarray]] = None
        self.mkm180_y: Optional[np.ndarray] = None
        self.mkm180_y_plus: Optional[np.ndarray] = None

        self.NK_H4_stats: Optional[Dict[str, np.ndarray]] = None
        self.NK_H6_stats: Optional[Dict[str, np.ndarray]] = None
        self.NK_ref_y_H4: Optional[np.ndarray] = None
        self.NK_ref_y_H6: Optional[np.ndarray] = None
        self.NK_ref_y_uv_H4: Optional[np.ndarray] = None
        self.NK_ref_y_uv_H6: Optional[np.ndarray] = None

        self.xcomp_H6_stats: Optional[Dict[str, np.ndarray]] = None
        self.xcomp_yplus_uu_H6: Optional[np.ndarray] = None

    def load_all(self) -> None:
        """Load all enabled reference datasets"""
        if self.config.mkm180_ch_ref_on:
            self._load_mkm180()

        if self.config.mhd_NK_ref_on:
            self._load_noguchi_kasagi()

        #if self.config.mhd_XCompact_ref_on:
        #    self._load_xcompact()

    def _load_mkm180(self) -> None:
        """Load MKM180 reference data"""
        try:
            ref_means = np.loadtxt(self.REF_MKM180_MEANS_PATH)
            ref_reystress = np.loadtxt(self.REF_MKM180_REYSTRESS_PATH)

            self.mkm180_y = ref_means[:, 0]
            self.mkm180_y_plus = ref_means[:, 1]

            self.mkm180_stats = {
                'ux_velocity': ref_means[:, 2],
                'u_prime_sq': ref_reystress[:, 2],
                'v_prime_sq': ref_reystress[:, 3],
                'w_prime_sq': ref_reystress[:, 4],
                'u_prime_v_prime': ref_reystress[:, 5]
            }
            print("mkm180 reference data loaded successfully.")
        except Exception as e:
            print(f"mkm180 reference is disabled or required data is missing: {e}")

    def _load_noguchi_kasagi(self) -> None:
        """Load Noguchi & Kasagi MHD reference data"""
        try:
            NK_data = {}
            for key, path in self.NK_REF_PATHS.items():
                NK_data[key] = np.loadtxt(path)

            self.NK_ref_y_H4 = NK_data['ref_NK_Ha_4'][:, 1]
            self.NK_ref_y_H6 = NK_data['ref_NK_Ha_6'][:, 1]
            self.NK_ref_y_uv_H4 = NK_data['ref_NK_uv_Ha_4'][:, 1]
            self.NK_ref_y_uv_H6 = NK_data['ref_NK_uv_Ha_6'][:, 1]

            self.NK_H4_stats = {
                'ux_velocity': NK_data['ref_NK_Ha_4'][:, 2] * 1.02169,
                'u_prime_sq': np.square(NK_data['ref_NK_Ha_4'][:, 3]),
                'u_prime_v_prime': -1 * NK_data['ref_NK_uv_Ha_4'][:, 2],
                'v_prime_sq': np.square(NK_data['ref_NK_Ha_4'][:, 4]),
                'w_prime_sq': np.square(NK_data['ref_NK_Ha_4'][:, 5])
            }

            self.NK_H6_stats = {
                'ux_velocity': NK_data['ref_NK_Ha_6'][:, 2],
                'u_prime_sq': np.square(NK_data['ref_NK_Ha_6'][:, 3]),
                'u_prime_v_prime': -1 * NK_data['ref_NK_uv_Ha_6'][:, 2],
                'v_prime_sq': np.square(NK_data['ref_NK_Ha_6'][:, 4]),
                'w_prime_sq': np.square(NK_data['ref_NK_Ha_6'][:, 5])
            }
            print("Noguchi & Kasagi MHD channel reference data loaded successfully.")
        except Exception as e:
            print(f"NK mhd reference is disabled or required data is missing: {e}")


# =====================================================================================================================================================
# TURBULENCE STATISTICS CLASSES
# =====================================================================================================================================================

class TurbStatistic(ABC):
    """Abstract base class for turbulence statistics"""

    def __init__(self, name: str, label: str, required_quantities: List[str]):
        self.name = name
        self.label = label
        self.required_quantities = required_quantities
        self.raw_results: Dict[Tuple[str, str], np.ndarray] = {}
        self.processed_results: Dict[Tuple[str, str], np.ndarray] = {}

    @abstractmethod
    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute the statistic from required data"""
        pass

    def compute_for_case(self, case: str, timestep: str, data_loader: TurbulenceData) -> bool:
        """Compute statistic for a specific case and timestep"""
        # Gather required data
        data_dict = {}
        for quantity in self.required_quantities:
            if not data_loader.has(case, quantity, timestep):
                print(f"Missing {quantity} data for {self.name} calculation: {case}, {timestep}")
                return False
            data_dict[quantity] = data_loader.get(case, quantity, timestep)

        # Compute statistic
        result = self.compute(data_dict)
        self.raw_results[(case, timestep)] = result
        return True

    def normalize(self, ux_data: np.ndarray, Re_bulk: str, by_u_tau_sq: bool = True) -> np.ndarray:
        """Normalize values with respect to u_tau or u_tau_sq"""
        if self.name == 'ux_velocity' and by_u_tau_sq:
            # Special case: normalize ux by u_tau (not squared)
            return op.norm_ux_velocity_wrt_u_tau(ux_data, Re_bulk)
        elif by_u_tau_sq:
            return op.norm_turb_stat_wrt_u_tau_sq
        else:
            return lambda x: x  # No normalization

    def apply_symmetric_average(self, values: np.ndarray) -> np.ndarray:
        """Apply symmetric averaging"""
        return op.symmetric_average(values)

    def get_half_domain(self, values: np.ndarray) -> np.ndarray:
        """Get first half of domain"""
        return values[:(len(values)//2)]


class StreamwiseVelocity(TurbStatistic):
    """Streamwise velocity profile (ux)"""

    def __init__(self):
        super().__init__('ux_velocity', 'Streamwise Velocity', ['ux'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.read_profile(data_dict['ux'])

class ReynoldsStressUU(TurbStatistic):
    """Reynolds stress u'u'"""

    def __init__(self):
        super().__init__('u_prime_sq', "<u'u'>", ['ux', 'uu'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.compute_u_prime_sq(data_dict['ux'], data_dict['uu'])

class ReynoldsStressUV(TurbStatistic):
    """Reynolds stress u'v'"""

    def __init__(self):
        super().__init__('u_prime_v_prime', "<u'v'>", ['ux', 'uy', 'uv'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.compute_u_prime_v_prime(data_dict['ux'], data_dict['uy'], data_dict['uv'])

class ReynoldsStressVV(TurbStatistic):
    """Reynolds stress v'v'"""

    def __init__(self):
        super().__init__('v_prime_sq', "<v'v'>", ['uy', 'vv'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.compute_v_prime_sq(data_dict['uy'], data_dict['vv'])

class ReynoldsStressWW(TurbStatistic):
    """Reynolds stress w'w'"""

    def __init__(self):
        super().__init__('w_prime_sq', "<w'w'>", ['uz', 'ww'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.compute_w_prime_sq(data_dict['uz'], data_dict['ww'])
    
class TurbulentKineticEnergy(TurbStatistic):
    """Turbulent Kinetic Energy (TKE)"""

    def __init__(self):
        super().__init__('TKE', 'k', ['ux', 'uy', 'uz', 'uu', 'vv', 'ww'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        u_prime_sq = op.compute_u_prime_sq(data_dict['ux'], data_dict['uu'])
        v_prime_sq = op.compute_v_prime_sq(data_dict['uy'], data_dict['vv'])
        w_prime_sq = op.compute_w_prime_sq(data_dict['uz'], data_dict['ww'])
        return op.compute_tke(u_prime_sq, v_prime_sq, w_prime_sq)

class Temperature(TurbStatistic):
    """Temperature profile"""

    def __init__(self):
        super().__init__('temperature', 'Temperature', ['T'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.read_profile(data_dict['T'])

# =====================================================================================================================================================
# PIPELINE CLASS
# =====================================================================================================================================================

class TurbulenceStatsPipeline:
    """Orchestrates the computation and processing of turbulence statistics"""

    def __init__(self, config: Config, data_loader: TurbulenceData):
        self.config = config
        self.data_loader = data_loader
        self.statistics: List[TurbStatistic] = []
        self._register_statistics()

    def _register_statistics(self) -> None:
        """Register all enabled statistics based on configuration"""
        if self.config.ux_velocity_on:
            self.statistics.append(StreamwiseVelocity())

        if self.config.u_prime_sq_on:
            self.statistics.append(ReynoldsStressUU())

        if self.config.u_prime_v_prime_on:
            self.statistics.append(ReynoldsStressUV())

        if self.config.v_prime_sq_on:
            self.statistics.append(ReynoldsStressVV())

        if self.config.w_prime_sq_on:
            self.statistics.append(ReynoldsStressWW())

        if self.config.tke_on:
            self.statistics.append(TurbulentKineticEnergy())

        if self.config.temp_on:
            self.statistics.append(Temperature())

    def compute_all(self) -> None:
        """Compute all registered statistics for all cases and timesteps"""
        for stat in self.statistics:
            for case in self.config.cases:
                for timestep in self.config.timesteps:
                    stat.compute_for_case(case, timestep, self.data_loader)

    def process_all(self) -> None:
        """Apply normalization and averaging to all computed statistics"""

        for stat in self.statistics:
            for (case, timestep), values in stat.raw_results.items():

                # Get ux data for normalization
                ux_data = self.data_loader.get(case, 'ux', timestep)
                if ux_data is None:
                    print(f'Missing ux data for normalization: {case}, {timestep}')
                    continue

                ref_Re = op.get_ref_Re(case, self.config.cases, self.config.Re)

                # Normalize
                if self.config.norm_by_u_tau_sq and stat.name != 'temperature':
                    normed = op.norm_turb_stat_wrt_u_tau_sq(ux_data, values, ref_Re)
                else:
                    normed = values

                # Special normalization for ux velocity
                if self.config.norm_ux_by_u_tau and stat.name == 'ux_velocity':
                    normed = op.norm_ux_velocity_wrt_u_tau(ux_data, ref_Re)
                    print(f'ux velocity normalised by u_tau for {case}, {timestep}')

                # Symmetric averaging (skip for u'v')
                if self.config.half_channel_plot: 
                    if stat.name != 'u_prime_v_prime' and stat.name != 'temperature':
                        normed_avg = op.symmetric_average(normed)
                        stat.processed_results[(case, timestep)] = normed_avg
                        # print(f'Symmetric averaged data for {case}, {timestep}')
                    else:
                        stat.processed_results[(case, timestep)] = normed[:(len(normed)//2)]
                        #print(f'First half extracted for {case}, {timestep}')
                else:
                    stat.processed_results[(case, timestep)] = normed

                # Print flow info
                cur_Re = op.get_Re(case, self.config.cases, self.config.Re, ux_data, self.config.forcing)
                op.print_flow_info(ux_data, ref_Re, cur_Re, case, timestep)

    def get_statistic(self, name: str) -> Optional[TurbStatistic]:
        """Get a specific statistic by name"""
        for stat in self.statistics:
            if stat.name == name:
                return stat
        return None


# =====================================================================================================================================================
# PLOTTING CLASS
# =====================================================================================================================================================

class TurbulencePlotter:
    """Handles all plotting logic for turbulence statistics"""

    def __init__(self, config: Config, plot_config: PlotConfig, data_loader: TurbulenceData):
        self.config = config
        self.plot_config = plot_config
        self.data_loader = data_loader

    def plot(self, statistics: List[TurbStatistic], reference_data: Optional[ReferenceData] = None):
        """Main plotting method - delegates to single or multi plot"""
        if len(statistics) == 1 or not self.config.multi_plot:
            return self._plot_single_figure(statistics, reference_data)
        else:
            return self._plot_multi_figure(statistics, reference_data)

    def _plot_single_figure(self, statistics: List[TurbStatistic],
                           reference_data: Optional[ReferenceData] = None):
        """Create a single combined plot for all statistics"""
        plt.figure(figsize=(10, 6))
        plt.gcf().canvas.manager.set_window_title('Turbulence Statistics')

        for stat in statistics:
            for (case, timestep), values in stat.processed_results.items():

                # Get y coordinates
                y_plus = self._get_y_plus(case, timestep)
                if y_plus is None:
                    continue

                # Get plotting aesthetics
                color = self._get_color(case, stat.name)
                label = f'{stat.label}, {case.replace("_", " = ")}'

                # Plot main data
                self._plot_line(plt, y_plus, values, label, color)

                # Plot reference data
                if reference_data:
                    self._plot_reference_data(plt, stat.name, case, reference_data)

                # Add log scale reference lines
                if stat.name == 'ux_velocity' and self.config.ux_velocity_log_ref_on and self.config.log_y_scale:
                    self._plot_log_reference_lines(plt, y_plus)

        plt.xlabel('$y^+$')
 
        if len(statistics) == 1:
            plt.ylabel(f"Normalised {statistics[0].name.replace('_', ' ')}")
        else:
            plt.ylabel("Normalised Reynolds Stresses")

        plt.legend()
        plt.grid(True)

        return plt.gcf()

    def _plot_multi_figure(self, statistics: List[TurbStatistic],
                          reference_data: Optional[ReferenceData] = None):
        """Create separate subplots for each statistic"""
        n_stats = len(statistics)
        ncols = math.ceil(math.sqrt(n_stats))
        nrows = math.ceil(n_stats / ncols)

        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(15, 10),
            constrained_layout=True
        )
        fig.canvas.manager.set_window_title('Turbulence Statistics')

        # Ensure axs is always 2D array
        if nrows == 1 and ncols == 1:
            axs = np.array([[axs]])
        elif nrows == 1 or ncols == 1:
            axs = axs.reshape(nrows, ncols)

        # Plot each statistic
        for i, stat in enumerate(statistics):
            row = i // ncols
            col = i % ncols
            ax = axs[row, col]

            for (case, timestep), values in stat.processed_results.items():

                # Get y coordinates
                y_plus = self._get_y_plus(case, timestep)
                if y_plus is None:
                    continue

                # Get plotting aesthetics
                color = self._get_color(case, stat.name)
                label = f'{case.replace("_", " = ")}'

                # Plot main data
                self._plot_line(ax, y_plus, values, label, color)

                # Plot reference data
                if reference_data:
                    self._plot_reference_data(ax, stat.name, case, reference_data)

                # Add log scale reference lines
                if stat.name == 'ux_velocity' and self.config.ux_velocity_log_ref_on and self.config.log_y_scale:
                    self._plot_log_reference_lines(ax, y_plus)

            # Set subplot properties
            ax.set_title(f'{stat.label}')
            ax.set_ylabel(f"Normalised {stat.name.replace('_', ' ')}")
            ax.grid(True)
            ax.legend(fontsize='small')

            if row == nrows - 1:
                ax.set_xlabel('$y^+$')

        # Hide unused subplots
        for i in range(n_stats, nrows * ncols):
            row = i // ncols
            col = i % ncols
            axs[row, col].set_visible(False)

        return fig

    def _plot_line(self, ax, x: np.ndarray, y: np.ndarray, label: str, color: str) -> None:
        """Plot a single line with appropriate scale"""
        if self.config.log_y_scale:
            ax.semilogx(x, y, label=label, linestyle='-', marker='', color=color)
        elif self.config.linear_y_scale:
            ax.plot(x, y, label=label, linestyle='-', marker='', color=color)
        else:
            print('Plotting input incorrectly defined')

    def _plot_reference_data(self, ax, stat_name: str, case: str,
                            reference_data: ReferenceData) -> None:
        """Plot reference data for a given statistic"""
        # MKM180 reference
        if self.config.mkm180_ch_ref_on and reference_data.mkm180_stats:
            if stat_name in reference_data.mkm180_stats:
                self._plot_line(ax, reference_data.mkm180_y_plus,
                              reference_data.mkm180_stats[stat_name],
                              f'{self.plot_config.stat_labels[stat_name]} MKM180',
                              'black')

        # Noguchi & Kasagi reference
        if self.config.mhd_NK_ref_on:
            if case == 'Ha_4' and reference_data.NK_H4_stats and stat_name in reference_data.NK_H4_stats:
                if self.config.log_y_scale:
                    ax.semilogx(reference_data.NK_ref_y_H4,
                              reference_data.NK_H4_stats[stat_name],
                              linestyle='', marker='o',
                              label='Ha = 4, Noguchi & Kasagi',
                              color=self.plot_config.colors_1[stat_name], markevery=2)
                else:
                    ax.plot(reference_data.NK_ref_y_H4,
                          reference_data.NK_H4_stats[stat_name],
                          linestyle='', marker='o',
                          label='Ha = 4, Noguchi & Kasagi',
                          color=self.plot_config.colors_1[stat_name], markevery=2)

            elif case == 'Ha_6' and reference_data.NK_H6_stats and stat_name in reference_data.NK_H6_stats:
                if self.config.log_y_scale:
                    ax.semilogx(reference_data.NK_ref_y_H6,
                              reference_data.NK_H6_stats[stat_name],
                              linestyle='', marker='o',
                              label='Ha = 6, Noguchi & Kasagi',
                              color=self.plot_config.colors_2[stat_name], markevery=2)
                else:
                    ax.plot(reference_data.NK_ref_y_H6,
                          reference_data.NK_H6_stats[stat_name],
                          linestyle='', marker='o',
                          label='Ha = 6, Noguchi & Kasagi',
                          color=self.plot_config.colors_2[stat_name], markevery=2)

    def _plot_log_reference_lines(self, ax, y_plus: np.ndarray) -> None:
        """Plot reference lines for log-scale velocity plots"""
        if self.config.log_y_scale:
            ax.semilogx(y_plus[:15], y_plus[:15], '--', linewidth=1,
                       label='$u^+ = y^+$', color='black', alpha=0.5)
            u_plus_ref = 2.5 * np.log(y_plus) + 5.5
            ax.plot(y_plus, u_plus_ref, '--', linewidth=1,
                   label='$u^+ = 2.5ln(y^+) + 5.5$', color='black', alpha=0.5)

    def _get_y_plus(self, case: str, timestep: str) -> Optional[np.ndarray]:
        """Calculate y+ coordinates for a case"""
        ux_data = self.data_loader.get(case, 'ux', timestep)
        if ux_data is None:
            print(f'Missing ux data for plotting: {case}, {timestep}')
            return None

        if self.config.half_channel_plot:
            y = (ux_data[:len(ux_data)//2, 1] + 1)
        else:
            y = ux_data[:, 1]

        if self.config.norm_y_to_y_plus:
            cur_Re = op.get_Re(case, self.config.cases, self.config.Re, ux_data, self.config.forcing)
            return op.norm_y_to_y_plus(y, ux_data, cur_Re)
        else:
            y_plus = y
            return y_plus

    def _get_color(self, case: str, stat_name: str) -> str:
        """Get colour for a case and statistic"""
        if len(self.config.cases) <= len(self.plot_config.colours) and stat_name in self.plot_config.stat_labels:
            colour_set = op.get_col(case, self.config.cases, self.plot_config.colours)
            return colour_set[stat_name]
        else:
            return np.random.choice(list(mcolors.CSS4_COLORS.keys()))

    def save_figure(self, fig, filename: str = 'reynold_stresses.png') -> None:
        """Save figure to file"""
        fig.savefig(filename,
                   dpi=300,
                   bbox_inches='tight',
                   pad_inches=0.1,
                   facecolor='white',
                   edgecolor='none',
                   transparent=True,
                   orientation='landscape')
        print(f'Figure saved as {filename}')

    def display_figure(self) -> None:
        """Display figure"""
        print('Displaying figure...')
        plt.show()


# =====================================================================================================================================================
# MAIN EXECUTION
# =====================================================================================================================================================

def main():
    """Main execution function"""
    # Import configuration
    import config as config_module
    config = Config.from_module(config_module)

    print("="*120)
    print("TURBULENCE STATISTICS PROCESSING")
    print("="*120)

    # Load turbulence data
    print("\nLoading turbulence data...")
    data_loader = TurbulenceData(
        config.folder_path,
        config.cases,
        config.timesteps,
        config.quantities
    )
    data_loader.load_all()

    # Load reference data
    print("\nLoading reference data...")
    reference_data = ReferenceData(config)
    reference_data.load_all()

    # Compute statistics
    print("\nComputing turbulence statistics...")
    pipeline = TurbulenceStatsPipeline(config, data_loader)
    pipeline.compute_all()

    # Process statistics (normalize, average)
    print("\nProcessing statistics...")
    print("\nFlow info:\n")
    pipeline.process_all()

    # Plot results
    if config.display_fig or config.save_fig:
        print("\nGenerating plots...")
        plot_config = PlotConfig()
        plotter = TurbulencePlotter(config, plot_config, data_loader)

        fig = plotter.plot(pipeline.statistics, reference_data)

        if config.save_fig:
            plotter.save_figure(fig)

        if config.display_fig:
            plotter.display_figure()
    else:
        print('\nNo output option selected')

    print("\n" + "="*120)
    print("PROCESSING COMPLETE")
    print("="*120)


if __name__ == '__main__':
    main()
