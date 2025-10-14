import numpy as np
import pandas as pd

class LiquidLithiumProperties:
    """
    Thermophysical properties of liquid lithium.
    Valid range: Tm (453.65 K) to approximately 1500 K
    
    Note: Pressure effects are neglected for most properties as liquids
    are assumed incompressible. Properties are primarily temperature-dependent.
    """
    
    def __init__(self):
        self.T_ref = 453.65  # K, reference temperature (melting point)
        self.T_melt = 453.65  # K, melting point
        self.T_boil = 1615.0  # K, boiling point at 1 atm
        self.M = 6.9410  # g/mol, molar mass of Li
        
    def phase(self, T, P):
        """Determine phase based on temperature"""
        if T < self.T_melt:
            return "Solid"
        elif T < self.T_boil:
            return "Liquid"
        else:
            return "Vapor"
    
    def density_mass(self, T):
        """Mass density in kg/m³"""
        return 278.5 - 0.04657 * T + 274.6 * (1 - T / 3500)**0.467
    
    def density_molar(self, T):
        """Molar density in mol/L"""
        rho_mass = self.density_mass(T)  # kg/m³
        rho_mol = rho_mass / self.M  # mol/m³
        return rho_mol / 1000  # mol/L
    
    def molar_volume(self, T):
        """Molar volume in L/mol"""
        return 1.0 / self.density_molar(T)
    
    def internal_energy(self, T, T_ref):
        """
        Molar internal energy in kJ/mol relative to reference temperature.
        For liquids: dU ≈ Cv*dT
        """
        Cv = self.heat_capacity_v(T)  # J/(mol·K)
        U = Cv * (T - T_ref) / 1000  # kJ/mol
        return U
    
    def enthalpy(self, T, T_ref):
        """
        delta H = int(Cp dT) from T_ref to T
        """

        return (4754 * (T - T_ref) - (0.925 * (T**2 - T_ref**2)) / 2 + (0.000291 * (T**3 - T_ref**3)) / 3) / 1000  # kJ/mol
    
    def entropy(self, T, T_ref):
        """
        Molar entropy in J/(mol·K) relative to reference temperature.
        dS = Cp/T * dT for constant pressure
        """
        Cp = self.heat_capacity_p(T)  # J/(mol·K)
        if T > T_ref:
            S = Cp * np.log(T / T_ref)
        else:
            S = 0
        return S
    
    def heat_capacity_p(self, T):
        """Molar heat capacity at constant pressure Cp in J/(mol·K)"""
        Cp_mass = 4754 - 0.925 * T + 0.000291 * T**2  # J/(kg·K)
        Cp_molar = Cp_mass * self.M / 1000  # J/(mol·K)
        return Cp_molar
    
    def heat_capacity_v(self, T):
        """
        Molar heat capacity at constant volume Cv in J/(mol·K)
        For liquids: Cv ≈ Cp (difference is small)
        """
        return self.heat_capacity_p(T) * 0.98  # Approximate
    
    def speed_of_sound(self, T):
        """
        Speed of sound in m/s
        Estimated from bulk modulus and density
        """
        # Bulk modulus for Li ~ 11-12 GPa
        K = 11.5e9  # Pa
        rho = self.density_mass(T)  # kg/m³
        c = np.sqrt(K / rho)
        return c
    
    def joule_thomson(self, T):
        """
        Joule-Thomson coefficient in K/MPa
        For liquids, typically very small and can be negative or positive
        Approximated as nearly zero
        """
        return 0.0  # Negligible for liquids
    
    def viscosity(self, T):
        """Dynamic viscosity in µPa·s"""
        mu_Pa_s = np.exp(-4.164 - 0.6374 * np.log(T) + (292.1 / T))  # Pa·s
        return mu_Pa_s * 1e6  # Convert to µPa·s
    
    def thermal_conductivity(self, T):
        """Thermal conductivity in W/(m·K)"""
        return 22.28 + 0.05 * T - 0.00001243 * T**2  # W/(m·K)

def generate_property_table(T_min, T_max, pressure=0.1, n_points=20, 
                           save_csv=False, filename='lithium_properties.csv'):
    """
    Generate a property table for liquid lithium over a temperature range.
    
    Parameters:
    -----------
    T_min : float
        Minimum temperature in K
    T_max : float
        Maximum temperature in K
    pressure : float
        Pressure in MPa (default 0.1 MPa = ~1 atm)
    n_points : int
        Number of temperature points
    save_csv : bool
        Whether to save the table as CSV
    filename : str
        Output filename if save_csv=True
    
    Returns:
    --------
    pandas.DataFrame
        Property table with requested columns
    """
    
    li = LiquidLithiumProperties()
    
    # Check validity
    if T_min < li.T_melt:
        print(f"Warning: T_min ({T_min} K) is below melting point ({li.T_melt} K)")
    if T_max > li.T_boil:
        print(f"Warning: T_max ({T_max} K) is above boiling point ({li.T_boil} K)")
    
    # Generate temperature array
    T_array = np.linspace(T_min, T_max, n_points)
    
    # Calculate properties in the requested order
    data = {
        'Temperature (K)': T_array,
        'Pressure (MPa)': [pressure] * n_points,
        'Density (mol/l)': [li.density_molar(T) for T in T_array],
        'Volume (l/mol)': [li.molar_volume(T) for T in T_array],
        'Internal Energy (kJ/mol)': [li.internal_energy(T) for T in T_array],
        'Enthalpy (kJ/mol)': [li.enthalpy(T) for T in T_array],
        'Entropy (J/mol*K)': [li.entropy(T) for T in T_array],
        'Cv (J/mol*K)': [li.heat_capacity_v(T) for T in T_array],
        'Cp (J/mol*K)': [li.heat_capacity_p(T) for T in T_array],
        'Sound Spd. (m/s)': [li.speed_of_sound(T) for T in T_array],
        'Joule-Thomson (K/MPa)': [li.joule_thomson(T) for T in T_array],
        'Viscosity (uPa*s)': [li.viscosity(T) for T in T_array],
        'Therm. Cond. (W/m*K)': [li.thermal_conductivity(T) for T in T_array],
        'Phase': [li.phase(T, pressure) for T in T_array]
    }
    
    df = pd.DataFrame(data)
    
    # Save if requested
    if save_csv:
        df.to_csv(filename, index=False)
        print(f"Property table saved to {filename}")
    
    return df

# usage
if __name__ == "__main__":
    # Generate table from melting point to 1200 K
    T_min = 500  # K
    T_max = 1200  # K
    pressure = 0.1  # MPa
    
    print("Generating liquid lithium property table...")
    print(f"Temperature range: {T_min} K to {T_max} K")
    print(f"Pressure: {pressure} MPa\n")
    
    table = generate_property_table(T_min, T_max, pressure=pressure, 
                                   n_points=15, save_csv=True)
    
    # Display table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.6e}'.format)
    print(table)
    
    # Example: Access properties at specific temperature
    print("\n" + "="*70)
    print("Example: Properties at 800 K:")
    print("="*70)
    li = LiquidLithiumProperties()
    T = 800
    print(f"Molar density: {li.density_molar(T):.4f} mol/L")
    print(f"Molar volume: {li.molar_volume(T):.4f} L/mol")
    print(f"Enthalpy: {li.enthalpy(T):.4f} kJ/mol")
    print(f"Cp: {li.heat_capacity_p(T):.4f} J/(mol·K)")
    print(f"Viscosity: {li.viscosity(T):.2f} µPa·s")
    print(f"Thermal conductivity: {li.thermal_conductivity(T):.2f} W/(m·K)")