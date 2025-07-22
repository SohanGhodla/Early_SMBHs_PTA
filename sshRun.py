import sys
import numpy as np
from src.sgwb import SGWBEnergyDensity

def main(mu):
    # mu passed on the command line
    f_min, f_max, num_f = 1e-9, 1e-5, 30
    f_vals = np.logspace(np.log10(f_min), np.log10(f_max), num_f)
    z_max = 6

    mass1 = np.logspace(4.2, 10.3, 100)
    mass2 = np.logspace(4.2, 10.3, 100)
    m1_grid, m2_grid = np.meshgrid(mass1, mass2)

    SGWB_vs_f = SGWBEnergyDensity(m1_grid, m2_grid, mass1, mass2)
    Omega_GW_vals = SGWB_vs_f.compute_Omega_GW(f_vals, z_max, mu)

    data = np.vstack((f_vals, Omega_GW_vals)).T
    filename = f'Omega_GW_data_mu_low_mass{mu}.csv'
    np.savetxt(filename, data, delimiter=',',
               header='Frequency [Hz], Omega_GW(f)')
    print(f"Data saved to '{filename}'.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run.py <mu>")
        sys.exit(1)
    mu = float(sys.argv[1])
    main(mu)
