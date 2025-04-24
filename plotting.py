import matplotlib.pyplot as plt
import numpy as np

def plot_lattice_results(results):
    """
    Plot results from the 2D lattice Ising model simulation.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing temperature, magnetization, energy, and entropy arrays
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Critical temperature for 2D square lattice Ising model
    Tc = 2.269
    
    # Plot absolute magnetization vs. temperature
    axes[0, 0].plot(results['temperature'], results['magnetization'], 'o-')
    axes[0, 0].set_xlabel('Temperature (T)')
    axes[0, 0].set_ylabel('Absolute Magnetization (|m|)')
    axes[0, 0].set_title('Magnetization vs. Temperature')
    axes[0, 0].axvline(x=Tc, color='r', linestyle='--', label=f'T_c = {Tc}')
    axes[0, 0].legend()
    
    # Plot energy per spin vs. temperature
    axes[0, 1].plot(results['temperature'], results['energy'], 'o-')
    axes[0, 1].set_xlabel('Temperature (T)')
    axes[0, 1].set_ylabel('Energy per Spin')
    axes[0, 1].set_title('Energy vs. Temperature')
    axes[0, 1].axvline(x=Tc, color='r', linestyle='--', label=f'T_c = {Tc}')
    axes[0, 1].legend()
    
    # Plot entropy vs. temperature
    axes[1, 0].plot(results['temperature'], results['entropy'], 'o-')
    axes[1, 0].set_xlabel('Temperature (T)')
    axes[1, 0].set_ylabel('Entropy (S)')
    axes[1, 0].set_title('Entropy vs. Temperature')
    axes[1, 0].axvline(x=Tc, color='r', linestyle='--', label=f'T_c = {Tc}')
    axes[1, 0].legend()
    
    # Plot susceptibility approximation (if available)
    if 'susceptibility' in results:
        axes[1, 1].plot(results['temperature'], results['susceptibility'], 'o-')
        axes[1, 1].set_xlabel('Temperature (T)')
        axes[1, 1].set_ylabel('Susceptibility (χ)')
        axes[1, 1].set_title('Susceptibility vs. Temperature')
        axes[1, 1].axvline(x=Tc, color='r', linestyle='--', label=f'T_c = {Tc}')
        axes[1, 1].legend()
    else:
        # If susceptibility not available, calculate derivative of magnetization
        temps = results['temperature']
        mags = results['magnetization']
        
        # Simple numerical differentiation
        susceptibility = np.zeros_like(temps)
        for i in range(1, len(temps)-1):
            susceptibility[i] = abs((mags[i+1] - mags[i-1]) / (temps[i+1] - temps[i-1]))
        
        axes[1, 1].plot(temps[1:-1], susceptibility[1:-1], 'o-')
        axes[1, 1].set_xlabel('Temperature (T)')
        axes[1, 1].set_ylabel('Susceptibility (est.)')
        axes[1, 1].set_title('Estimated Susceptibility vs. Temperature')
        axes[1, 1].axvline(x=Tc, color='r', linestyle='--', label=f'T_c = {Tc}')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('lattice_results.png', dpi=300)
    plt.show()
    
    return fig