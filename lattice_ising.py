import numpy as np

def SimulateIsingLattice(L=50, T_high=5.0, T_low=1.0, cool_rate=0.05, eta=0.005, rho=0.0005, learn_interval=100):
    """
    Monte Carlo simulation of 2D Ising model on a square lattice.
    
    Parameters:
    -----------
    L : int
        Lattice dimension (L x L)
    T_high : float
        Starting temperature
    T_low : float
        Ending temperature
    cool_rate : float
        Temperature decrement step
    eta : float
        Hebbian learning rate
    rho : float
        Decay factor
    learn_interval : int
        Learning interval (in sweeps)
        
    Returns:
    --------
    dict : Results containing temperatures, magnetizations, energies, and entropies
    """
    # ----- Initialization -----
    N = L * L  # total sites
    lat = np.random.choice([-1, 1], size=(L, L))  # Random initial state
    J = 0.15 + 0.05 * np.random.randn(L, L, 4)  # 4 neighbours: up/right/down/left
    h = np.random.uniform(-0.02, 0.02, size=(L, L))  # External fields
    
    # Helper function for Monte Carlo sweep
    def sweep(temp):
        """One Monte‑Carlo sweep (Metropolis)"""
        for _ in range(N):
            i, j = np.random.randint(0, L, 2)
            nn = lat[(i+1)%L, j] + lat[(i-1)%L, j] + lat[i, (j+1)%L] + lat[i, (j-1)%L]
            dE = 2 * lat[i, j] * (nn + h[i, j])
            if dE <= 0 or np.random.rand() < np.exp(-dE / temp):
                lat[i, j] *= -1
    
    # Helper function for Hebbian learning
    def hebbian():
        """Local Hebbian update to nearest‑neighbour couplings"""
        for i in range(L):
            for j in range(L):
                spins = [lat[(i-1)%L, j], lat[i, (j+1)%L], lat[(i+1)%L, j], lat[i, (j-1)%L]]
                for k, s_k in enumerate(spins):
                    J[i, j, k] += eta * (lat[i, j] * s_k - rho * J[i, j, k])
    
    # Helper function to calculate energy
    def calculate_energy():
        """Calculate system energy per spin"""
        energy = 0
        for i in range(L):
            for j in range(L):
                s = lat[i, j]
                nn = lat[(i+1)%L, j] + lat[(i-1)%L, j] + lat[i, (j+1)%L] + lat[i, (j-1)%L]
                energy += -s * nn/2  # Divide by 2 to avoid double counting
                energy += -h[i, j] * s
        return energy / N  # Energy per spin
    
    # Helper function to calculate entropy
    def calculate_entropy():
        """Calculate Shannon entropy of the spin configuration"""
        p_plus = np.sum(lat == 1) / N
        p_minus = 1 - p_plus
        
        # Avoid log(0) errors
        if p_plus == 0 or p_minus == 0:
            return 0
        
        return -(p_plus * np.log2(p_plus) + p_minus * np.log2(p_minus))
    
    # ----- Critical scan -----
    temps = []
    magnetizations = []
    energies = []
    entropies = []
    
    T = T_high
    while T > T_low:
        # First equilibrate the system
        for _ in range(500):
            sweep(T)
        
        # Then collect measurements
        mag_values = []
        energy_values = []
        entropy_values = []
        
        for sweep_idx in range(1000):
            sweep(T)
            if sweep_idx % learn_interval == 0:
                hebbian()
            
            # Take measurements every 10 sweeps
            if sweep_idx % 10 == 0:
                mag_values.append(np.abs(np.mean(lat)))
                energy_values.append(calculate_energy())
                entropy_values.append(calculate_entropy())
        
        temps.append(T)
        magnetizations.append(np.mean(mag_values))
        energies.append(np.mean(energy_values))
        entropies.append(np.mean(entropy_values))
        
        print(f"Temperature: {T:.2f}, Magnetization: {magnetizations[-1]:.4f}, Energy: {energies[-1]:.4f}")
        T -= cool_rate
    
    # Return results as a dictionary
    return {
        'temperature': np.array(temps),
        'magnetization': np.array(magnetizations),
        'energy': np.array(energies),
        'entropy': np.array(entropies)
    }

# Example usage if run directly
if __name__ == "__main__":
    results = SimulateIsingLattice()
    print("Simulation complete. Results:", results.keys())