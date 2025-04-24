from lattice_ising import SimulateIsingLattice
from plotting import plot_lattice_results

# Run a critical temperature scan
results = SimulateIsingLattice(
    L=50,               # 50x50 lattice
    T_high=5.0,         # Starting temperature 
    T_low=1.0,          # Ending temperature
    cool_rate=0.05,     # Temperature decrement
    eta=0.005,          # Hebbian learning rate
    rho=0.0005,         # Decay factor
    learn_interval=100  # Learning interval (in sweeps)
)

# Plot results
plot_lattice_results(results)