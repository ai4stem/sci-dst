import matplotlib.pyplot as plt
from smallworld_ising import SmallWorldIsingModel

# Create model
model = SmallWorldIsingModel(
    n_nodes=400,        # Number of idea elements
    k=4,                # Initial connections per node
    p_rewire=0.1,       # Rewiring probability
    t_high=5.0,         # High temperature
    t_low=1.0,          # Low temperature
    eta=0.005,          # Learning rate
    rho=0.0005          # Decay factor
)

# Run temperature scan
print("Running temperature scan...")
scan_results = model.critical_scan()

# Save scan results visualization
fig_scan = model.plot_critical_scan(scan_results)
plt.savefig('smallworld_scan_results.png')
plt.close(fig_scan)

# Run creativity cycles
print("Running creativity cycles...")
cycle_results = model.creativity_cycle(n_cycles=3)

# Save cycle results visualization
fig_cycle = model.plot_creativity_cycle(cycle_results)
plt.savefig('smallworld_cycle_results.png')
plt.close(fig_cycle)

# Visualize network
fig_network = plt.figure(figsize=(10, 8))
model.visualize_network()
plt.savefig('smallworld_network.png')
plt.close(fig_network)

print("Simulation complete.")