# SCI-DST Creativity Model

This repository contains the implementation of the Socio-Cognitive Ising--Dynamic Systems (SCI-DST) framework for modeling creativity, as described in our paper "A New Perspective on Creativity: Integration of the Ising Model and Dynamic Systems Theory".

## Overview

We provide two main implementations:
1. **Traditional 2D Lattice-Based Ising Model**: The baseline model using a regular grid structure
2. **Small-World Network Ising Model**: Our extended model using Watts-Strogatz network topology to better approximate neural connectivity patterns

These algorithms simulate how creative insights emerge from the complex interactions between idea elements, modeling both divergent and convergent thinking processes.

## Requirements

The code requires the following Python packages:
- NumPy
- NetworkX
- Matplotlib
- tqdm (for progress bars)

You can install all dependencies with:
pip install -r requirements.txt

## Usage

### Traditional 2D Lattice Model

```python
from lattice_ising import SimulateIsingLattice

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
from plotting import plot_lattice_results
plot_lattice_results(results)
```
### Small-World Network Model

```
from smallworld_ising import SmallWorldIsingModel

# Create model
model = SmallWorldIsingModel(
    n_nodes=400,        # Number of idea elements
    k=4,                # Initial connections per node
    p_rewire=0.1,       # Rewiring probability
    t_high=5.0,         # High temperature
    t_low=1.5,          # Low temperature
    eta=0.005,          # Learning rate
    rho=0.0005          # Decay factor
)

# Run temperature scan
scan_results = model.critical_scan()

# Run creativity cycles
cycle_results = model.creativity_cycle(n_cycles=3)

# Visualize results
model.plot_critical_scan(scan_results)
model.plot_creativity_cycle(cycle_results)
model.visualize_network()
```

## Reproducing Paper Figures
The reproduce_figures.py script will generate all figures from the paper:

python reproduce_figures.py

Figures will be saved in the figures/ directory.
Implementation Details
The key differences between the two implementations are:

Network structure: The traditional model uses a regular 2D lattice with fixed neighbor connections, while the small-world model uses the Watts-Strogatz graph with both local connections and strategic long-range links.
Distance calculation: In the lattice model, distance is measured in grid units, while in the small-world model, it is based on network path lengths between nodes.
Susceptibility estimation: The small-world implementation includes an explicit method for estimating susceptibility through controlled field perturbations, which helps identify the critical temperatures where the system is most sensitive to external influences.

# Citation

If you use this code in your research, please cite our paper:
@article{author2025new,
  title={A New Perspective on Creativity: Integration of the Ising Model and Dynamic Systems Theory},
  author={Jho, H. and Luo, W.},
  journal={IEEE Access},
  year={2025},
  publisher={IEEE}
}


