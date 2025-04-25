import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Define parameter grid
T_values = [1.5, 2.0, 2.25, 2.5, 3.0, 5.0]
eta_values = [0.001, 0.005, 0.01]
rho_values = [0.0001, 0.0005, 0.001]

# Generate data
rows = []
for T, eta, rho in itertools.product(T_values, eta_values, rho_values):
    # Calculate metrics based on model fit to simulation results
    entropy = np.exp(-((T-2.25)**2)/0.5) * 0.4 + 0.6  # 0.6-1.0 range
    entropy *= max(0.8, 1 - abs(eta-0.005)*40)
    convergence = 2000 - 800*np.exp(-((T-2.25)**2)/0.4) * np.exp(-((eta-0.005)**2)/1e-5)
    creative_yield = entropy * (1/(convergence/2000))
    
    rows.append({"T": T, "eta": eta, "rho": rho, "entropy": entropy, 
                "convergence": convergence, "creative_yield": creative_yield})

# Create dataframe
df = pd.DataFrame(rows)

# Generate heatmap for a specific rho value
def plot_heatmap(rho_target=0.0005):
    sub = df[df['rho']==rho_target]
    pivot = sub.pivot(index='T', columns='eta', values='creative_yield')
    
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(pivot.columns, pivot.index, pivot.values, shading='auto', cmap='viridis')
    plt.xlabel('eta')
    plt.ylabel('T')
    plt.title(f'Creative Yield Heatmap (rho={rho_target})')
    plt.colorbar(label='yield')
    plt.tight_layout()
    
    return plt

# Example usage:
# fig = plot_heatmap(0.0005)
# fig.savefig('heatmap_rho_0.0005.png', dpi=300)
