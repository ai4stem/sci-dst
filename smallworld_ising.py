import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

class SmallWorldIsingModel:
    """
    Ising model on a Small-World network topology for modeling creativity.
    
    This model extends the traditional 2D Ising model by implementing a Watts-Strogatz
    small-world network structure, which better resembles actual brain connectivity
    patterns with both local and long-range connections.
    """
    
    def __init__(self, n_nodes=400, k=4, p_rewire=0.1, t_high=5.0, t_low=1.5, 
                 eta=0.005, rho=0.0005, learn_interval=100):
        """
        Initialize the Small-World Ising Model.
        
        Parameters:
        -----------
        n_nodes : int
            Number of nodes (idea elements) in the network
        k : int
            Initial number of neighbors to connect to each node
        p_rewire : float
            Probability of rewiring each edge (controls small-world property)
        t_high : float
            High temperature for divergent thinking phase
        t_low : float
            Low temperature for convergent thinking phase
        eta : float
            Hebbian learning rate
        rho : float
            Decay factor for coupling strengths
        learn_interval : int
            Interval (in sweeps) between Hebbian updates
        """
        self.n_nodes = n_nodes
        self.k = k
        self.p_rewire = p_rewire
        self.t_high = t_high
        self.t_low = t_low
        self.eta = eta
        self.rho = rho
        self.learn_interval = learn_interval
        
        # Create small-world network using Watts-Strogatz model
        self.graph = nx.watts_strogatz_graph(n=n_nodes, k=k, p=p_rewire)
        
        # Initialize spin states randomly (-1 or +1)
        self.spins = np.random.choice([-1, 1], size=n_nodes)
        
        # Initialize coupling strengths from Gaussian distribution
        # with distance-weighted exponential decay
        self.init_coupling_strengths()
        
        # Initialize external fields (contextual influences)
        self.h = np.random.uniform(-0.02, 0.02, size=n_nodes)
        
        # Store metrics for analysis
        self.magnetization_history = []
        self.energy_history = []
        self.entropy_history = []
        self.temperature_history = []
        
    def init_coupling_strengths(self):
        """Initialize coupling strengths with distance-dependent weights"""
        # Create coupling matrix
        self.J = np.zeros((self.n_nodes, self.n_nodes))
        
        # Calculate shortest path lengths between all node pairs
        path_lengths = dict(nx.shortest_path_length(self.graph))
        
        # Base coupling strength: Gaussian around 0.15
        base_J = 0.15 + 0.05 * np.random.randn(self.n_nodes, self.n_nodes)
        
        # Set coupling strengths with distance-dependent decay
        gamma = 3.0  # Decay parameter
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j and j in path_lengths[i]:
                    d_ij = path_lengths[i][j]
                    self.J[i, j] = base_J[i, j] * np.exp(-d_ij / gamma)
                else:
                    self.J[i, j] = 0
    
    def calculate_energy(self):
        """Calculate the total energy of the system"""
        energy = 0
        
        # Interaction energy
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                if self.graph.has_edge(i, j):
                    energy -= self.J[i, j] * self.spins[i] * self.spins[j]
        
        # External field contribution
        energy -= np.sum(self.h * self.spins)
        
        return energy
    
    def calculate_magnetization(self):
        """Calculate the absolute magnetization of the system"""
        return np.abs(np.mean(self.spins))
    
    def calculate_entropy(self):
        """Calculate Shannon entropy of the spin configuration"""
        p_plus = np.sum(self.spins == 1) / self.n_nodes
        p_minus = 1 - p_plus
        
        # Avoid log(0) errors
        if p_plus == 0 or p_minus == 0:
            return 0
        
        return -(p_plus * np.log2(p_plus) + p_minus * np.log2(p_minus))
    
    def calculate_susceptibility(self, samples=10):
        """Estimate magnetic susceptibility (sensitivity to perturbations)"""
        # Store original spins
        original_spins = self.spins.copy()
        original_m = self.calculate_magnetization()
        
        # Apply small field perturbation
        delta_h = 0.01
        perturbed_m_values = []
        
        for _ in range(samples):
            # Randomly perturb the field
            perturbed_h = self.h.copy() + delta_h * np.random.normal(size=self.n_nodes)
            
            # Store original h
            original_h = self.h.copy()
            
            # Set perturbed field
            self.h = perturbed_h
            
            # Run a few sweeps
            for _ in range(10):
                self.monte_carlo_sweep(self.t_low)
            
            # Measure resulting magnetization
            perturbed_m_values.append(self.calculate_magnetization())
            
            # Restore original state
            self.spins = original_spins.copy()
            self.h = original_h
        
        # Calculate average response
        avg_perturbed_m = np.mean(perturbed_m_values)
        
        # Susceptibility = rate of change of magnetization with field
        susceptibility = np.abs(avg_perturbed_m - original_m) / delta_h
        
        return susceptibility

    def hamming_distance(self, pattern):
        """ Calculate Hamming distance to a target pattern """
        if len(pattern) != self.n_nodes:
            print("Error: Pattern length mismatch for Hamming distance calculation.")
            return -1 # 또는 에러 발생시키기
        # 패턴이 numpy 배열인지 확인 (직접 비교는 보통 문제 없음)
        pattern_arr = np.array(pattern)
        # 다른 요소의 수를 전체 노드 수로 나눔 (정규화된 해밍 거리)
        return np.sum(self.spins != pattern_arr) / self.n_nodes

    def monte_carlo_sweep(self, temperature):
        """Perform one Monte Carlo sweep (n_nodes attempted spin flips)"""
        for _ in range(self.n_nodes):
            # Select a random node
            i = np.random.randint(0, self.n_nodes)
            
            # Calculate energy change for flipping this spin
            delta_E = 0
            
            # Contribution from neighbors
            for j in self.graph.neighbors(i):
                delta_E += 2 * self.J[i, j] * self.spins[i] * self.spins[j]
            
            # Contribution from external field
            delta_E += 2 * self.h[i] * self.spins[i]
            
            # Metropolis acceptance criterion
            if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / temperature):
                self.spins[i] *= -1
    
    def hebbian_update(self):
        """Update coupling strengths using Hebbian learning rule"""
        for i in range(self.n_nodes):
            for j in self.graph.neighbors(i):
                # Hebbian update: strengthen connections between co-activated nodes
                delta_J = self.eta * (self.spins[i] * self.spins[j] - self.rho * self.J[i, j])
                self.J[i, j] += delta_J
                self.J[j, i] += delta_J  # Keep J symmetric
    
    def critical_scan(self, t_start=5.0, t_end=1.0, t_step=0.05, sweeps_per_temp=1000):
        """
        Perform a temperature scan to identify phase transitions.
        
        Parameters:
        -----------
        t_start : float
            Starting temperature
        t_end : float
            Ending temperature
        t_step : float
            Temperature decrement step
        sweeps_per_temp : int
            Number of Monte Carlo sweeps at each temperature
        
        Returns:
        --------
        results : dict
            Dictionary containing arrays of temperature, magnetization, 
            energy, entropy, and susceptibility
        """
        # Initialize arrays to store results
        temps = np.arange(t_start, t_end - t_step/2, -t_step)
        n_temps = len(temps)
        
        magnetizations = np.zeros(n_temps)
        energies = np.zeros(n_temps)
        entropies = np.zeros(n_temps)
        susceptibilities = np.zeros(n_temps)
        
        # Run simulation at decreasing temperatures
        for i, temp in enumerate(tqdm(temps, desc="Temperature scan")):
            # Equilibrate the system at this temperature
            for sweep in range(sweeps_per_temp):
                self.monte_carlo_sweep(temp)
                
                # Apply Hebbian learning at specified intervals
                if sweep % self.learn_interval == 0:
                    self.hebbian_update()
            
            # Measure observables
            magnetizations[i] = self.calculate_magnetization()
            energies[i] = self.calculate_energy() / self.n_nodes  # Energy per spin
            entropies[i] = self.calculate_entropy()
            
            # Estimate susceptibility (computational expensive, so do less frequently)
            if i % 5 == 0:
                susceptibilities[i] = self.calculate_susceptibility()
            else:
                # Interpolate from neighboring points if not computing
                if i > 0:
                    susceptibilities[i] = susceptibilities[i-1]
            
            # Store in history
            self.magnetization_history.append(magnetizations[i])
            self.energy_history.append(energies[i])
            self.entropy_history.append(entropies[i])
            self.temperature_history.append(temp)
        
        # Return results as dictionary
        return {
            'temperature': temps,
            'magnetization': magnetizations,
            'energy': energies,
            'entropy': entropies,
            'susceptibility': susceptibilities
        }
    
    def creativity_cycle(self, n_cycles=3, sweeps_per_phase=1000):
        """
        Run alternating divergent-convergent thinking cycles.
        
        Parameters:
        -----------
        n_cycles : int
            Number of divergent-convergent cycles to run
        sweeps_per_phase : int
            Number of Monte Carlo sweeps in each phase
            
        Returns:
        --------
        results : dict
            Dictionary containing history of magnetization, energy, 
            entropy, and temperature
        """
        results = {
            'cycle': [],
            'phase': [],
            'temperature': [],
            'magnetization': [],
            'energy': [],
            'entropy': []
        }
        
        for cycle in range(n_cycles):
            # Divergent phase (high temperature)
            for sweep in tqdm(range(sweeps_per_phase), 
                              desc=f"Cycle {cycle+1}/{n_cycles} - Divergent"):
                self.monte_carlo_sweep(self.t_high)
                
                # Apply Hebbian learning at specified intervals
                if sweep % self.learn_interval == 0:
                    self.hebbian_update()
                
                # Record metrics (at intervals to save memory)
                if sweep % (sweeps_per_phase // 10) == 0:
                    results['cycle'].append(cycle)
                    results['phase'].append('divergent')
                    results['temperature'].append(self.t_high)
                    results['magnetization'].append(self.calculate_magnetization())
                    results['energy'].append(self.calculate_energy() / self.n_nodes)
                    results['entropy'].append(self.calculate_entropy())
            
            # Convergent phase (low temperature)
            for sweep in tqdm(range(sweeps_per_phase), 
                              desc=f"Cycle {cycle+1}/{n_cycles} - Convergent"):
                self.monte_carlo_sweep(self.t_low)
                
                # Apply Hebbian learning at specified intervals
                if sweep % self.learn_interval == 0:
                    self.hebbian_update()
                
                # Record metrics (at intervals to save memory)
                if sweep % (sweeps_per_phase // 10) == 0:
                    results['cycle'].append(cycle)
                    results['phase'].append('convergent')
                    results['temperature'].append(self.t_low)
                    results['magnetization'].append(self.calculate_magnetization())
                    results['energy'].append(self.calculate_energy() / self.n_nodes)
                    results['entropy'].append(self.calculate_entropy())
        
        return results
    
    def visualize_network(self, ax=None, node_size=50):
        """Visualize the current state of the network"""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))
        
        # Create a position layout for the network
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Create a node color map based on spin values
        node_colors = ['red' if spin == 1 else 'blue' for spin in self.spins]
        
        # Calculate edge weights based on coupling strengths
        edge_weights = [abs(self.J[u, v]) * 3 for u, v in self.graph.edges()]
        
        # Draw the network
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, 
                              node_size=node_size, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(self.graph, pos, width=edge_weights, 
                              alpha=0.5, edge_color='gray', ax=ax)
        
        # Add title with current magnetization
        mag = self.calculate_magnetization()
        ent = self.calculate_entropy()
        ax.set_title(f"Network State: |m|={mag:.2f}, S={ent:.2f}")
        
        ax.set_axis_off()
        return ax

    def plot_critical_scan(self, results):
        """Plot results from a critical temperature scan"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot absolute magnetization vs. temperature
        axes[0, 0].plot(results['temperature'], results['magnetization'], 'o-')
        axes[0, 0].set_xlabel('Temperature (T)')
        axes[0, 0].set_ylabel('Absolute Magnetization (|m|)')
        axes[0, 0].set_title('Magnetization vs. Temperature')
        
        # Plot energy per spin vs. temperature
        axes[0, 1].plot(results['temperature'], results['energy'], 'o-')
        axes[0, 1].set_xlabel('Temperature (T)')
        axes[0, 1].set_ylabel('Energy per Spin')
        axes[0, 1].set_title('Energy vs. Temperature')
        
        # Plot entropy vs. temperature
        axes[1, 0].plot(results['temperature'], results['entropy'], 'o-')
        axes[1, 0].set_xlabel('Temperature (T)')
        axes[1, 0].set_ylabel('Entropy (S)')
        axes[1, 0].set_title('Entropy vs. Temperature')
        
        # Plot susceptibility vs. temperature
        axes[1, 1].plot(results['temperature'], results['susceptibility'], 'o-')
        axes[1, 1].set_xlabel('Temperature (T)')
        axes[1, 1].set_ylabel('Susceptibility (χ)')
        axes[1, 1].set_title('Susceptibility vs. Temperature')
        
        # Estimate critical temperature (maximum susceptibility)
        valid_indices = results['susceptibility'] > 0
        if np.any(valid_indices):
            t_crit_idx = np.argmax(results['susceptibility'][valid_indices])
            t_crit = results['temperature'][valid_indices][t_crit_idx]
            
            # Add vertical line at critical temperature
            for ax in axes.flat:
                ax.axvline(t_crit, color='r', linestyle='--', 
                          label=f'T_c ≈ {t_crit:.3f}')
                ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_creativity_cycle(self, results):
        """Plot results from creativity cycles"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # Convert lists to arrays for easier manipulation
        x = np.arange(len(results['temperature']))
        temp = np.array(results['temperature'])
        mag = np.array(results['magnetization'])
        ent = np.array(results['entropy'])
        
        # Plot temperature
        axes[0].plot(x, temp, '-')
        axes[0].set_ylabel('Temperature (T)')
        axes[0].set_title('Creativity Cycles: Temperature Schedule')
        
        # Color background by phase
        phase_colors = {'divergent': 'lightsalmon', 'convergent': 'lightblue'}
        for i, phase in enumerate(results['phase']):
            if i == 0 or results['phase'][i] != results['phase'][i-1]:
                start_x = i
            if i == len(results['phase'])-1 or results['phase'][i] != results['phase'][i+1]:
                end_x = i
                axes[0].axvspan(start_x, end_x, alpha=0.3, color=phase_colors[phase])
        
        # Plot magnetization
        axes[1].plot(x, mag, '-')
        axes[1].set_ylabel('Abs. Magnetization (|m|)')
        axes[1].set_title('Convergence of Ideas (Higher = More Focus)')
        
        # Plot entropy
        axes[2].plot(x, ent, '-')
        axes[2].set_ylabel('Entropy (S)')
        axes[2].set_xlabel('Simulation Steps')
        axes[2].set_title('Idea Diversity (Higher = More Diverse)')
        
        # Mark cycle boundaries
        for c in range(1, max(results['cycle'])+1):
            cycle_start = next((i for i, cycle in enumerate(results['cycle']) if cycle == c), None)
            if cycle_start:
                for ax in axes:
                    ax.axvline(x=cycle_start, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig


# Example usage of the model
if __name__ == "__main__":
    # Create Small-World Ising Model for creativity
    model = SmallWorldIsingModel(
        n_nodes=400,      # 400 idea elements (20x20 equivalent)
        k=4,              # Each node initially connected to 4 neighbors
        p_rewire=0.1,     # 10% of edges are rewired (small-world property)
        t_high=5.0,       # High temperature for divergent thinking
        t_low=1.5,        # Low temperature for convergent thinking
        eta=0.005,        # Hebbian learning rate
        rho=0.0005,       # Decay factor
        learn_interval=100  # Apply Hebbian learning every 100 sweeps
    )
    
    # Visualization of initial network state
    plt.figure(figsize=(10, 8))
    model.visualize_network()
    plt.savefig('initial_network.png')
    plt.close()
    
    print("Starting critical temperature scan...")
    # Perform critical temperature scan
    scan_results = model.critical_scan(
        t_start=5.0,
        t_end=1.0,
        t_step=0.1,
        sweeps_per_temp=500  # Reduced for demonstration
    )
    
    # Plot critical scan results
    fig = model.plot_critical_scan(scan_results)
    plt.savefig('critical_scan.png')
    plt.close(fig)
    
    print("Starting creativity cycles simulation...")
    # Run creativity cycles
    cycle_results = model.creativity_cycle(
        n_cycles=3,
        sweeps_per_phase=500  # Reduced for demonstration
    )
    
    # Plot creativity cycle results
    fig = model.plot_creativity_cycle(cycle_results)
    plt.savefig('creativity_cycles.png')
    plt.close(fig)
    
    # Visualization of final network state
    plt.figure(figsize=(10, 8))
    model.visualize_network()
    plt.savefig('final_network.png')
    plt.close()
    
    print("Simulation complete.")