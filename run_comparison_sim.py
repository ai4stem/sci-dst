# run_comparison_sim.py

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy # To copy model states

# smallworld_ising.py 파일에서 SmallWorldIsingModel 클래스를 임포트합니다.
try:
    from smallworld_ising import SmallWorldIsingModel
except ImportError:
    print("Error: smallworld_ising.py not found or SmallWorldIsingModel class is missing.")
    exit()

# --- Simulation Parameters ---
N_NODES = 100           # 노드 수 (비교를 위해 약간 줄임)
K = 4                   # 초기 연결 수
P_REWIRE = 0.1          # 재배선 확률
T_HIGH = 5.0            # SCI-DST 고온
T_LOW = 1.0             # SCI-DST 저온 (개별화 시뮬레이션과 일치)
T_BM = 1.0              # 볼츠만 머신 온도 (T_LOW와 동일하게 시작)
# T_BM = 2.8            # 또는 임계점 근처 다른 온도 시도
N_CYCLES = 3            # SCI-DST 사이클 수
SWEEPS_PER_PHASE = 500 # 각 단계별 스윕 수 (비교를 위해 약간 줄임)
TOTAL_SWEEPS = N_CYCLES * 2 * SWEEPS_PER_PHASE # 총 스윕 수
STEPS_PER_RECORDING = SWEEPS_PER_PHASE // 25 # 단계당 기록 횟수 (25 포인트)
RECORDING_INTERVAL = SWEEPS_PER_PHASE // STEPS_PER_RECORDING

# --- Boltzmann Machine Class ---

class BoltzmannMachine:
    """ A simple Boltzmann Machine for comparison """
    def __init__(self, graph, J_matrix, temperature):
        self.graph = graph
        self.J = J_matrix
        self.temperature = temperature
        self.n_nodes = graph.number_of_nodes()
        self.spins = np.random.choice([-1, 1], size=self.n_nodes) # Initial random spins

    def set_spins(self, initial_spins):
        """ Set specific initial spin configuration """
        if len(initial_spins) == self.n_nodes:
            self.spins = np.array(initial_spins).copy()
        else:
            print("Error: Initial spins length mismatch.")

    def calculate_energy(self):
        """ Calculate BM energy (interaction term only) """
        energy = 0
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                 # J is symmetric, only count pairs once
                 # Assumes J matrix stores couplings directly
                energy -= self.J[i, j] * self.spins[i] * self.spins[j]
        return energy

    def monte_carlo_sweep(self):
        """ Perform one Monte Carlo sweep at constant temperature """
        for _ in range(self.n_nodes):
            i = np.random.randint(0, self.n_nodes)
            delta_E = 0
            # Energy change calculation based on neighbors in J matrix
            # Assumes J[i,j] is non-zero only if nodes are connected in graph
            # Simplified delta_E for BM (only interaction term)
            neighbor_sum = 0
            for j in range(self.n_nodes):
                 if i != j: # Could use graph.neighbors(i) if J reflects graph structure only
                      neighbor_sum += self.J[i, j] * self.spins[j]

            delta_E = 2 * self.spins[i] * neighbor_sum

            if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / self.temperature):
                self.spins[i] *= -1

    def calculate_magnetization(self):
        """ Calculate absolute magnetization """
        return np.abs(np.mean(self.spins))

    def calculate_entropy(self):
        """ Calculate Shannon entropy """
        p_plus = np.sum(self.spins == 1) / self.n_nodes
        p_minus = 1.0 - p_plus
        if p_plus == 0 or p_minus == 0:
            return 0.0
        return -(p_plus * np.log2(p_plus) + p_minus * np.log2(p_minus))

    def hamming_distance(self, pattern):
        """ Calculate Hamming distance to a target pattern """
        if len(pattern) != self.n_nodes:
            return -1 # Error
        return np.sum(self.spins != pattern) / self.n_nodes


# --- Helper Functions ---

def create_J_from_pattern(graph, pattern):
    """ Create a J matrix biased towards a specific pattern using Hebbian rule """
    n_nodes = graph.number_of_nodes()
    J = np.zeros((n_nodes, n_nodes))
    if len(pattern) != n_nodes:
        print("Error: Pattern length must match number of nodes.")
        return J

    # Simple Hebbian storage
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            # Store pattern correlation - nodes connected in graph get stronger storage
            # Scale factor 1/N often used but optional here
            # Adding some noise or distance decay could make it more realistic
            if graph.has_edge(i,j): # Optional: only store if connected
                 J[i,j] = pattern[i] * pattern[j]
                 J[j,i] = J[i,j] # Symmetric

    # Optional: Add some baseline noise?
    # J += 0.01 * np.random.randn(n_nodes, n_nodes)
    # np.fill_diagonal(J, 0) # No self-interaction
    # J = (J + J.T) / 2 # Ensure symmetry after noise

    return J

def plot_comparison_results(results_sci_dst, results_bm, target_pattern):
    """ Plots comparison of SCI-DST and Boltzmann Machine dynamics """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    sweeps_sci = np.arange(len(results_sci_dst['hamming'])) * RECORDING_INTERVAL
    sweeps_bm = np.arange(len(results_bm['hamming'])) * RECORDING_INTERVAL

    # 1. Hamming Distance from Target Pattern
    axes[0].plot(sweeps_sci, results_sci_dst['hamming'], label='SCI-DST (Cycling T)', color='blue')
    axes[0].plot(sweeps_bm, results_bm['hamming'], label=f'Boltzmann Machine (T={T_BM})', color='red')
    axes[0].set_ylabel('Hamming Dist. from P')
    axes[0].set_title('Exploration Dynamics: Distance from Initial Pattern (P)')
    axes[0].legend()
    axes[0].grid(True, linestyle=':')
    axes[0].set_ylim(bottom=-0.05) # Start y-axis slightly below 0

    # 2. Entropy
    axes[1].plot(sweeps_sci, results_sci_dst['entropy'], label='SCI-DST (Cycling T)', color='blue')
    axes[1].plot(sweeps_bm, results_bm['entropy'], label=f'Boltzmann Machine (T={T_BM})', color='red')
    axes[1].set_ylabel('Entropy (S)')
    axes[1].set_title('System Diversity (Entropy)')
    axes[1].legend()
    axes[1].grid(True, linestyle=':')

    # 3. Magnetization (Optional, less direct comparison here)
    axes[2].plot(sweeps_sci, results_sci_dst['magnetization'], label='SCI-DST (Cycling T)', color='blue')
    axes[2].plot(sweeps_bm, results_bm['magnetization'], label=f'Boltzmann Machine (T={T_BM})', color='red')
    axes[2].set_ylabel('Abs. Magnetization (|m|)')
    axes[2].set_title('Overall Coherence (Magnetization)')
    axes[2].set_xlabel(f'Simulation Sweeps (recorded every {RECORDING_INTERVAL} sweeps)')
    axes[2].legend()
    axes[2].grid(True, linestyle=':')


    # Mark SCI-DST cycle boundaries roughly
    steps_per_cycle_approx = 2 * SWEEPS_PER_PHASE
    num_cycles_recorded = len(sweeps_sci) // (2 * (SWEEPS_PER_PHASE // RECORDING_INTERVAL))
    for cycle in range(1, num_cycles_recorded):
           boundary_step = cycle * steps_per_cycle_approx
           for ax in axes:
               ax.axvline(boundary_step, color='k', linestyle='--', alpha=0.3)


    plt.tight_layout()
    plt.savefig('comparison_sci_dst_vs_bm.png', dpi=300)
    print("\nSaved comparison plot to comparison_sci_dst_vs_bm.png")
    # plt.show()

# --- Main Comparison Simulation ---

print("--- Setting up Comparison Simulation ---")

# 1. Create Base Graph (same for both models)
base_graph = nx.watts_strogatz_graph(n=N_NODES, k=K, p=P_REWIRE)

# 2. Define Target Pattern P
target_pattern = np.random.choice([-1, 1], size=N_NODES)
print(f"   Defined target pattern P (length {N_NODES}).")

# 3. Create J matrix based on Pattern P (same for both models)
#    Note: This J is FIXED for the comparison run (eta=0 for SCI-DST)
common_j_matrix = create_J_from_pattern(base_graph, target_pattern)
print(f"   Created common J matrix storing pattern P.")

# 4. Initialize SCI-DST Model
print("\nInitializing SCI-DST Model...")
sci_dst_model = SmallWorldIsingModel(
    n_nodes=N_NODES, k=K, p_rewire=P_REWIRE, # Provide graph params if class needs them
    t_high=T_HIGH, t_low=T_LOW,
    eta=0.0, rho=0.0, # IMPORTANT: Turn off plasticity for this comparison
    learn_interval=1000000 # Effectively disable learning
)
sci_dst_model.graph = base_graph # Ensure same graph object
sci_dst_model.J = common_j_matrix.copy() # Use the fixed J
sci_dst_model.spins = target_pattern.copy() # Start at the target pattern
# sci_dst_model.h = np.zeros(N_NODES) # Optional: Set external field to zero for simplicity

print("   SCI-DST Model initialized, starting at pattern P, plasticity OFF.")

# 5. Initialize Boltzmann Machine
print("\nInitializing Boltzmann Machine...")
bm_model = BoltzmannMachine(
    graph=base_graph,
    J_matrix=common_j_matrix.copy(), # Use the same fixed J
    temperature=T_BM
)
bm_model.set_spins(target_pattern.copy()) # Start at the target pattern
print(f"   Boltzmann Machine initialized, starting at pattern P, T={T_BM}.")


# 6. Run SCI-DST Simulation (Cycles)
print("\nRunning SCI-DST Simulation...")
sci_dst_results = {'hamming': [], 'entropy': [], 'magnetization': []}
# Initial state recording
sci_dst_results['hamming'].append(sci_dst_model.hamming_distance(target_pattern))
sci_dst_results['entropy'].append(sci_dst_model.calculate_entropy())
sci_dst_results['magnetization'].append(sci_dst_model.calculate_magnetization())

sweep_count = 0
for cycle in range(N_CYCLES):
    desc_d = f"SCI-DST Cycle {cycle+1}/{N_CYCLES} - Divergent"
    for sweep in tqdm(range(SWEEPS_PER_PHASE), desc=desc_d, leave=False):
        sci_dst_model.monte_carlo_sweep(T_HIGH)
        sweep_count += 1
        if sweep_count % RECORDING_INTERVAL == 0:
             sci_dst_results['hamming'].append(sci_dst_model.hamming_distance(target_pattern))
             sci_dst_results['entropy'].append(sci_dst_model.calculate_entropy())
             sci_dst_results['magnetization'].append(sci_dst_model.calculate_magnetization())

    desc_c = f"SCI-DST Cycle {cycle+1}/{N_CYCLES} - Convergent"
    for sweep in tqdm(range(SWEEPS_PER_PHASE), desc=desc_c, leave=False):
        sci_dst_model.monte_carlo_sweep(T_LOW)
        sweep_count += 1
        if sweep_count % RECORDING_INTERVAL == 0:
            sci_dst_results['hamming'].append(sci_dst_model.hamming_distance(target_pattern))
            sci_dst_results['entropy'].append(sci_dst_model.calculate_entropy())
            sci_dst_results['magnetization'].append(sci_dst_model.calculate_magnetization())

print("   SCI-DST Simulation finished.")

# 7. Run Boltzmann Machine Simulation (Constant T)
print("\nRunning Boltzmann Machine Simulation...")
bm_results = {'hamming': [], 'entropy': [], 'magnetization': []}
# Initial state recording
bm_results['hamming'].append(bm_model.hamming_distance(target_pattern))
bm_results['entropy'].append(bm_model.calculate_entropy())
bm_results['magnetization'].append(bm_model.calculate_magnetization())

sweep_count_bm = 0
desc_bm = f"BM Simulation (T={T_BM})"
for sweep in tqdm(range(TOTAL_SWEEPS), desc=desc_bm, leave=False):
    bm_model.monte_carlo_sweep() # Uses its stored constant temperature
    sweep_count_bm += 1
    if sweep_count_bm % RECORDING_INTERVAL == 0:
        bm_results['hamming'].append(bm_model.hamming_distance(target_pattern))
        bm_results['entropy'].append(bm_model.calculate_entropy())
        bm_results['magnetization'].append(bm_model.calculate_magnetization())

print("   Boltzmann Machine Simulation finished.")

# 8. Plot Comparison
if sci_dst_results['hamming'] and bm_results['hamming']:
     print("\nPlotting comparison results...")
     plot_comparison_results(sci_dst_results, bm_results, target_pattern)
else:
     print("\nNo results generated to plot.")

print("\nComparison simulation script finished.")