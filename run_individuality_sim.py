# run_individuality_sim.py

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm # tqdm 추가 (진행률 표시)

# smallworld_ising.py 파일에서 SmallWorldIsingModel 클래스를 임포트합니다.
# 이 파일이 같은 디렉토리에 있거나 파이썬 경로에 있어야 합니다.
try:
    from smallworld_ising import SmallWorldIsingModel
except ImportError:
    print("Error: smallworld_ising.py not found or SmallWorldIsingModel class is missing.")
    exit()

# --- Simulation Parameters ---
N_NODES = 400           # 노드 수 (아이디어 요소)
K = 4                   # 초기 연결 수
P_REWIRE = 0.1          # 재배선 확률
T_HIGH = 5.0            # 고온 (발산적 사고)
T_LOW = 1.0             # 저온 (수렴적 사고)
ETA = 0.005             # 학습률
RHO = 0.0005            # 감쇠 상수
LEARN_INTERVAL = 100    # 학습 간격 (스윕 단위)
N_CYCLES = 3            # 실행할 사이클 수
SWEEPS_PER_PHASE = 1000 # 각 단계별 스윕 수 (조정 가능)
STEPS_PER_RECORDING = SWEEPS_PER_PHASE // 20 # 단계당 기록 횟수 (20 포인트)

# --- J Matrix Generation Functions ---

def create_J_matrix(graph, base_mean=0.15, base_std=0.05, gamma=3.0, cluster_assignments=None, intra_strength_factor=1.0, inter_strength_factor=0.5):
    """
    특정 구조 (클러스터링)를 가진 결합 강도(J) 행렬을 생성합니다.

    Parameters:
    -----------
    graph : networkx.Graph
        네트워크 그래프 객체
    base_mean : float
        기본 결합 강도의 평균
    base_std : float
        기본 결합 강도의 표준편차
    gamma : float
        거리 기반 감쇠 파라미터
    cluster_assignments : np.array or None
        각 노드의 클러스터 할당 정보. None이면 클러스터링 없음.
    intra_strength_factor : float
        클러스터 내부 연결 강도 증폭 계수
    inter_strength_factor : float
        클러스터 간 연결 강도 조절 계수 (1.0이면 변화 없음)

    Returns:
    --------
    np.array : 생성된 J 행렬 (n_nodes x n_nodes)
    """
    n_nodes = graph.number_of_nodes()
    J = np.zeros((n_nodes, n_nodes))
    # 모든 노드 쌍 간의 최단 경로 길이를 미리 계산합니다.
    try:
        path_lengths = dict(nx.shortest_path_length(graph))
    except nx.NetworkXNoPath:
        print("Warning: Graph is not connected. Using approximate path lengths.")
        # 연결되지 않은 경우, 연결된 컴포넌트 내에서만 계산하거나 다른 방식 사용
        path_lengths = {}
        for source in graph.nodes():
             path_lengths[source] = nx.shortest_path_length(graph, source=source)


    # 모든 노드 쌍에 대해 기본 노이즈 추가
    base_J_noise = base_mean + base_std * np.random.randn(n_nodes, n_nodes)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes): # 자기 루프 및 중복 계산 방지
             # 경로가 존재하는 경우에만 계산
             if i in path_lengths and j in path_lengths[i]:
                d_ij = path_lengths[i][j]
                current_base_J = base_J_noise[i, j] # Use noisy base strength

                # 클러스터링 계수 적용 (cluster_assignments가 제공된 경우)
                if cluster_assignments is not None:
                    if cluster_assignments[i] == cluster_assignments[j]:
                        # 같은 클러스터 내 연결 강화
                        current_base_J *= intra_strength_factor
                    else:
                        # 다른 클러스터 간 연결 조절
                        current_base_J *= inter_strength_factor

                # 거리 기반 감쇠 적용
                final_J = current_base_J * np.exp(-d_ij / gamma)
                J[i, j] = final_J
                J[j, i] = final_J # 대칭성 유지
             # else: 경로 없으면 J[i, j] = 0 유지

    return J

# --- Plotting Function ---

def plot_individuality_cycles(results_dict, sweeps_per_phase, points_per_phase):
    """
    다른 조건들에 대한 창의성 사이클 결과를 비교하여 플로팅합니다.

    Parameters:
    -----------
    results_dict : dict
        키는 조건 이름, 값은 해당 조건의 결과 딕셔너리
        (결과 딕셔너리는 'magnetization', 'entropy' 등 시계열 데이터 포함)
    sweeps_per_phase : int
        각 단계(고온/저온)의 스윕 수
    points_per_phase : int
        각 단계당 기록된 데이터 포인트 수
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_dict))) # 조건별 색상

    total_points = 0
    # 총 데이터 포인트 수 확인 (모든 조건이 동일하다고 가정)
    if results_dict:
        total_points = len(next(iter(results_dict.values()))['magnetization'])

    # x축 계산 (기록된 스텝 기준)
    steps_per_recording_interval = sweeps_per_phase // points_per_phase
    x_axis = np.arange(total_points) * steps_per_recording_interval

    for i, (condition_name, results) in enumerate(results_dict.items()):
        # 시계열 데이터 길이 확인
        if len(results['magnetization']) != total_points:
             print(f"Warning: Data length mismatch for condition '{condition_name}'. Skipping plot.")
             continue

        axes[0].plot(x_axis, results['magnetization'], label=condition_name, color=colors[i], alpha=0.8)
        axes[1].plot(x_axis, results['entropy'], label=condition_name, color=colors[i], alpha=0.8)

    axes[0].set_ylabel('Abs. Magnetization (|m|)')
    axes[0].set_title('Magnetization Dynamics Comparison by Condition')
    axes[0].legend()
    axes[0].grid(True, linestyle=':')

    axes[1].set_ylabel('Entropy (S)')
    axes[1].set_title('Entropy Dynamics Comparison by Condition')
    axes[1].set_xlabel(f'Simulation Sweeps (recorded every {steps_per_recording_interval} sweeps)')
    axes[1].legend()
    axes[1].grid(True, linestyle=':')

    # 사이클 경계 표시 (대략적)
    steps_per_cycle_approx = 2 * sweeps_per_phase # 고온 + 저온
    num_cycles_recorded = total_points // (2 * points_per_phase)

    for cycle in range(1, num_cycles_recorded):
           boundary_step = cycle * steps_per_cycle_approx
           axes[0].axvline(boundary_step, color='k', linestyle='--', alpha=0.5)
           axes[1].axvline(boundary_step, color='k', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('individuality_comparison_cycles.png', dpi=300)
    print("\nSaved comparison plot to individuality_comparison_cycles.png")
    # plt.show() # 필요시 주석 해제

# --- Main Simulation Loop ---

# 비교할 조건 정의
conditions = {
    "Default (No Clusters)": {
        "cluster_assignments": None,
        "intra_strength_factor": 1.0, # 변경 없음
        "inter_strength_factor": 1.0  # 변경 없음
    },
    "Structured (Stronger Clusters)": {
        "n_clusters": 5,
        "intra_strength_factor": 2.5,  # 기존 1.5에서 대폭 증가 (예: 2.5배 강화)
        "inter_strength_factor": 0.5   # 기존 0.8에서 대폭 감소 (예: 0.5배로 약화)
    }
}

all_results = {} # 모든 조건의 결과를 저장할 딕셔너리

for name, params in conditions.items():
    print(f"\n--- Running Simulation for Condition: {name} ---")

    # 1. 모델 초기화 (매번 새로운 그래프와 초기 스핀 상태로 시작)
    model = SmallWorldIsingModel(
        n_nodes=N_NODES, k=K, p_rewire=P_REWIRE,
        t_high=T_HIGH, t_low=T_LOW,
        eta=ETA, rho=RHO, learn_interval=LEARN_INTERVAL
    )
    # 초기 외부장(h)도 고정하거나 필요시 여기서 설정 가능
    # model.h = ...

    # 2. 특정 조건에 맞는 J 행렬 생성 및 설정
    cluster_assignments = None
    if "n_clusters" in params and params["n_clusters"] is not None:
        # 간단한 클러스터 할당 (노드 번호 기준)
        nodes_per_cluster = N_NODES // params["n_clusters"]
        if nodes_per_cluster == 0: nodes_per_cluster = 1 # n_nodes가 클러스터 수보다 작을 때 방지
        cluster_assignments = np.array([min(node // nodes_per_cluster, params["n_clusters"] - 1) for node in range(N_NODES)])
        print(f"   Assigned nodes to {params['n_clusters']} clusters.")

    j_matrix = create_J_matrix(
        model.graph,
        cluster_assignments=cluster_assignments,
        intra_strength_factor=params.get("intra_strength_factor", 1.0),
        inter_strength_factor=params.get("inter_strength_factor", 1.0)
    )
    model.J = j_matrix # 모델 객체의 J 행렬을 덮어쓰기
    print(f"   Generated J matrix with specified structure.")

    # 3. 창의성 사이클 시뮬레이션 실행 (상세 데이터 기록)
    print(f"   Running {N_CYCLES} creativity cycles ({SWEEPS_PER_PHASE} sweeps per phase)...")
    temp_history = []
    mag_history = []
    ent_history = []
    steps_processed = 0

    for cycle in range(N_CYCLES):
        desc_divergent = f"Cycle {cycle+1}/{N_CYCLES} - Divergent Phase"
        # 발산 단계 (고온)
        for sweep in tqdm(range(SWEEPS_PER_PHASE), desc=desc_divergent, leave=False):
            model.monte_carlo_sweep(T_HIGH)
            if sweep > 0 and sweep % LEARN_INTERVAL == 0: # 초기 상태 제외하고 학습
                model.hebbian_update()
            if sweep % (SWEEPS_PER_PHASE // STEPS_PER_RECORDING) == 0:
                 temp_history.append(T_HIGH)
                 mag_history.append(model.calculate_magnetization())
                 ent_history.append(model.calculate_entropy())

        desc_convergent = f"Cycle {cycle+1}/{N_CYCLES} - Convergent Phase"
        # 수렴 단계 (저온)
        for sweep in tqdm(range(SWEEPS_PER_PHASE), desc=desc_convergent, leave=False):
            model.monte_carlo_sweep(T_LOW)
            if sweep > 0 and sweep % LEARN_INTERVAL == 0: # 초기 상태 제외하고 학습
                 model.hebbian_update()
            if sweep % (SWEEPS_PER_PHASE // STEPS_PER_RECORDING) == 0:
                 temp_history.append(T_LOW)
                 mag_history.append(model.calculate_magnetization())
                 ent_history.append(model.calculate_entropy())

    # 현재 조건의 결과 저장
    all_results[name] = {
        'temperature': np.array(temp_history),
        'magnetization': np.array(mag_history),
        'entropy': np.array(ent_history)
        # 필요하다면 다른 결과 (예: 최종 J 행렬)도 저장 가능
        # 'final_J': model.J.copy()
    }
    print(f"   Finished condition: {name}")


# --- 결과 비교 플로팅 ---
if all_results:
    print("\nPlotting comparison results...")
    plot_individuality_cycles(all_results, SWEEPS_PER_PHASE, STEPS_PER_RECORDING)
else:
    print("\nNo results generated to plot.")

print("\nIndividuality simulation script finished.")