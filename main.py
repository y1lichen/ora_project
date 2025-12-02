import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gurobipy as gp
from gurobipy import GRB
from collections import deque
from instance_generator import InstanceGenerator
import time

# Import models
from smilp_model import solve_smilp_mco
from traditional_method import solve_deterministic_model, evaluate_baseline_mean_value_model


def extract_schedule_via_sequencing_vars(model, data):
    if not model or model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        print("Model not optimal or solved.")
        return {}

    activities = data['activities']
    resources = data['resources']
    
    # 1. 提取資源指派 (x)
    assignment = {} 
    for v in model.getVars():
        if v.VarName.startswith("x_") and v.X > 0.5:
            parts = v.VarName.split('_')
            act_id, inst_id = int(parts[1]), int(parts[2])
            assignment[act_id] = inst_id
            
    # 2. 提取順序 (s1)
    precedence_map = {} 
    for v in model.getVars():
        if v.VarName.startswith("s1_") and v.X > 0.5:
            parts = v.VarName.split('_')
            i, j = int(parts[1]), int(parts[2])
            precedence_map[(i, j)] = True

    # 3. 建立 Resource Queues (Sorting)
    queues = {}
    instance_groups = {} 
    global_res_idx = 0
    
    # 初始化
    for r_name, r_info in resources.items():
        cap = r_info['capacity']
        real_cap = cap if cap < 10 else 1
        for k in range(real_cap):
            instance_groups[global_res_idx] = []
            global_res_idx += 1
            
    # 將活動分配到對應的資源實例
    for a in activities:
        act_id = a['id']
        r_type = a['resource_type']
        r_cap = resources[[n for n, r in resources.items() if r['id'] == r_type][0]]['capacity']
        
        if r_cap < 10 and act_id in assignment:
            target_inst = assignment[act_id]
        else:
            start_idx = 0
            for r_k, r_v in resources.items():
                if r_v['id'] == r_type: break
                c = r_v['capacity']
                start_idx += (c if c < 10 else 1)
            target_inst = start_idx
            
        if target_inst not in instance_groups: instance_groups[target_inst] = []
        instance_groups[target_inst].append(a)

    # 拓撲排序
    for inst_id, acts in instance_groups.items():
        if not acts:
            queues[inst_id] = []
            continue
            
        first_act = acts[0]
        r_type = first_act['resource_type']
        r_cap = resources[[n for n, r in resources.items() if r['id'] == r_type][0]]['capacity']
        
        if r_cap >= 10:
            queues[inst_id] = sorted(acts, key=lambda x: x['scheduled_start'])
            continue

        adj = {a['id']: [] for a in acts}
        in_degree = {a['id']: 0 for a in acts}
        act_map = {a['id']: a for a in acts}
        ids = [a['id'] for a in acts]
        
        for i in range(len(ids)):
            for j in range(len(ids)):
                id_a, id_b = ids[i], ids[j]
                if id_a == id_b: continue
                if precedence_map.get((id_a, id_b), False):
                    adj[id_a].append(id_b)
                    in_degree[id_b] += 1
                    
        sorted_acts = []
        acts_by_arrival = sorted(ids, key=lambda uid: act_map[uid]['scheduled_start'])
        zero_in_degree = deque([uid for uid in acts_by_arrival if in_degree[uid] == 0])
        
        while zero_in_degree:
            u_id = zero_in_degree.popleft()
            sorted_acts.append(act_map[u_id])
            for v_id in adj[u_id]:
                in_degree[v_id] -= 1
                if in_degree[v_id] == 0:
                    zero_in_degree.append(v_id)
        
        if len(sorted_acts) < len(acts):
            sorted_acts = sorted(acts, key=lambda x: x['scheduled_start'])
            
        queues[inst_id] = sorted_acts
        
    return queues

# --- 3. 模擬器 (更新版：支援回傳 Log) ---
def simulate_realization(queues, data, random_seed=None, return_log=False):
    """
    模擬真實執行情況
    return_log: 若為 True，回傳 (total_wait, schedule_log)
    """
    if random_seed: np.random.seed(random_seed)
    patients = data['patients']
    activities = data['activities']
    
    # 生成真實執行時間
    realized_durations = {}
    for a in activities:
        mu = a['mean_duration']
        sigma = 1.0 
        log_mu = np.log(mu) - (sigma**2)/2
        dur = np.random.lognormal(log_mu, sigma)
        realized_durations[a['id']] = max(1, round(dur))
        
    # 初始化狀態
    instance_ready_time = {k: 0 for k in queues.keys()}
    patient_ready_time = {p['id']: p['arrival_time'] for p in patients}
    activity_end_times = {}
    
    finished_acts = set()
    total_wait = 0
    active_queues = {k: deque(v) for k, v in queues.items()}
    num_total = len(activities)
    
    schedule_log = [] # 儲存 Gantt 圖資料
    
    while len(finished_acts) < num_total:
        progress = False
        
        for inst_id, q in active_queues.items():
            if not q: continue
            
            act = q[0]
            p_id = act['patient_id']
            
            if act['is_start']:
                p_ready = patient_ready_time[p_id]
            else:
                prev_id = act['predecessor']
                if prev_id not in finished_acts: continue
                p_ready = activity_end_times[prev_id]
            
            r_ready = instance_ready_time[inst_id]
            
            # 執行
            start_t = max(p_ready, r_ready)
            dur = realized_durations[act['id']]
            end_t = start_t + dur
            
            # 更新狀態
            instance_ready_time[inst_id] = end_t
            activity_end_times[act['id']] = end_t
            finished_acts.add(act['id'])
            q.popleft()
            
            wait = start_t - p_ready
            total_wait += wait
            progress = True
            
            if return_log:
                schedule_log.append({
                    'inst_id': inst_id,
                    'patient_id': p_id,
                    'start': start_t,
                    'end': end_t,
                    'resource_name': act['resource_name'],
                    'duration': dur
                })
            
        if not progress and len(finished_acts) < num_total:
            break
            
    if return_log:
        return total_wait, schedule_log
    return total_wait

# --- 4. 繪圖函式 (新增) ---
def plot_results(wait_det, wait_smilp, log_det, log_smilp, resources, num_patients):
    """
    繪製 Boxplot 比較圖與 Gantt Chart 排程圖
    """
    # 建立資源名稱映射
    inst_to_name = {}
    global_res_idx = 0
    y_labels = []
    y_ticks = []
    
    for r_name, r_info in resources.items():
        cap = r_info['capacity']
        real_cap = cap if cap < 10 else 1
        for k in range(real_cap):
            name_str = f"{r_name} {k+1}" if real_cap > 1 else r_name
            inst_to_name[global_res_idx] = name_str
            y_ticks.append(global_res_idx)
            y_labels.append(name_str)
            global_res_idx += 1
            
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Boxplot
    ax1 = plt.subplot(3, 1, 1)
    bp = ax1.boxplot([wait_det, wait_smilp], labels=['Traditional (Mean-based)', 'SMILP (Stochastic)'], 
                     patch_artist=True, vert=False, widths=0.5)
    colors = ['#ff9999', '#66b3ff']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        
    avg_d = np.mean(wait_det)
    avg_s = np.mean(wait_smilp)
    improv = (avg_d - avg_s) / avg_d * 100
    ax1.set_title(f'Total Waiting Time Comparison (N={num_patients} Patients)\nImprovement: {improv:.2f}%', fontsize=14)
    ax1.set_xlabel('Total Wait Time (min)')
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Gantt Chart Helper
    def draw_gantt(ax, log_data, title):
        cmap = plt.get_cmap('tab20')
        patient_colors = {i: cmap(i % 20) for i in range(num_patients)}
        
        max_end = 0
        for entry in log_data:
            inst = entry['inst_id']
            start = entry['start']
            dur = entry['duration']
            pid = entry['patient_id']
            max_end = max(max_end, entry['end'])
            
            ax.barh(inst, dur, left=start, height=0.6, 
                    color=patient_colors[pid], edgecolor='black', alpha=0.8)
            if dur > 2:
                ax.text(start + dur/2, inst, f"P{pid}", ha='center', va='center', 
                        color='white', fontsize=7, fontweight='bold')

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Time (min)')
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        return max_end

    # 2. Traditional Schedule
    ax2 = plt.subplot(3, 1, 2)
    max_t_d = draw_gantt(ax2, log_det, "Traditional Method Schedule (Single Simulation Run)")
    
    # 3. SMILP Schedule
    ax3 = plt.subplot(3, 1, 3)
    max_t_s = draw_gantt(ax3, log_smilp, "SMILP Method Schedule (Single Simulation Run)")
    
    final_max = max(max_t_d, max_t_s)
    ax2.set_xlim(0, final_max * 1.05)
    ax3.set_xlim(0, final_max * 1.05)
    
    plt.tight_layout()
    plt.savefig("experiment_results.png", dpi=300)

# --- 5. 主執行流程 (更新版) ---
def run_experiment():
    """
    Paper experiment framework (Section 4.2):
    1. Solve deterministic baseline model
    2. Solve SMILP model with MCO
    3. Evaluate baseline model with K SAA replicates
    4. Compute Value of Stochastic Solution (VSS)
    5. Compare performance metrics
    """
    print("="*60)
    print("SMILP Experiment: Patient Scheduling with Stochastic Durations")
    print("="*60)
    
    # Parameters
    NUM_PATIENTS = 20
    MCO_N0 = 5              # Initial SAA sample size
    MCO_N_PRIME = 20        # Simulation sample size
    MCO_K = 3               # MCO replicates
    EVAL_K = 5              # Baseline evaluation replicates
    EPSILON = 0.05          # AOI convergence tolerance
    
    print(f"\nExperiment Configuration:")
    print(f"  - Patients: {NUM_PATIENTS}")
    print(f"  - MCO N0={MCO_N0}, N'={MCO_N_PRIME}, K={MCO_K}, ε={EPSILON}")
    print(f"  - Baseline eval: K={EVAL_K} replicates")
    
    # 1. Generate data
    print(f"\n[1/4] Generating instance...")
    gen = InstanceGenerator(num_patients=NUM_PATIENTS)
    train_data = gen.generate_data(num_scenarios=MCO_N_PRIME)
    print(f"      Generated {len(train_data['activities'])} activities across {NUM_PATIENTS} patients")
    
    # 2. Solve deterministic baseline model
    print(f"\n[2/4] Solving Deterministic Baseline Model...")
    m_baseline = solve_deterministic_model(train_data, time_limit=60)
    if m_baseline.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
        print("      ERROR: Baseline model failed to solve.")
        return
    print(f"      ✓ Baseline deterministic objective: {m_baseline.ObjVal:.4f}")
    
    # 3. Solve SMILP model with MCO
    print(f"\n[3/4] Solving SMILP Model with MCO Algorithm...")
    smilp_result = solve_smilp_mco(
        train_data,
        N0=MCO_N0,
        N_prime=MCO_N_PRIME,
        K=MCO_K,
        epsilon=EPSILON,
        time_limit=300,
        gap=0.05
    )
    print(f"      ✓ SMILP optimal sample size: N={smilp_result.optimal_sample_size}")
    print(f"      ✓ SMILP lower bound (v̄_N): {smilp_result.lower_bound:.4f}")
    print(f"      ✓ SMILP upper bound (v̄_N'): {smilp_result.upper_bound:.4f}")
    print(f"      ✓ AOI convergence history: {[f'{x:.6f}' for x in smilp_result.aoi_history]}")
    
    # 4. Evaluate baseline with K SAA replicates (paper evaluation procedure)
    print(f"\n[4/4] Evaluating Baseline Mean Value Model (K={EVAL_K} replicates)...")
    baseline_avg_obj, baseline_objectives = evaluate_baseline_mean_value_model(
        m_baseline,
        train_data,
        K=EVAL_K,
        time_limit=300,
        gap=0.05
    )
    
    # 5. Compute Value of Stochastic Solution (VSS)
    print(f"\n" + "="*60)
    print("FINAL RESULTS & VSS COMPUTATION")
    print("="*60)
    
    v_smilp = smilp_result.upper_bound  # v̄_N'
    v_baseline = baseline_avg_obj        # v̄base_N'
    vss = v_baseline - v_smilp           # VSS = v̄base_N' - v̄_N'
    vss_percent = (vss / v_baseline * 100) if v_baseline != 0 else 0
    
    print(f"\nValue of Stochastic Solution (VSS):")
    print(f"  SMILP solution (v̄_N'):          {v_smilp:.4f} min")
    print(f"  Baseline mean value (v̄base_N'): {v_baseline:.4f} min")
    print(f"  VSS = v̄base_N' - v̄_N':         {vss:.4f} min")
    print(f"  VSS percentage:                   {vss_percent:.2f}%")
    
    if vss > 0:
        print(f"\n  ✓ Stochastic solution is {vss_percent:.2f}% BETTER than deterministic baseline")
        print(f"    This demonstrates the value of modeling activity duration uncertainty.")
    else:
        print(f"\n  Note: Baseline mean value solution achieved better objective value.")
        print(f"    This can occur due to SAA approximation gaps.")
    
    print(f"\nSMILP Solution Quality:")
    print(f"  Approximation gap: {smilp_result.aoi_history[-1] if smilp_result.aoi_history else 'N/A':.6f}")
    print(f"  Scenario samples used: {smilp_result.optimal_sample_size}")
    print(f"  Validation samples: {MCO_N_PRIME}")
    
    print(f"\n" + "="*60)

if __name__ == "__main__":
    run_experiment()