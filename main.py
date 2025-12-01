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
from smilp_model import solve_smilp_model
from traditional_method import solve_deterministic_model


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
    print("=== Starting Experiment ===")
    
    # 參數設定
    NUM_PATIENTS = 20
    TRAIN_SCENARIOS = 30 
    TEST_SCENARIOS = 100 
    
    # 1. 生成數據
    gen = InstanceGenerator(num_patients=NUM_PATIENTS)
    train_data = gen.generate_data(num_scenarios=TRAIN_SCENARIOS)
    
    # 2. 求解模型
    print("\n[Traditional] Solving Deterministic Model...")
    m_det = solve_deterministic_model(train_data, time_limit=60)
    sched_det = extract_schedule_via_sequencing_vars(m_det, train_data)
    
    print("\n[SMILP] Solving Stochastic Model...")
    m_smilp = solve_smilp_model(train_data, time_limit=300, gap=0.05)
    sched_smilp = extract_schedule_via_sequencing_vars(m_smilp, train_data)
    
    # 3. 模擬與收集資料
    print(f"\n[Simulation] Running {TEST_SCENARIOS} tests...")
    wait_det, wait_smilp = [], []
    last_log_det, last_log_smilp = None, None
    
    np.random.seed(999)
    for i in range(TEST_SCENARIOS):
        seed = i * 1000
        is_last = (i == TEST_SCENARIOS - 1)
        
        if is_last:
            w_d, log_d = simulate_realization(sched_det, train_data, random_seed=seed, return_log=True)
            w_s, log_s = simulate_realization(sched_smilp, train_data, random_seed=seed, return_log=True)
            last_log_det = log_d
            last_log_smilp = log_s
        else:
            w_d = simulate_realization(sched_det, train_data, random_seed=seed)
            w_s = simulate_realization(sched_smilp, train_data, random_seed=seed)
            
        wait_det.append(w_d)
        wait_smilp.append(w_s)
        
    # 4. 輸出與繪圖
    avg_d = np.mean(wait_det)
    avg_s = np.mean(wait_smilp)
    improv = (avg_d - avg_s) / avg_d * 100
    
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print(f"Traditional: {avg_d:.2f} min")
    print(f"SMILP:       {avg_s:.2f} min")
    print(f"Improvement: {improv:.2f}%")
    print("="*40)
    
    print("Generating plots...")
    plot_results(wait_det, wait_smilp, last_log_det, last_log_smilp, train_data['resources'], NUM_PATIENTS)

if __name__ == "__main__":
    run_experiment()