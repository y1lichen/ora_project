import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from collections import deque
from instance_generator import InstanceGenerator
import time

# Import updated models
from smilp_model import solve_smilp_mco
from traditional_method import solve_deterministic_model, evaluate_baseline_mean_value_model

def extract_schedule_via_sequencing_vars(model, data):
    """
    從 Gurobi 模型中解析 x 和 s1 變數，建立用於甘特圖的佇列
    修正：處理 NoneType 比較錯誤，改用病人到達時間作為替代排序鍵值
    """
    if not model:
        print("No model provided for extraction.")
        return {}
    
    # 嘗試檢查模型狀態，若無法存取則忽略
    try:
        if model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
            return {}
    except:
        pass

    activities = data['activities']
    resources = data['resources']
    patients = data['patients']
    
    # 建立病人 ID -> 病人資料的查找表 (用來查 arrival_time)
    patient_map = {p['id']: p for p in patients}

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

    # 3. 建立 Resource Queues
    queues = {}
    instance_groups = {} 
    global_res_idx = 0
    
    # 初始化所有資源實例的空列表
    for r_name, r_info in resources.items():
        cap = r_info['capacity']
        for k in range(cap):
            instance_groups[global_res_idx] = []
            global_res_idx += 1
            
    # 將活動分配到對應的資源群組
    for a in activities:
        act_id = a['id']
        # 只有當活動有被指派時才處理 (Baseline 有時是隱式指派，這裡假設顯式)
        if act_id in assignment:
            target_inst = assignment[act_id]
            if target_inst in instance_groups:
                instance_groups[target_inst].append(a)

    # 4. 拓撲排序 (決定每個資源上的執行順序)
    for inst_id, acts in instance_groups.items():
        if not acts:
            queues[inst_id] = []
            continue
            
        adj = {a['id']: [] for a in acts}
        in_degree = {a['id']: 0 for a in acts}
        act_map = {a['id']: a for a in acts}
        ids = [a['id'] for a in acts]
        
        # 建立相依圖
        for i in range(len(ids)):
            for j in range(len(ids)):
                id_a, id_b = ids[i], ids[j]
                if id_a == id_b: continue
                # 若模型決定 id_b 在 id_a 之後 (s1=1)
                # 注意：這裡假設 s1_i_j=1 代表 j 排在 i 後面 (或是不能在 i 前面)
                # 具體依賴你的 SMILP 模型約束方向。
                # 這裡使用通用的拓樸排序邏輯。
                if precedence_map.get((id_a, id_b), False):
                     adj[id_a].append(id_b)
                     in_degree[id_b] += 1

        # --- [關鍵修正] ---
        # 定義排序鍵值函式：若 scheduled_start 為 None，改用病人到達時間
        def get_sort_key(uid):
            act = act_map[uid]
            t_sched = act['scheduled_start']
            if t_sched is None:
                # 取得該活動所屬病人的到達時間
                t_sched = patient_map[act['patient_id']]['arrival_time']
            # 返回 tuple (時間, id) 以確保唯一性與確定性
            return (t_sched, uid)

        sorted_acts = []
        # 使用修正後的鍵值進行初始排序
        acts_by_arrival = sorted(ids, key=get_sort_key)
        
        # 開始拓撲排序
        zero_in_degree = deque([uid for uid in acts_by_arrival if in_degree[uid] == 0])
        
        processed_count = 0
        while zero_in_degree:
            u_id = zero_in_degree.popleft()
            sorted_acts.append(act_map[u_id])
            processed_count += 1
            
            # 將相鄰節點的入度減一，若為0則加入佇列（保持 acts_by_arrival 的相對順序）
            # 這裡簡單處理，直接找 adj
            neighbors = sorted(adj[u_id], key=get_sort_key)
            for v_id in neighbors:
                in_degree[v_id] -= 1
                if in_degree[v_id] == 0:
                    zero_in_degree.append(v_id)
        
        # 如果有循環 (Cycle) 導致無法完成排序，退回到簡單的時間排序
        if processed_count < len(acts):
            # print(f"Warning: Cycle detected or unconnected graph in resource {inst_id}, falling back to arrival sort.")
            sorted_acts = sorted(acts, key=lambda x: get_sort_key(x['id']))
            
        queues[inst_id] = sorted_acts
        
    return queues

def simulate_realization(queues, data, random_seed=None, return_log=False):
    """
    模擬真實執行情況 (用於繪製甘特圖)
    """
    if random_seed: np.random.seed(random_seed)
    activities = data['activities']
    patients = data['patients']
    
    realized_durations = {}
    for a in activities:
        mu = a['mean_duration']
        sigma = 1.0 
        log_mu = np.log(mu) - (sigma**2)/2
        dur = np.random.lognormal(log_mu, sigma)
        realized_durations[a['id']] = max(1, round(dur))
        
    instance_ready_time = {k: 0 for k in queues.keys()}
    patient_ready_time = {p['id']: p['arrival_time'] for p in patients}
    activity_end_times = {}
    finished_acts = set()
    total_wait = 0
    active_queues = {k: deque(v) for k, v in queues.items()}
    num_total = len(activities)
    schedule_log = []
    
    # 簡單的時間步進模擬或事件驅動模擬
    # 這裡使用簡單的死結預防迴圈 (若無進展則跳出)
    max_loops = 10000
    loops = 0
    
    while len(finished_acts) < num_total and loops < max_loops:
        loops += 1
        progress = False
        for inst_id, q in active_queues.items():
            if not q: continue
            act = q[0]
            p_id = act['patient_id']
            
            # Check prerequisites
            if act['is_start']:
                if p_id not in patient_ready_time: continue # Should not happen
                p_ready = patient_ready_time[p_id]
            else:
                prev_id = act['predecessor']
                if prev_id not in finished_acts: continue
                p_ready = activity_end_times[prev_id]
            
            r_ready = instance_ready_time[inst_id]
            
            # Execute
            start_t = max(p_ready, r_ready)
            dur = realized_durations[act['id']]
            end_t = start_t + dur
            
            instance_ready_time[inst_id] = end_t
            activity_end_times[act['id']] = end_t
            finished_acts.add(act['id'])
            q.popleft()
            
            wait = start_t - p_ready
            total_wait += wait
            progress = True
            
            if return_log:
                schedule_log.append({
                    'inst_id': inst_id, 'patient_id': p_id,
                    'start': start_t, 'end': end_t, 'duration': dur
                })
                
        if not progress and len(finished_acts) < num_total:
            # Deadlock or waiting
            break
            
    if return_log:
        return total_wait, schedule_log
    return total_wait

def plot_results(log_det, log_smilp, resources, num_patients):
    """
    繪製 Gantt Chart 比較圖
    """
    print("\nGenerating Gantt Charts...")
    
    # 1. 建立資源 ID 到名稱的映射 (Y軸標籤)
    inst_to_name = {}
    global_res_idx = 0
    y_ticks = []
    y_labels = []
    
    # 確保順序與 extract_schedule 一致
    for r_name, r_info in resources.items():
        cap = r_info['capacity']
        for k in range(cap):
            # 只有當容量 > 1 時才顯示編號，例如 "Nurse 1", "Nurse 2"
            name_str = f"{r_name} {k+1}" if cap > 1 else r_name
            inst_to_name[global_res_idx] = name_str
            y_ticks.append(global_res_idx)
            y_labels.append(name_str)
            global_res_idx += 1
            
    # 2. 設定顏色 (每個病人一個顏色)
    # 使用 tab20 色盤，若病人超過 20 人則循環使用
    cmap = plt.get_cmap('tab20')
    patient_colors = {i: cmap(i % 20) for i in range(num_patients)}

    # 建立畫布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # 輔助函式：畫單一甘特圖
    def draw_gantt_on_ax(ax, log_data, title):
        max_time = 0
        # 繪製長條
        for entry in log_data:
            inst = entry['inst_id']
            start = entry['start']
            dur = entry['duration']
            pid = entry['patient_id']
            end = entry['end']
            max_time = max(max_time, end)
            
            # 畫 Bar
            rect = ax.barh(inst, dur, left=start, height=0.6, 
                           color=patient_colors[pid], edgecolor='black', alpha=0.8)
            
            # 在 Bar 中間標示病人 ID (若時間夠長)
            if dur > 2: 
                ax.text(start + dur/2, inst, f"P{pid}", 
                        ha='center', va='center', color='white', 
                        fontsize=8, fontweight='bold', clip_on=True)
        
        # 設定軸標籤與樣式
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.invert_yaxis() # 讓 ID 0 (Intake) 在最上面
        return max_time

    # 繪製兩個子圖
    max_t1 = draw_gantt_on_ax(ax1, log_det, "Baseline (Deterministic) Schedule - Simulation Realization")
    max_t2 = draw_gantt_on_ax(ax2, log_smilp, "SMILP (Stochastic) Schedule - Simulation Realization")
    
    # 設定 X 軸範圍 (取兩者最大值，讓比例尺一致)
    final_max = max(max_t1, max_t2)
    ax1.set_xlim(0, final_max * 1.05)
    ax2.set_xlim(0, final_max * 1.05)
    ax2.set_xlabel("Time (minutes)", fontsize=12)

    plt.tight_layout()
    output_file = "schedule_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

def run_experiment():
    print("="*60)
    print("SMILP Experiment: Fair Comparison & VSS Computation")
    print("="*60)
    
    NUM_PATIENTS = 20
    MCO_N0 = 5
    MCO_N_PRIME = 20
    MCO_K = 3
    EPSILON = 0.05
    
    # 1. Generate Data
    print(f"\n[1/4] Generating instance ({NUM_PATIENTS} patients)...")
    gen = InstanceGenerator(num_patients=NUM_PATIENTS)
    train_data = gen.generate_data(num_scenarios=MCO_N_PRIME)
    
    # 2. Solve Deterministic Baseline
    print(f"\n[2/4] Solving Deterministic Baseline Model...")
    m_baseline = solve_deterministic_model(train_data)
    print(f"      Baseline Plan Objective (Mean): {m_baseline.ObjVal:.4f}")
    
    # 3. Solve SMILP with MCO
    print(f"\n[3/4] Solving SMILP Model (MCO)...")
    smilp_result = solve_smilp_mco(
        train_data, N0=MCO_N0, N_prime=MCO_N_PRIME, K=MCO_K, epsilon=EPSILON
    )
    print(f"      SMILP Optimal N: {smilp_result.optimal_sample_size}")
    print(f"      SMILP Upper Bound (v_N'): {smilp_result.upper_bound:.4f}")
    
    # 4. Fair Evaluation of Baseline (Using SMILP's validation scenarios)
    print(f"\n[4/4] Evaluating Baseline on Shared Validation Scenarios...")
    # 這裡是關鍵：我們從 smilp_result 拿出它用來計算上界的那組場景
    shared_scenarios = smilp_result.validation_scenarios
    
    if shared_scenarios is None:
        print("Error: No validation scenarios returned from SMILP.")
        return

    baseline_realized_avg, _ = evaluate_baseline_mean_value_model(
        m_baseline, train_data, shared_scenarios
    )
    
    # 5. VSS Calculation
    print(f"\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    v_smilp = smilp_result.upper_bound
    v_base = baseline_realized_avg
    vss = v_base - v_smilp
    imp = (vss / v_base * 100) if v_base > 0 else 0
    
    print(f"SMILP Performance (v_N'):      {v_smilp:.4f} min")
    print(f"Baseline Performance (v_base): {v_base:.4f} min")
    print(f"VSS (Value of Stoch. Sol.):    {vss:.4f} min")
    print(f"Improvement:                   {imp:.2f}%")
    
    # 6. Plotting (Single Instance Visualization)
    # Extract schedules from the solved models
    queues_det = extract_schedule_via_sequencing_vars(m_baseline, train_data)
    queues_smilp = extract_schedule_via_sequencing_vars(smilp_result.final_model, train_data)
    
    # Run one simulation for the Gantt chart
    _, log_det = simulate_realization(queues_det, train_data, random_seed=999, return_log=True)
    _, log_smilp = simulate_realization(queues_smilp, train_data, random_seed=999, return_log=True)
    
    # 呼叫你原本的繪圖函式 (請確保 plot_results 函式定義存在)
    plot_results(log_det, log_smilp, train_data['resources'], NUM_PATIENTS)
    print("\nExperiment Complete.")

if __name__ == "__main__":
    run_experiment()