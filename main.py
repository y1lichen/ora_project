import numpy as np
import random
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from collections import deque
import time

# Import models
from smilp_model import solve_smilp_model
from traditional_method import solve_deterministic_model

# --- 1. 資料生成器 ---
class InstanceGenerator:
    def __init__(self, num_patients=20, arrival_interval=10, random_seed=42):
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.num_patients = num_patients
        self.arrival_interval = arrival_interval
        
        self.resources = {
            'Intake':        {'capacity': 100, 'id': 0}, 
            'Radiology Tech':{'capacity': 1,   'id': 1}, 
            'Provider':      {'capacity': 2,   'id': 2}, 
            'Ortho Tech':    {'capacity': 1,   'id': 3}, 
            'Discharge':     {'capacity': 100, 'id': 4}
        }
        
        self.pathway_data = [
            {'prob': 0.3803, 'sequence': ['Intake', 'Radiology Tech', 'Provider', 'Ortho Tech', 'Discharge'], 'means': [4.7, 3.48, 4.52, 11.62, 3.43], 'variances':[3.25, 4.85, 13.9, 185.52, 2.63]},
            {'prob': 0.2455, 'sequence': ['Intake', 'Provider', 'Ortho Tech', 'Discharge'], 'means': [4.93, 4.99, 12.47, 3.69], 'variances':[4.0, 20.11, 206.96, 2.92]},
            {'prob': 0.1393, 'sequence': ['Intake', 'Ortho Tech', 'Discharge'], 'means': [4.75, 11.82, 3.44], 'variances':[4.35, 232.7, 3.36]},
            {'prob': 0.1378, 'sequence': ['Intake', 'Radiology Tech', 'Ortho Tech', 'Discharge'], 'means': [4.91, 3.58, 11.25, 3.49], 'variances':[3.75, 5.79, 224.31, 3.31]},
            {'prob': 0.0971, 'sequence': ['Intake', 'Radiology Tech', 'Provider', 'Discharge'], 'means': [5.06, 3.52, 6.15, 3.62], 'variances':[3.98, 4.95, 30.61, 4.06]}
        ]

    def _convert_params_to_lognormal(self, mean, var):
        sigma2 = np.log(1 + var / (mean**2))
        sigma = np.sqrt(sigma2)
        mu = np.log(mean) - 0.5 * sigma2
        return mu, sigma

    def generate_data(self, num_scenarios=1):
        patients, activities = [], []
        current_arrival_time = 0
        activity_global_id = 0
        path_probs = np.array([p['prob'] for p in self.pathway_data])
        path_probs /= np.sum(path_probs)
        
        for p_id in range(self.num_patients):
            path_idx = np.random.choice(len(self.pathway_data), p=path_probs)
            path_info = self.pathway_data[path_idx]
            sequence, means, variances = path_info['sequence'], path_info['means'], path_info['variances']
            
            patient_activities = []
            interval = max(1, np.random.normal(self.arrival_interval, 1.0)) 
            current_arrival_time += int(round(interval))
            previous_act_id = None
            
            for step_idx, res_name in enumerate(sequence):
                act_id = activity_global_id
                activity_global_id += 1
                arithmetic_mean = means[step_idx]
                arithmetic_var = variances[step_idx]
                mu, sigma = self._convert_params_to_lognormal(arithmetic_mean, arithmetic_var)
                
                durations = np.maximum(1, np.round(np.random.lognormal(mu, sigma, num_scenarios)))
                
                activities.append({
                    'id': act_id,
                    'patient_id': p_id,
                    'resource_type': self.resources[res_name]['id'],
                    'resource_name': res_name,
                    'durations': durations.tolist(),
                    'mean_duration': arithmetic_mean,
                    'is_start': (step_idx == 0),
                    'scheduled_start': current_arrival_time if step_idx == 0 else 0,
                    'predecessor': previous_act_id
                })
                patient_activities.append(act_id)
                previous_act_id = act_id
            
            patients.append({'id': p_id, 'pathway_type': path_idx, 'activity_ids': patient_activities, 'arrival_time': current_arrival_time})
            
        return {'patients': patients, 'activities': activities, 'resources': self.resources, 'num_scenarios': num_scenarios}

# --- 2. 統一排程提取函式 ---
def extract_schedule_via_sequencing_vars(model, data):
    """
    從模型中提取排程 (適用於 SMILP 和 Deterministic)。
    回傳值：queues (key: global_instance_id, value: sorted activities)
    """
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
            # 無限資源或未指派，找該類型的第一個實例
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
            # 無限資源：FCFS (按排定到達時間)
            queues[inst_id] = sorted(acts, key=lambda x: x['scheduled_start'])
            continue

        # 有限資源：使用 s1 順序
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
        # 使用 arrival time 作為 tie-breaker
        acts_by_arrival = sorted(ids, key=lambda uid: act_map[uid]['scheduled_start'])
        zero_in_degree = deque([uid for uid in acts_by_arrival if in_degree[uid] == 0])
        
        while zero_in_degree:
            u_id = zero_in_degree.popleft()
            sorted_acts.append(act_map[u_id])
            for v_id in adj[u_id]:
                in_degree[v_id] -= 1
                if in_degree[v_id] == 0:
                    zero_in_degree.append(v_id)
        
        # Fallback
        if len(sorted_acts) < len(acts):
            sorted_acts = sorted(acts, key=lambda x: x['scheduled_start'])
            
        queues[inst_id] = sorted_acts
        
    return queues

# --- 3. 模擬器 (Out-of-sample Testing) ---
def simulate_realization(queues, data, random_seed=None):
    if random_seed: np.random.seed(random_seed)
    patients = data['patients']
    activities = data['activities']
    
    # 生成真實執行時間
    realized_durations = {}
    for a in activities:
        mu = a['mean_duration']
        # 假設 CV=1.0 的 lognormal 
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
    
    # 離散事件模擬 (簡易版)
    while len(finished_acts) < num_total:
        progress = False
        
        for inst_id, q in active_queues.items():
            if not q: continue
            
            act = q[0]
            p_id = act['patient_id']
            
            # 檢查 1: 病患是否到達該步驟
            if act['is_start']:
                p_ready = patient_ready_time[p_id]
            else:
                prev_id = act['predecessor']
                if prev_id not in finished_acts: continue # 前置任務未完成
                p_ready = activity_end_times[prev_id]
            
            # 檢查 2: 資源是否空閒
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
            
        if not progress and len(finished_acts) < num_total:
            break # Deadlock check
            
    return total_wait

# --- 4. 主執行流程 ---
def run_experiment():
    print("=== Starting Experiment ===")
    
    # 參數設定
    NUM_PATIENTS = 20
    TRAIN_SCENARIOS = 30 # 用於優化
    TEST_SCENARIOS = 100 # 用於模擬評估
    
    # 1. 生成訓練數據
    gen = InstanceGenerator(num_patients=NUM_PATIENTS)
    train_data = gen.generate_data(num_scenarios=TRAIN_SCENARIOS)
    
    # 2. 求解傳統方法
    print("\n[Traditional] Solving Deterministic Model (Mean Value)...")
    start = time.time()
    m_det = solve_deterministic_model(train_data, time_limit=60)
    print(f"Solved in {time.time()-start:.2f}s")
    sched_det = extract_schedule_via_sequencing_vars(m_det, train_data)
    
    # 3. 求解 SMILP 方法
    print(f"\n[SMILP] Solving Stochastic Model ({TRAIN_SCENARIOS} Scenarios)...")
    start = time.time()
    m_smilp = solve_smilp_model(train_data, time_limit=300, gap=0.05)
    print(f"Solved in {time.time()-start:.2f}s")
    sched_smilp = extract_schedule_via_sequencing_vars(m_smilp, train_data)
    
    # 4. 模擬比較
    print(f"\n[Simulation] Running {TEST_SCENARIOS} out-of-sample tests...")
    wait_det = []
    wait_smilp = []
    
    np.random.seed(999) # 確保模擬的隨機性一致
    for i in range(TEST_SCENARIOS):
        seed = i * 1000
        w_d = simulate_realization(sched_det, train_data, random_seed=seed)
        w_s = simulate_realization(sched_smilp, train_data, random_seed=seed)
        wait_det.append(w_d)
        wait_smilp.append(w_s)
        
    # 5. 結果統計
    avg_d = np.mean(wait_det)
    avg_s = np.mean(wait_smilp)
    improv = (avg_d - avg_s) / avg_d * 100
    
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(f"Traditional (Avg Wait): {avg_d:.2f} min")
    print(f"SMILP       (Avg Wait): {avg_s:.2f} min")
    print(f"Improvement:            {improv:.2f}%")
    print("="*40)
    
    # 簡單繪圖
    plt.figure(figsize=(10, 6))
    plt.boxplot([wait_det, wait_smilp], labels=['Traditional', 'SMILP'], patch_artist=True)
    plt.title(f'Total Waiting Time Comparison (N={NUM_PATIENTS})')
    plt.ylabel('Total Wait Time (min)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    run_experiment()