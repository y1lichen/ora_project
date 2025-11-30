import matplotlib.pyplot as plt
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from instance_generator import InstanceGenerator
from traditional_method import solve_deterministic_model
from smilp_model import solve_smilp_model

# 論文使用 N=100 (訓練) 進行 MCO
# NUM_PATIENTS = 10     # 建議 10-20
NUM_PATIENTS = 20
# TRAIN_SCENARIOS = 30  # SAA 訓練場景數 (論文建議 30-100)
TRAIN_SCENARIOS = 20
TEST_SCENARIOS = 100  # 模擬測試場景數 (Out-of-sample)

def extract_sequence_from_model(model, data):
    activities = data['activities']
    b_vals = {}
    for v in model.getVars():
        if v.VarName.startswith("b["):
            parts = v.VarName.split('[')[1].split(']')[0].split(',')
            act_id = int(parts[0])
            if act_id not in b_vals: b_vals[act_id] = []
            b_vals[act_id].append(v.X)
    
    avg_start_times = {k: np.mean(v) for k, v in b_vals.items()}
    resource_acts = {r_id: [] for r_id in [res['id'] for res in data['resources'].values()]}
    
    for a in activities:
        r_id = a['resource_type']
        if a['id'] in avg_start_times:
            resource_acts[r_id].append((avg_start_times[a['id']], a))
            
    final_sequences = {}
    for r_name, r_info in data['resources'].items():
        r_id = r_info['id']
        sorted_acts = sorted(resource_acts[r_id], key=lambda x: x[0])
        final_sequences[r_id] = [x[1] for x in sorted_acts]
    return final_sequences

def simulate_realization(sequence, data, random_seed=None):
    """模擬真實執行情況 (Out-of-Sample Test)"""
    if random_seed: np.random.seed(random_seed)
    resources = data['resources']
    activities = data['activities']
    patients = data['patients']
    
    
    realized_durations = {}

    for a in activities:
        mu_mins = a['mean_duration']
        sigma = 0.5 # 這可能要調整
        real_mu = np.log(mu_mins) - (sigma**2)/2
        dur = np.random.lognormal(real_mu, sigma)
        realized_durations[a['id']] = max(1, round(dur))
        
    resource_availability = {} 
    for r_name, r_info in resources.items():
        resource_availability[r_info['id']] = [0] * r_info['capacity']
    patient_availability = {p['id']: p['arrival_time'] for p in patients}
    
    activity_start_times = {}
    activity_end_times = {}
    total_wait_time = 0
    
    queues = {r_id: [act for act in seq] for r_id, seq in sequence.items()}
    finished_acts = set()
    num_total_acts = len(activities)
    
    while len(finished_acts) < num_total_acts:
        progress_made = False
        for r_id, queue in queues.items():
            if not queue: continue
            act = queue[0]
            p_id = act['patient_id']
            
            patient_ready_time = patient_availability[p_id]
            if not act['is_start']:
                prev_id = act['predecessor']
                if prev_id not in finished_acts: continue
            
            # 資源是否有空
            res_channels = resource_availability[r_id]
            earliest_channel_idx = np.argmin(res_channels)
            res_ready_time = res_channels[earliest_channel_idx]
            
            # 開始
            start_t = max(patient_ready_time, res_ready_time)
            duration = realized_durations[act['id']]
            end_t = start_t + duration
            
            activity_start_times[act['id']] = start_t
            activity_end_times[act['id']] = end_t
            wait = start_t - patient_ready_time
            total_wait_time += wait
            
            resource_availability[r_id][earliest_channel_idx] = end_t
            patient_availability[p_id] = end_t
            finished_acts.add(act['id'])
            queue.pop(0)
            progress_made = True
        if not progress_made and len(finished_acts) < num_total_acts: break
            
    return total_wait_time, activity_start_times, activity_end_times, realized_durations

def run_experiment_and_plot():
    gen = InstanceGenerator(num_patients=NUM_PATIENTS, random_seed=42) 
    
    # SMILP 用多場景訓練
    data_smilp = gen.generate_data(num_scenarios=TRAIN_SCENARIOS)
    
    # 傳統方法
    data_det = gen.generate_data(num_scenarios=1) 
    
    model_det = solve_deterministic_model(data_det)
    if not model_det or model_det.Status != GRB.OPTIMAL:
        print("error")
        return

    model_smilp = solve_smilp_model(data_smilp)
    if not model_smilp or model_smilp.Status != GRB.OPTIMAL:
        print("SMILP error")
        return

    seq_det = extract_sequence_from_model(model_det, data_det)
    seq_smilp = extract_sequence_from_model(model_smilp, data_smilp)
    
    wait_times_det, wait_times_smilp = [], []
    last_sim_det, last_sim_smilp = None, None
    
    np.random.seed(999) 
    for i in range(TEST_SCENARIOS):
        seed = i * 100
        wt_d, start_d, end_d, dur_d = simulate_realization(seq_det, data_det, random_seed=seed)
        wt_s, start_s, end_s, dur_s = simulate_realization(seq_smilp, data_smilp, random_seed=seed)
        wait_times_det.append(wt_d)
        wait_times_smilp.append(wt_s)
        if i == 0: 
            last_sim_det = (start_d, end_d, dur_d)
            last_sim_smilp = (start_s, end_s, dur_s)

    # 繪圖
    fig = plt.figure(figsize=(16, 12))
    ax1 = plt.subplot(2, 1, 1)
    box = ax1.boxplot([wait_times_det, wait_times_smilp], patch_artist=True, vert=False, 
                      labels=['Traditional (Deterministic)', 'Innovative (SMILP)'])
    colors = ['#ff9999', '#66b3ff']
    for patch, color in zip(box['boxes'], colors): patch.set_facecolor(color)
        
    mean_det = np.mean(wait_times_det)
    mean_smilp = np.mean(wait_times_smilp)
    improvement = (mean_det - mean_smilp) / mean_det * 100
    
    print(f"\n結果摘要:")
    print(f"傳統方法平均等待時間: {mean_det:.2f} min")
    print(f"SMILP方法平均等待時間: {mean_smilp:.2f} min")
    print(f"改善幅度: {improvement:.2f}%")
    
    ax1.set_title(f'Total Waiting Time Comparison (Improvement: {improvement:.1f}%)', fontsize=14)
    ax1.set_xlabel('Total Waiting Time (Minutes)')
    
    ax2 = plt.subplot(2, 1, 2)
    def plot_gantt(ax, sim_data, resources, activities, offset_y, title):
        start, end, _ = sim_data
        cmap = plt.get_cmap('tab20')
        patient_colors = {i: cmap(i % 20) for i in range(50)}
        y_ticks, y_labels = [], []
        for idx, r_name in enumerate(resources.keys()):
            r_id = resources[r_name]['id']
            y_pos = offset_y + idx * 10
            y_ticks.append(y_pos)
            y_labels.append(f"{title}\n{r_name}")
            for act in activities:
                if act['resource_type'] == r_id and act['id'] in start:
                    st, dur = start[act['id']], end[act['id']] - start[act['id']]
                    ax.barh(y_pos, dur, left=st, height=8, color=patient_colors[act['patient_id']], edgecolor='black')
                    ax.text(st + dur/2, y_pos, f"P{act['patient_id']}", ha='center', va='center', color='white', fontsize=8)
        return y_ticks, y_labels

    yt_d, yl_d = plot_gantt(ax2, last_sim_det, data_det['resources'], data_det['activities'], 80, "[Traditional]")
    yt_s, yl_s = plot_gantt(ax2, last_sim_smilp, data_smilp['resources'], data_smilp['activities'], 0, "[SMILP]")
    ax2.set_yticks(yt_s + yt_d)
    ax2.set_yticklabels(yl_s + yl_d)
    ax2.set_xlabel('Time (Minutes)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment_and_plot()