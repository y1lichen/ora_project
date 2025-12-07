import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from instance_generator import InstanceGenerator
from smilp_model import SMILPSolver
from baseline_model import BaselineSolver

def solve_second_stage_evaluation(first_stage_sol, data, scenario_idx):
    """
    評估函數：固定 x, s1, s2，針對特定場景 n 計算最佳的 b (開始時間)
    這是一個簡單的 LP 問題。
    """
    acts = data['activities']
    x_fixed = first_stage_sol['x']
    s1_fixed = first_stage_sol['s1']
    s2_fixed = first_stage_sol['s2']
    
    m = gp.Model("Eval_SecondStage")
    m.setParam('OutputFlag', 0)
    M = 10000

    b = m.addVars([a['id'] for a in acts], lb=0, vtype=GRB.CONTINUOUS, name="b")
    
    wait_time = 0
    
    # 重建約束 (2b-2g)，但 s1, s2, x 是常數
    for a in acts:
        a_id = a['id']
        duration = a['durations'][scenario_idx]
        
        if a['is_start']:
            sch = a['scheduled_start']
            m.addConstr(b[a_id] >= sch)
            wait_time += (b[a_id] - sch)
        else:
            pred_id = a['predecessor']
            pred_act = next(act for act in acts if act['id'] == pred_id)
            pred_dur = pred_act['durations'][scenario_idx]
            m.addConstr(b[a_id] >= b[pred_id] + pred_dur)
            wait_time += (b[a_id] - b[pred_id] - pred_dur)
            
    # Sequencing constraints (Fixed from stage 1)
    for (pair, val_s1) in s1_fixed.items():
        a_id, a_prime_id = pair
        if pair not in s2_fixed: continue
        val_s2 = s2_fixed[pair]
        
        a_prime_act = next(act for act in acts if act['id'] == a_prime_id)
        d_prime = a_prime_act['durations'][scenario_idx]
        
        # (2d) M * s1 >= b_a - b_a' + 1
        m.addConstr(M * val_s1 >= b[a_id] - b[a_prime_id] + 0.001)
        # (2e) M(1-s1) >= b_a' - b_a
        m.addConstr(M * (1 - val_s1) >= b[a_prime_id] - b[a_id])
        # (2f) M * s2 >= b_a' - b_a + d'
        m.addConstr(M * val_s2 >= b[a_prime_id] - b[a_id] + d_prime)
        # (2g) M(1-s2) >= b_a - b_a' - d' + 1
        m.addConstr(M * (1 - val_s2) >= b[a_id] - b[a_prime_id] - d_prime + 0.001)

    m.setObjective(wait_time, GRB.MINIMIZE)
    m.optimize()
    
    if m.status == GRB.OPTIMAL:
        return m.objVal, {k: v.X for k, v in b.items()}
    else:
        return float('inf'), {}

def visualize_schedule(data, b_sol, x_sol, scenario_idx, title="Schedule"):
    """
    繪製甘特圖
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 資源 Y 軸映射
    res_id_to_name = {v['id']: k for k, v in data['resources'].items()}
    sorted_res_ids = sorted(res_id_to_name.keys())
    
    # 顏色庫 (每個病人一個顏色)
    colors = plt.cm.tab20(np.linspace(0, 1, len(data['patients'])))
    
    for a in data['activities']:
        a_id = a['id']
        p_id = a['patient_id']
        duration = a['durations'][scenario_idx]
        
        if a_id not in b_sol: continue
        start_t = b_sol[a_id]
        
        # 找出分配的資源
        assigned_res = None
        for r_id in data['resources'].values():
            rid = r_id['id']
            if x_sol.get((a_id, rid), 0) > 0.5:
                assigned_res = rid
                break
        
        if assigned_res is not None:
            ax.barh(assigned_res, duration, left=start_t, height=0.5, 
                    color=colors[p_id % 20], edgecolor='black', alpha=0.8)
            # 標註 Activity 名稱和病人 ID
            text_label = f"P{p_id}\n{a['name'][:4]}"
            ax.text(start_t + duration/2, assigned_res, text_label, 
                    ha='center', va='center', color='white', fontsize=7, fontweight='bold')

    ax.set_yticks(sorted_res_ids)
    ax.set_yticklabels([res_id_to_name[i] for i in sorted_res_ids])
    ax.set_xlabel("Time (minutes)")
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close() # 關閉圖表釋放記憶體

def plot_waiting_time_comparison(smilp_waits, baseline_waits):
    """
    繪製 SMILP 與 Baseline 等待時間分佈的比較圖 (Boxplot)
    """
    plt.figure(figsize=(10, 6))
    
    # 準備數據
    data = [smilp_waits, baseline_waits]
    labels = ['SMILP (Stochastic)', 'Baseline (Deterministic)']
    
    # 繪製箱型圖
    bplot = plt.boxplot(data, labels=labels, patch_artist=True, medianprops=dict(color="black"))
    
    # 填充顏色
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    # 加入平均值點
    means = [np.mean(smilp_waits), np.mean(baseline_waits)]
    plt.plot([1, 2], means, 'rs', label='Mean')
    
    # 標題與標籤
    plt.title('Total Waiting Time Distribution Comparison (N=100 Test Scenarios)', fontsize=14)
    plt.ylabel('Total Waiting Time (minutes)', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    # 顯示數值摘要
    text_str = f"SMILP Mean: {means[0]:.1f}\nBaseline Mean: {means[1]:.1f}\nVSS: {means[1]-means[0]:.1f}"
    plt.text(1.5, max(max(smilp_waits), max(baseline_waits)) * 0.95, text_str, 
             horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    filename = "Waiting_Time_Comparison.png"
    plt.savefig(filename)
    print(f"Comparison plot saved as {filename}")
    plt.close()

def main():
    # 參數設定
    N_TRAIN = 30    # SAA 樣本數
    N_TEST = 5000    # 測試樣本數
    NUM_PATIENTS = 20
    
    print(f"1. Generating Training Data (Patients={NUM_PATIENTS}, Scenarios={N_TRAIN})...")
    gen = InstanceGenerator(num_patients=NUM_PATIENTS, arrival_interval=10)
    train_data = gen.generate_data(num_scenarios=N_TRAIN)
    
    print("2. Solving Baseline (Mean Value Model)...")
    baseline = BaselineSolver(train_data, time_limit=300)
    res_baseline = baseline.solve()
    
    print("3. Solving SMILP (Stochastic Model)...")

    smilp = SMILPSolver(train_data, time_limit=3600)
    
    # [重要] 強制 Gurobi 專注於尋找可行解 (利用 Warm Start)
    smilp.model.setParam('MIPFocus', 1) 
    
    res_smilp = smilp.build_and_solve()
    if not res_smilp or not res_baseline:
        print("Optimization failed.")
        return

    print(f"4. Generating Test Data (Scenarios={N_TEST})...")
    test_data = train_data.copy()
    test_data['num_scenarios'] = N_TEST
    
    # 對每個 Activity 重新採樣 (Out-of-sample testing)
    # [建議] 這裡可以手動放大變異數來測試 SMILP 的魯棒性
    for a in test_data['activities']:
        mu, sigma = gen._convert_params_to_lognormal(a['mean_duration'], a['var_duration'])
        # sigma = sigma * 1.5 # (Optional) Uncomment to stress test
        new_samples = np.maximum(1, np.round(np.random.lognormal(mu, sigma, N_TEST)))
        a['durations'] = new_samples.tolist()

    print("5. Evaluating Performance...")
    smilp_waits = []
    baseline_waits = []
    
    for n in range(N_TEST):
        val_s, b_s = solve_second_stage_evaluation(res_smilp, test_data, n)
        smilp_waits.append(val_s)
        
        val_b, b_b = solve_second_stage_evaluation(res_baseline, test_data, n)
        baseline_waits.append(val_b)
        
        if n == 0:
            print("   Visualizing Scenario 0 (Gantt Charts)...")
            visualize_schedule(test_data, b_s, res_smilp['x'], 0, "SMILP Schedule (Test Scenario 0)")
            visualize_schedule(test_data, b_b, res_baseline['x'], 0, "Baseline Schedule (Test Scenario 0)")

    # 繪製等待時間比較圖
    print("   Plotting Waiting Time Comparison...")
    plot_waiting_time_comparison(smilp_waits, baseline_waits)

    avg_smilp = np.mean(smilp_waits)
    avg_base = np.mean(baseline_waits)
    vss = avg_base - avg_smilp
    
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(f"Avg Wait Time (SMILP):    {avg_smilp:.2f} mins")
    print(f"Avg Wait Time (Baseline): {avg_base:.2f} mins")
    print(f"Value of Stochastic Solution (VSS): {vss:.2f} mins")
    
    if avg_base > 0:
        print(f"Improvement: {(vss/avg_base)*100:.2f}%")
    else:
        print("Improvement: N/A (Baseline wait time is 0)")

if __name__ == "__main__":
    main()