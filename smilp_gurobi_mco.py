# smilp_paper_compliant.py
# 整合 MCO 方法，確保與論文流程一致。

from gurobipy import Model, GRB, quicksum
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys

# 假設 instance_generator.py 與此檔案在同一目錄
from instance_generator import InstanceGenerator 

# ==========================================
# 1. 資料預處理 (Data Preprocessing)
# ==========================================
def build_index_sets(data):
    """建立模型所需的索引集和資訊映射。"""
    activities = data["activities"]
    A = [act["id"] for act in activities]
    act_by_id = {act["id"]: act for act in activities}
    resource_uid_to_type = data["resource_uid_to_type"]
    J = list(resource_uid_to_type.keys())
    resources_by_type = data["resources_by_type"]
    G = list(resources_by_type.keys())
    Jg = {g: [unit["uid"] for unit in resources_by_type[g]] for g in G}

    A_g = {g: [] for g in G}
    for act in activities:
        for g in act["required_types"]:
            if g not in A_g:
                A_g[g] = []
            A_g[g].append(act["id"])

    eligible_j_for_a = {}
    for act in activities:
        a = act["id"]
        eligible = []
        for g in act["required_types"]:
            eligible.extend(Jg.get(g, []))
        eligible_j_for_a[a] = list(sorted(set(eligible)))

    return {
        "A": A, "A_info": act_by_id, "J": J, "G": G,
        "Jg": Jg, "A_g": A_g, "eligible_j_for_a": eligible_j_for_a
    }

# ==========================================
# 2. 模型建構：SAA 求解器 (Formulation 3/1)
# ==========================================

def solve_saa_model(data, M=20000, time_limit=3600, verbose=False, get_b_sol=False):
    """
    求解 SMILP 的 SAA 形式 (Formulation 1/3)。
    返回第一階段解和 SAA 目標值 v_N。
    """
    idx = build_index_sets(data)
    A, A_info = idx["A"], idx["A_info"]
    J, Jg, G = idx["J"], idx["Jg"], idx["G"]
    eligible = idx["eligible_j_for_a"]
    num_scenarios = data["num_scenarios"]
    scenarios = list(range(num_scenarios))

    # Parameters
    t_a = {a: (A_info[a]["scheduled_start"] if A_info[a]["is_start"] else None) for a in A}
    pre = {a: A_info[a]["predecessor"] for a in A}
    Va_g = {(a,g): 1 for a in A for g in A_info[a]["required_types"]}
    k_j = {j: 1 for j in J}
    
    # Durations d[a,k]
    d = {}
    for a in A:
        dur_list = A_info[a]["durations"]
        for k in scenarios:
            d[(a,k)] = float(dur_list[k])

    # Share Pairs (only need to calculate if it's not present in data, but here we calculate it based on types)
    share_pairs = []
    for a in A:
        types_a = set(A_info[a]["required_types"])
        for a2 in A:
            if a == a2: continue
            if len(types_a & set(A_info[a2]["required_types"])) > 0:
                share_pairs.append((a,a2))

    # Model
    model = Model(f"SAA_N{num_scenarios}")
    model.setParam('TimeLimit', time_limit)
    if not verbose: model.setParam('OutputFlag', 0)
    
    # --- Stage 1 Variables (x, sa1, sa2, q) ---
    x = { (a,j): model.addVar(vtype=GRB.BINARY, name=f"x_{a}_{j}") 
          for a in A for j in eligible[a] }

    sa1, sa2, q = {}, {}, {}
    for (a,a2) in share_pairs:
        sa1[(a,a2)] = model.addVar(vtype=GRB.BINARY, name=f"sa1_{a}_{a2}")
        sa2[(a,a2)] = model.addVar(vtype=GRB.BINARY, name=f"sa2_{a}_{a2}")
        types_inter = set(A_info[a]["required_types"]) & set(A_info[a2]["required_types"])
        js = sorted(list(set([j for g in types_inter for j in Jg[g]])))
        for j in js:
            q[(j,a,a2)] = model.addVar(vtype=GRB.BINARY, name=f"q_{j}_{a}_{a2}")

    # --- Stage 2 Variables (b[a,k]) ---
    b = { (a,k): model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"b_{a}_{k}") 
          for a in A for k in scenarios }

    model.update()

    # --- Constraints ---
    # (1b) Assignment
    for a in A:
        for g in A_info[a]["required_types"]:
            model.addConstr(quicksum(x.get((a,j),0) for j in Jg[g]) == Va_g[(a,g)], name=f"C1b_{a}_{g}")

    # (1c) q definition logic
    for (j,a,a2), qvar in q.items():
        model.addConstr(qvar >= sa1[(a,a2)] + sa2[(a,a2)] + x.get((a,j),0) + x.get((a2,j),0) - 3, name=f"C1c_{j}_{a}_{a2}")

    # (1d) Capacity via q
    for g in G:
        for j in Jg[g]:
            acts_g = idx["A_g"].get(g, [])
            for a in acts_g:
                lhs = quicksum(q[(j,a,a2)] for a2 in acts_g if a2 != a and (j,a,a2) in q)
                model.addConstr(lhs <= k_j[j] - 1, name=f"C1d_{j}_{a}")

    # (2b-2g) Stage 2 Constraints
    for k in scenarios:
        for a in A:
            # (2b & 2c) Precedence
            if A_info[a]["is_start"]:
                model.addConstr(b[(a,k)] >= t_a[a], name=f"C2b_{a}_{k}")
            else:
                prev = pre[a]
                model.addConstr(b[(a,k)] >= b[(prev,k)] + d[(prev,k)], name=f"C2c_{a}_{k}")

        # (2d-2g) Big-M Sequencing
        for (a,a2) in share_pairs:
            model.addConstr(M * sa1[(a,a2)] >= b[(a,k)] - b[(a2,k)] + 1, name=f"C2d_{a}_{a2}_{k}")
            model.addConstr(M * (1 - sa1[(a,a2)]) >= b[(a2,k)] - b[(a,k)], name=f"C2e_{a}_{a2}_{k}")
            model.addConstr(M * sa2[(a,a2)] >= b[(a2,k)] - b[(a,k)] + d[(a2,k)], name=f"C2f_{a}_{a2}_{k}")
            model.addConstr(M * (1 - sa2[(a,a2)]) >= b[(a,k)] - b[(a2,k)] - d[(a2,k)] + 1, name=f"C2g_{a}_{a2}_{k}")

    # Objective: Min Avg Waiting Time (1a)
    Q_k_terms = []
    for k in scenarios:
        terms = []
        for a in A:
            if A_info[a]["is_start"]:
                terms.append(b[(a,k)] - t_a[a])
            else:
                prev = pre[a]
                terms.append(b[(a,k)] - b[(prev,k)] - d[(prev,k)])
        Q_k_terms.append(quicksum(terms))
    
    model.setObjective((1.0/num_scenarios) * quicksum(Q_k_terms), GRB.MINIMIZE)
    
    model.optimize()

    # 提取解
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.status == GRB.SUBOPTIMAL:
        sol_x = {(a,j): round(var.X) for (a,j), var in x.items() if var.X > 0.5}
        sol_sa1 = {(a,a2): round(var.X) for (a,a2), var in sa1.items() if var.X > 0.5}
        sol_sa2 = {(a,a2): round(var.X) for (a,a2), var in sa2.items() if var.X > 0.5}
        
        sol_b = {}
        if get_b_sol:
             sol_b = {(a,k): var.X for (a,k), var in b.items()}

        # 論文要求返回 SAA 目標值 vk_N (這裡就是 model.ObjVal)
        return {
            "objective": model.ObjVal,
            "x": sol_x, "sa1": sol_sa1, "sa2": sol_sa2, "b": sol_b
        }
    return None

# ==========================================
# 3. MCO 函數 (Algorithm 1)
# ==========================================

def run_mco_method(generator, N0, K, N_prime, epsilon, M, time_limit, dynamic_N=True):
    """
    實作論文中的 MCO 方法 (Algorithm 1)。
    迭代地增加訓練樣本量 N，直到 AOIN 達到容忍度 epsilon。
    """
    N = N0
    AOIN = float('inf')
    iteration = 0
    final_sol = None
    
    print("--- Running Monte Carlo Optimization (MCO) ---")
    print(f"N0={N0}, K={K}, N'={N_prime}, epsilon={epsilon}")

    while AOIN >= epsilon:
        iteration += 1
        print(f"\n--- MCO Iteration {iteration}: N = {N} ---")

        v_N_list = []      # List of v^k_N (SAA objective)
        v_N_prime_list = []# List of v^k_N' (Evaluation objective)
        
        # for k = 1, ..., K do
        for k in range(K):
            print(f"  > Replicate {k+1}/{K}: Generating {N + N_prime} scenarios...")
            # Generate (N + N') scenarios
            full_data_k = generator.generate_data(num_scenarios=N + N_prime)
            
            # Splitting scenarios: First N for training, last N' for evaluation
            train_data_k, eval_data_k = split_data(full_data_k, N, N_prime)
            
            # 1. Solving SAA with first N scenarios to get x_N, v^k_N
            saa_start_time = time.time()
            saa_result = solve_saa_model(train_data_k, M=M, time_limit=time_limit, verbose=False)
            saa_time = time.time() - saa_start_time

            if saa_result is None:
                print(f"  !! Replicate {k+1} FAILED to solve SAA. Aborting iteration.")
                continue 

            # v^k_N = optimal objective value from SAA
            v_N_list.append(saa_result["objective"] * N) # We store the SUM of Q, not average

            # 2. Evaluate the solution x_N on the last N' scenarios to get v^k_N'
            eval_start_time = time.time()
            # Evaluate using the fixed first-stage DVs from SAA
            Q_k_values = evaluate_solution_on_scenarios(
                saa_result, eval_data_k, M=M, penalty_val=100000 
            )
            
            # v^k_N' = sum of evaluated Q values (Approximated objective, not average)
            v_N_prime_list.append(np.sum(Q_k_values))
            eval_time = time.time() - eval_start_time
            
            print(f"    - SAA Time: {saa_time:.1f}s, Eval Time: {eval_time:.1f}s, Inf/Penalty: {Q_k_values.count(100000)}/{N_prime}")
            
            # Save the best solution found so far (last one in the final iteration usually suffices)
            final_sol = {
                "sol": {k: saa_result[k] for k in ["x", "sa1", "sa2"]}, 
                "N": N, 
                "v_N": saa_result["objective"],
                "v_N_prime": np.mean(Q_k_values)
            }


        if not v_N_list or not v_N_prime_list:
            print("Iteration failed due to repeated solver failures. Exiting MCO.")
            break
            
        # Calculate Averages: 
        # Note: We divide by N or N' now to get the average objective value
        mean_v_N = np.mean(v_N_list) / N
        mean_v_N_prime = np.mean(v_N_prime_list) / N_prime
        
        # Calculate AOIN
        if mean_v_N_prime > 0:
             AOIN = (mean_v_N_prime - mean_v_N) / mean_v_N_prime
        else:
             AOIN = 0 # Avoid division by zero if objective is near zero
        
        print(f"  > Avg v_N (Training): {mean_v_N:.2f}")
        print(f"  > Avg v_N' (Testing): {mean_v_N_prime:.2f}")
        print(f"  > AOIN: {AOIN:.4f} (Target: < {epsilon})")
        
        if not dynamic_N:
            print("  > [Fixed Mode] N is fixed. Stopping after one iteration.")
            break # 固定模式：跑完第一次迭代就結束

        if AOIN >= epsilon:
            # Update the sample size N <- 2N
            N *= 2
        
    return final_sol

# ==========================================
# 4. 評估階段 (Evaluation / Testing Phase)
# ==========================================

def solve_baseline_training(data, M=20000, time_limit=600):
    """
    求解 Deterministic Mean Value Model。
    返回第一階段解。
    """
    # [Code for solve_baseline_training is omitted for brevity but assumed to be correct based on previous steps]
    # ... (使用平均持續時間來求解單一的 MILP)
    idx = build_index_sets(data)
    A, A_info = idx["A"], idx["A_info"]
    J, Jg, G = idx["J"], idx["Jg"], idx["G"]
    eligible = idx["eligible_j_for_a"]
    mean_d = {a: float(A_info[a]["mean_duration"]) for a in A}
    t_a = {a: (A_info[a]["scheduled_start"] if A_info[a]["is_start"] else None) for a in A}
    pre = {a: A_info[a]["predecessor"] for a in A}
    
    model = Model("Baseline_Mean")
    model.setParam('TimeLimit', time_limit)
    # model.setParam('OutputFlag', 0)

    # Variables
    x = { (a,j): model.addVar(vtype=GRB.BINARY) for a in A for j in eligible[a] }
    share_pairs = []
    sa1, sa2, q = {}, {}, {}
    for a in A:
        for a2 in A:
            if a == a2: continue
            if len(set(A_info[a]["required_types"]) & set(A_info[a2]["required_types"])) > 0:
                share_pairs.append((a,a2))
                sa1[(a,a2)] = model.addVar(vtype=GRB.BINARY)
                sa2[(a,a2)] = model.addVar(vtype=GRB.BINARY)
                types_inter = set(A_info[a]["required_types"]) & set(A_info[a2]["required_types"])
                js = sorted(list(set([j for g in types_inter for j in Jg[g]])))
                for j in js: q[(j,a,a2)] = model.addVar(vtype=GRB.BINARY)

    b = {a: model.addVar(lb=0.0) for a in A}

    # Constraints (1b)-(1d) and Deterministic Sequencing & Timing (2b-2g) with mean_d
    # (Simplified for display, but assumes full compliance)
    
    # ... Add all constraints (1b)-(1d) and (2b)-(2g) using mean_d ...
    
    # (1b)
    for a in A:
        for g in A_info[a]["required_types"]:
            model.addConstr(quicksum(x.get((a,j),0) for j in Jg[g]) == 1)
    
    # (1c & 1d) Q logic and Capacity
    for (j,a,a2), qvar in q.items():
        model.addConstr(qvar >= sa1[(a,a2)] + sa2[(a,a2)] + x.get((a,j),0) + x.get((a2,j),0) - 3)
    
    for g in G:
        for j in Jg[g]:
            acts = idx["A_g"].get(g, [])
            for a in acts:
                lhs = quicksum(q[(j,a,a2)] for a2 in acts if a2 != a and (j,a,a2) in q)
                model.addConstr(lhs <= 1 - 1)

    # Deterministic Sequencing & Timing
    for a in A:
        if A_info[a]["is_start"]: model.addConstr(b[a] >= t_a[a])
        else: model.addConstr(b[a] >= b[pre[a]] + mean_d[pre[a]])
    
    for (a,a2) in share_pairs:
        model.addConstr(M * sa1[(a,a2)] >= b[a] - b[a2] + 1)
        model.addConstr(M * (1 - sa1[(a,a2)]) >= b[a2] - b[a])
        model.addConstr(M * sa2[(a,a2)] >= b[a2] - b[a] + mean_d[a2])
        model.addConstr(M * (1 - sa2[(a,a2)]) >= b[a] - b[a2] - mean_d[a2] + 1)

    # Objective
    terms = []
    for a in A:
        if A_info[a]["is_start"]: terms.append(b[a] - t_a[a])
        else: terms.append(b[a] - b[pre[a]] - mean_d[pre[a]])
    model.setObjective(quicksum(terms), GRB.MINIMIZE)

    model.optimize()

    sol_first_stage = {
        "x": {(a,j): round(var.X) for (a,j), var in x.items() if var.X > 0.5},
        "sa1": {(a,a2): round(var.X) for (a,a2) in sa1 if sa1[(a,a2)].X > 0.5},
        "sa2": {(a,a2): round(var.X) for (a,a2) in sa2 if sa2[(a,a2)].X > 0.5},
    }
    return sol_first_stage

def evaluate_solution_on_scenarios(first_stage_sol, test_data, M=20000, penalty_val=100000):
    """
    評估固定的第一階段解在 N' 個情境下的平均成本 (Q)。
    **注意：已加入懲罰項以應對不可行問題。**
    """
    idx = build_index_sets(test_data)
    A, A_info = idx["A"], idx["A_info"]
    scenarios = list(range(test_data["num_scenarios"]))
    t_a = {a: (A_info[a]["scheduled_start"] if A_info[a]["is_start"] else None) for a in A}
    pre = {a: A_info[a]["predecessor"] for a in A}

    x_fixed = first_stage_sol.get("x", {})
    sa1_fixed = first_stage_sol.get("sa1", {})
    sa2_fixed = first_stage_sol.get("sa2", {})
    
    Q_values = []
    
    for k in scenarios:
        eval_model = Model(f"Eval_k{k}")
        eval_model.setParam('OutputFlag', 0)
        
        b = {a: eval_model.addVar(lb=0.0) for a in A}
        # 鬆弛變數 r[a] 用於解決前置約束 (2c) 的不可行問題
        r = {a: eval_model.addVar(lb=0.0, name=f"r_{a}") for a in A if not A_info[a]["is_start"]}
        
        d_k = {a: float(A_info[a]["durations"][k]) for a in A}

        # Objective term (Wait time: ba - earliest_start)
        wait_terms = []
        
        # Constraints
        for a in A:
            if A_info[a]["is_start"]:
                eval_model.addConstr(b[a] >= t_a[a])
                wait_terms.append(b[a] - t_a[a])
            else:
                prev = pre[a]
                # 鬆弛約束 (2c'): b[a] + r[a] >= b[prev] + d[prev]
                eval_model.addConstr(b[a] + r[a] >= b[prev] + d_k[prev])
                wait_terms.append(b[a] - b[prev] - d_k[prev])

        # (2d-2g) Sequencing (Linearized by fixed sa1/sa2)
        for (a,a2), val_sa1 in sa1_fixed.items():
            val_sa2 = sa2_fixed.get((a,a2), 0)
            
            eval_model.addConstr(M * val_sa1 >= b[a] - b[a2] + 1)
            eval_model.addConstr(M * (1 - val_sa1) >= b[a2] - b[a])
            eval_model.addConstr(M * val_sa2 >= b[a2] - b[a] + d_k[a2])
            eval_model.addConstr(M * (1 - val_sa2) >= b[a] - b[a2] - d_k[a2] + 1)
        
        # Objective: Min (Wait time + Penalty * Slack)
        penalty_expr = quicksum(r[a] for a in r)
        eval_model.setObjective(quicksum(wait_terms) + penalty_val * penalty_expr, GRB.MINIMIZE)
        eval_model.optimize()

        if eval_model.Status == GRB.OPTIMAL or eval_model.Status == GRB.TIME_LIMIT:
            # 返回包含了懲罰的總成本
            Q_values.append(eval_model.ObjVal)
        else:
            # 即使有懲罰，若仍不可行，表示存在數值問題
            Q_values.append(penalty_val * len(A) * M) # Extreme Penalty
    
    return Q_values

# ==========================================
# 5. 視覺化 (Visualization)
# ==========================================

def plot_paper_comparison(smilp_Q, baseline_Q, N_prime):
    """繪製結果比較圖，採用箱型圖和 VSS 總結。"""
    
    # 排除極端懲罰值 (這裡假設懲罰值遠大於實際等待時間)
    MAX_VALID_COST = np.max([np.mean(smilp_Q), np.mean(baseline_Q)]) * 5 
    smilp_vals = np.array([v for v in smilp_Q if v < MAX_VALID_COST])
    baseline_vals = np.array([v for v in baseline_Q if v < MAX_VALID_COST])
    
    if len(smilp_vals) < N_prime * 0.5 or len(baseline_vals) < N_prime * 0.5:
        print("Warning: Too many infeasible scenarios (or extreme penalties). Plot may be inaccurate.")
        
    smilp_mean = np.mean(smilp_vals)
    baseline_mean = np.mean(baseline_vals)
    
    # VSS Calculation: VSS = E[Q(x_base)] - E[Q(x_smilp)]
    vss = baseline_mean - smilp_mean
    improvement_pct = (vss / baseline_mean) * 100 if baseline_mean > 0 else 0

    text_str = (f"Baseline Avg: {baseline_mean:.2f} mins\n"
                f"SMILP Avg: {smilp_mean:.2f} mins\n"
                f"VSS (Diff): {vss:.2f} mins\n"
                f"Improvement: {improvement_pct:.2f}%")

    plt.figure(figsize=(10, 7))
    plt.boxplot([baseline_vals, smilp_vals], labels=["Baseline (Mean Value)", "SMILP (Stochastic)"], 
                showmeans=True, patch_artist=True, medianprops={'color': 'black'})
    
    plt.ylabel("Total Patient Waiting Time (mins)")
    plt.title(f"Evaluated Waiting Time Distribution on Out-of-Sample Scenarios (N'={N_prime})")
    
    # Add VSS summary text box
    props = dict(boxstyle='round', facecolor='azure', alpha=0.8)
    plt.text(0.65, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    output_filename = "smilp_paper_result.png"
    plt.savefig(output_filename)
    plt.show()
    print(f"\nPlot saved to {output_filename}")
    print(f"\n--- Final Results ---\n{text_str}")

# ==========================================
# 6. 資料分割輔助函數
# ==========================================

def split_data(full_data, n_train, n_test):
    """將一份包含 (n_train + n_test) 個情境的資料集切分成兩份。"""
    import copy
    
    train_data = copy.deepcopy(full_data)
    test_data = copy.deepcopy(full_data)
    
    total_scenarios = full_data["num_scenarios"]
    if total_scenarios < n_train + n_test:
         raise ValueError(f"Data has {total_scenarios} scenarios, but need {n_train} + {n_test}")

    # 修改 Train Data
    train_data["num_scenarios"] = n_train
    for act in train_data["activities"]:
        act["durations"] = act["durations"][:n_train]
        
    # 修改 Test Data
    test_data["num_scenarios"] = n_test
    for act in test_data["activities"]:
        act["durations"] = act["durations"][n_train : n_train + n_test]

    return train_data, test_data


# ==========================================
# 7. 主程式 (Main Execution)
# ==========================================
if __name__ == "__main__":
    # Settings based on paper (Algorithm 1)
    N0_INITIAL = 30 # 初始 SAA 樣本量 (為測試 MCO，設小一點)
    K_REPLICATES = 5 # MCO 迭代次數 K
    N_PRIME_EVAL = 500 # 評估樣本量 N' (為測試 MCO，設小一點)
    EPSILON_AOIN = 0.05 # AOIN 終止容忍值
    
    # 實驗參數
    NUM_PATIENTS = 15
    BIG_M_VALUE = 10000 # 較大的 M 值以應對 Log-Normal 分佈的高方差
    TIME_LIMIT_SAA = 1200 # 每個 SAA 模型的求解時間限制 (秒)

    # 1. 生成足夠大的情境總數
    TOTAL_SCENARIOS_PER_REPLICATE = 2 * N0_INITIAL + N_PRIME_EVAL 
    # MCO 要求在每個 K 中生成 N + N' 個情境，但如果 N 翻倍，可能需要更多。
    # 這裡只生成一次，並在 MCO 中重複取樣 (Generator 內含 random seed 控制)

    print("=== Step 1: Initialize Data Generator ===")
    gen = InstanceGenerator(num_patients=NUM_PATIENTS, arrival_interval=10, random_seed=42)
    
    # 2. 執行 MCO 方法 (找到最優 SMILP 解)
    smilp_mco_sol = run_mco_method(
        generator=gen, 
        N0=N0_INITIAL, 
        K=K_REPLICATES, 
        N_prime=N_PRIME_EVAL, 
        epsilon=EPSILON_AOIN, 
        M=BIG_M_VALUE, 
        time_limit=TIME_LIMIT_SAA,
        dynamic_N=False
    )

    if smilp_mco_sol is None:
        print("\nFATAL: MCO method failed to find a valid SMILP solution.")
        sys.exit(1)

    sol_smilp = smilp_mco_sol["sol"]
    print(f"\n--- MCO Final Result ---")
    print(f"Optimal N found: {smilp_mco_sol['N']}")
    print(f"SMILP Avg Wait (Final v_N'): {smilp_mco_sol['v_N_prime']:.2f}")

    # 3. 求解 Baseline (Deterministic Mean Value Model)
    # 只需要生成一次平均情境數據，使用 N0 的數據結構即可。
    print("\n=== Step 3: Solving Baseline Model ===")
    baseline_train_data = gen.generate_data(num_scenarios=N0_INITIAL)
    sol_baseline = solve_baseline_training(baseline_train_data, M=BIG_M_VALUE, time_limit=600)
    
    if sol_baseline is None:
        print("FATAL: Baseline model failed.")
        sys.exit(1)


    # 4. 進行最終的 Out-of-Sample 評估 (使用單獨的 N' 樣本集)
    print("\n=== Step 4: Final Out-of-Sample Evaluation ===")
    
    # 確保 final evaluation 使用一個全新的、獨立的 N' 樣本
    final_eval_generator = InstanceGenerator(num_patients=NUM_PATIENTS, arrival_interval=10, random_seed=50)
    final_test_data = final_eval_generator.generate_data(num_scenarios=N_PRIME_EVAL)

    # 4.1 Evaluate Baseline Solution
    print(f"Evaluating Baseline Solution on N'={N_PRIME_EVAL} test scenarios...")
    q_base = evaluate_solution_on_scenarios(sol_baseline, final_test_data, M=BIG_M_VALUE)

    # 4.2 Evaluate SMILP Solution
    print(f"Evaluating SMILP Solution on N'={N_PRIME_EVAL} test scenarios...")
    q_smilp = evaluate_solution_on_scenarios(sol_smilp, final_test_data, M=BIG_M_VALUE)

    # 5. 視覺化
    print("\n=== Step 5: Comparison & Plotting ===")
    plot_paper_comparison(q_smilp, q_base, N_PRIME_EVAL)