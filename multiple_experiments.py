# smilp_paper_compliant.py
# Requires gurobipy installed and a valid Gurobi license.
from gurobipy import Model, GRB, quicksum
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
# 假設 instance_generator.py 與此檔案在同一目錄
from instance_generator import InstanceGenerator
import matplotlib.patches as mpatches

def get_schedule_for_scenario(first_stage_sol, data, k, M=1000, time_limit=60):
    """
    對單一情境 k 建 LP（b[a] 為變數），並以 first_stage_sol["x"], sa1, sa2 當常數
    回傳 schedule: list of dicts [{'activity': a, 'start': s, 'dur': d, 'resource': j}, ...]
    """
    idx = build_index_sets(data)
    A, A_info = idx["A"], idx["A_info"]
    scenarios = list(range(data["num_scenarios"]))
    if k not in scenarios:
        raise ValueError(f"Scenario k={k} not in data (num_scenarios={data['num_scenarios']})")
    
    # fixed decisions
    x_fixed = first_stage_sol["x"]
    sa1_fixed = first_stage_sol.get("sa1", {})
    sa2_fixed = first_stage_sol.get("sa2", {})

    # Build small eval model
    eval_model = Model(f"Gantt_Eval_k{k}")
    eval_model.setParam('OutputFlag', 0)
    if time_limit: eval_model.setParam('TimeLimit', time_limit)

    # b variables
    b = {a: eval_model.addVar(lb=0.0, name=f"b_{a}") for a in A}
    eval_model.update()

    # durations for this scenario
    d_k = {a: float(A_info[a]["durations"][k]) for a in A}
    t_a = {a: (A_info[a]["scheduled_start"] if A_info[a]["is_start"] else None) for a in A}
    pre = {a: A_info[a]["predecessor"] for a in A}

    # precedence
    for a in A:
        if A_info[a]["is_start"]:
            eval_model.addConstr(b[a] >= t_a[a])
        else:
            prev = pre[a]
            eval_model.addConstr(b[a] >= b[prev] + d_k[prev])

    # sequencing constraints using fixed sa1/sa2
    for (a,a2), val_sa1 in sa1_fixed.items():
        val_sa2 = sa2_fixed.get((a,a2), 0)
        # linearized big-M with constants
        eval_model.addConstr(M * val_sa1 >= b[a] - b[a2] + 1)
        eval_model.addConstr(M * (1 - val_sa1) >= b[a2] - b[a])
        eval_model.addConstr(M * val_sa2 >= b[a2] - b[a] + d_k[a2])
        eval_model.addConstr(M * (1 - val_sa2) >= b[a] - b[a2] - d_k[a2] + 1)

    # objective: minimize same waiting time metric (not strictly necessary for extracting feasible b,
    # but keeps solution consistent with evaluation)
    terms = []
    for a in A:
        if A_info[a]["is_start"]:
            terms.append(b[a] - t_a[a])
        else:
            terms.append(b[a] - b[pre[a]] - d_k[pre[a]])
    eval_model.setObjective(quicksum(terms), GRB.MINIMIZE)

    eval_model.optimize()
    if eval_model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        raise RuntimeError(f"Schedule extraction infeasible for scenario {k} (status {eval_model.Status})")

    # build schedule list
    schedule = []
    for a in A:
        start = b[a].X
        dur = d_k[a]
        # find assigned resource j from x_fixed mapping
        assigned_j = None
        for (aa,j), val in x_fixed.items():
            if aa == a:
                assigned_j = j
                break
        schedule.append({'activity': a, 'start': float(start), 'dur': float(dur), 'resource': assigned_j})

    # sort schedule by start time
    schedule = sorted(schedule, key=lambda x: x['start'])
    return schedule

def plot_gantt_schedule(schedule, filename=None, title=None, figsize=(12,6)):
    """
    schedule: list of dicts with keys 'activity', 'start', 'dur', 'resource'
    filename: if provided, save figure to this path
    """
    # Group by resource
    resources = sorted(list({s['resource'] for s in schedule}))
    resource_to_y = {r: i for i,r in enumerate(resources)}  # 0..R-1

    # Prepare plotting
    fig, ax = plt.subplots(figsize=figsize)
    yticks = []
    yticklabels = []
    bar_height = 0.6

    # For legend: unique activity colors (matplotlib chooses default colors)
    patches = []

    for s in schedule:
        r = s['resource']
        y = resource_to_y[r]
        ax.barh(y, s['dur'], left=s['start'], height=bar_height)
        # annotate activity id on bar
        ax.text(s['start'] + s['dur']/2, y, str(s['activity']), va='center', ha='center', fontsize=9, color='k')
        if r not in yticklabels:
            yticks.append(y)
            yticklabels.append(str(r))

    ax.set_yticks(list(resource_to_y.values()))
    ax.set_yticklabels([str(r) for r in resources])
    ax.set_xlabel("Time (same units as durations)")
    ax.set_ylabel("Resource (unit id)")
    ax.set_title(title or "Schedule Gantt Chart")
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=200)
        print(f"Gantt chart saved to {filename}")
    plt.show()


# ==========================================
# 1. 資料預處理 (Data Preprocessing)
# ==========================================
def build_index_sets(data):
    # Activities A
    activities = data["activities"]
    A = [act["id"] for act in activities]
    act_by_id = {act["id"]: act for act in activities}

    # Resources: J = list of uids
    resource_uid_to_type = data["resource_uid_to_type"]
    J = list(resource_uid_to_type.keys())

    # Resource types G and Jg mapping
    resources_by_type = data["resources_by_type"]
    G = list(resources_by_type.keys())
    Jg = {g: [unit["uid"] for unit in resources_by_type[g]] for g in G}

    # A(g): Activities requiring resource type g
    A_g = {g: [] for g in G}
    for act in activities:
        for g in act["required_types"]:
            if g not in A_g:
                A_g[g] = []
            A_g[g].append(act["id"])

    # eligible_j_for_a: Pre-calculate eligible resources to reduce variables
    eligible_j_for_a = {}
    for act in activities:
        a = act["id"]
        eligible = []
        for g in act["required_types"]:
            if g not in Jg:
                raise ValueError(f"Activity {a} requires unknown resource type {g}")
            eligible.extend(Jg[g])
        eligible_j_for_a[a] = list(sorted(set(eligible)))

    return {
        "A": A, "A_info": act_by_id, "J": J, "G": G,
        "Jg": Jg, "A_g": A_g, "eligible_j_for_a": eligible_j_for_a
    }

# ==========================================
# 2. 模型建構：訓練階段 (Training Phase)
# ==========================================

def solve_smilp_training(data, M=1000, time_limit=3600, verbose=True):
    """
    論文中的 SMILP 模型 (Formulation 1 & 2)。
    使用 N0 (Training) 樣本數來決定最佳的第一階段變數 (x, sa1, sa2, q)。
    """
    idx = build_index_sets(data)
    A, A_info = idx["A"], idx["A_info"]
    J, Jg, G = idx["J"], idx["Jg"], idx["G"]
    eligible = idx["eligible_j_for_a"]
    num_scenarios = data["num_scenarios"]
    scenarios = list(range(num_scenarios))

    # Parameters
    Va_g = {(a,g): 1 for a in A for g in A_info[a]["required_types"]}
    k_j = {j: 1 for j in J}
    t_a = {a: (A_info[a]["scheduled_start"] if A_info[a]["is_start"] else None) for a in A}
    pre = {a: A_info[a]["predecessor"] for a in A}
    
    # Durations d[a,k]
    d = {}
    for a in A:
        dur_list = A_info[a]["durations"]
        for k in scenarios:
            d[(a,k)] = float(dur_list[k])

    # Share Pairs
    share_pairs = []
    for a in A:
        types_a = set(A_info[a]["required_types"])
        for a2 in A:
            if a == a2: continue
            if len(types_a & set(A_info[a2]["required_types"])) > 0:
                share_pairs.append((a,a2))

    # Model
    model = Model("SMILP_Training")
    if time_limit:
        model.setParam('TimeLimit', time_limit)
    # if not verbose: model.setParam('OutputFlag', 0)
    model.setParam('MIPGap', 0.05)
    # --- Stage 1 Variables (Here and Now) ---
    x = {}
    for a in A:
        for j in eligible[a]:
            x[(a,j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{a}_{j}")

    sa1, sa2, q = {}, {}, {}
    for (a,a2) in share_pairs:
        sa1[(a,a2)] = model.addVar(vtype=GRB.BINARY, name=f"sa1_{a}_{a2}")
        sa2[(a,a2)] = model.addVar(vtype=GRB.BINARY, name=f"sa2_{a}_{a2}")
        
        types_inter = set(A_info[a]["required_types"]) & set(A_info[a2]["required_types"])
        js = sorted(list(set([j for g in types_inter for j in Jg[g]])))
        for j in js:
            q[(j,a,a2)] = model.addVar(vtype=GRB.BINARY, name=f"q_{j}_{a}_{a2}")

    model.update()

    # --- Stage 2 Variables (Wait and See) ---
    b = {}
    for a in A:
        for k in scenarios:
            b[(a,k)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"b_{a}_{k}")

    # --- Stage 1 Constraints ---
    # (1b) Assignment
    for a in A:
        for g in A_info[a]["required_types"]:
            model.addConstr(quicksum(x[(a,j)] for j in Jg[g] if (a,j) in x) == 1)

    # (1c) q definition logic
    for (j,a,a2), qvar in q.items():
        model.addConstr(qvar >= sa1[(a,a2)] + sa2[(a,a2)] + x.get((a,j),0) + x.get((a2,j),0) - 3)

    # (1d) Capacity via q
    for g in G:
        for j in Jg[g]:
            acts = idx["A_g"].get(g, [])
            for a in acts:
                # Sum of q over other activities a2
                lhs = quicksum(q[(j,a,a2)] for a2 in acts if a2 != a and (j,a,a2) in q)
                model.addConstr(lhs <= k_j[j] - 1)

    # --- Stage 2 Constraints ---
    for k in scenarios:
        # (2b & 2c) Precedence
        for a in A:
            if A_info[a]["is_start"]:
                model.addConstr(b[(a,k)] >= t_a[a])
            else:
                prev = pre[a]
                model.addConstr(b[(a,k)] >= b[(prev,k)] + d[(prev,k)])

        # (2d-2g) Big-M Sequencing
        for (a,a2) in share_pairs:
            # Note: Using M=1000 effectively
            model.addConstr(M * sa1[(a,a2)] >= b[(a,k)] - b[(a2,k)] + 1)
            model.addConstr(M * (1 - sa1[(a,a2)]) >= b[(a2,k)] - b[(a,k)])
            model.addConstr(M * sa2[(a,a2)] >= b[(a2,k)] - b[(a,k)] + d[(a2,k)])
            model.addConstr(M * (1 - sa2[(a,a2)]) >= b[(a,k)] - b[(a2,k)] - d[(a2,k)] + 1)

    # Objective: Min Avg Waiting Time
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
    
    print(f"Solving SMILP with {num_scenarios} scenarios...")
    model.optimize()

    # Extract First-Stage Solution ONLY
    sol_first_stage = {
        "x": {(a,j): 1 for (a,j) in x if x[(a,j)].X > 0.5},
        "sa1": {(a,a2): 1 for (a,a2) in sa1 if sa1[(a,a2)].X > 0.5},
        "sa2": {(a,a2): 1 for (a,a2) in sa2 if sa2[(a,a2)].X > 0.5},
        # q is derived, but can be passed if needed. Usually x, sa1, sa2 are sufficient to fix schedule.
    }
    return sol_first_stage

def solve_baseline_training(data, M=1000, time_limit=3600):
    """
    論文中的 Baseline Mean Value Model。
    只看「平均工期」來解一個 Deterministic MILP。
    """
    idx = build_index_sets(data)
    A, A_info = idx["A"], idx["A_info"]
    J, Jg, G = idx["J"], idx["Jg"], idx["G"]
    eligible = idx["eligible_j_for_a"]

    # Use Mean Durations
    mean_d = {a: float(A_info[a]["mean_duration"]) for a in A}
    t_a = {a: (A_info[a]["scheduled_start"] if A_info[a]["is_start"] else None) for a in A}
    pre = {a: A_info[a]["predecessor"] for a in A}
    
    # Model
    model = Model("Baseline_Mean")
    # model.setParam('OutputFlag', 0)
    if time_limit: model.setParam('TimeLimit', time_limit)

    # Variables (Same structure, but no scenario index k)
    x = {}
    for a in A:
        for j in eligible[a]:
            x[(a,j)] = model.addVar(vtype=GRB.BINARY)
    
    share_pairs = []
    sa1, sa2, q = {}, {}, {}
    for a in A:
        for a2 in A:
            if a == a2: continue
            if len(set(A_info[a]["required_types"]) & set(A_info[a2]["required_types"])) > 0:
                share_pairs.append((a,a2))
                sa1[(a,a2)] = model.addVar(vtype=GRB.BINARY)
                sa2[(a,a2)] = model.addVar(vtype=GRB.BINARY)
                # q generation simplified for brevity, similar to SMILP
                types_inter = set(A_info[a]["required_types"]) & set(A_info[a2]["required_types"])
                js = sorted(list(set([j for g in types_inter for j in Jg[g]])))
                for j in js:
                    q[(j,a,a2)] = model.addVar(vtype=GRB.BINARY)

    b = {a: model.addVar(lb=0.0) for a in A}

    # Constraints (Same logic as SMILP but deterministic)
    # (1b)
    for a in A:
        for g in A_info[a]["required_types"]:
            model.addConstr(quicksum(x[(a,j)] for j in Jg[g] if (a,j) in x) == 1)
    
    # (1c & 1d) Q logic and Capacity
    for (j,a,a2), qvar in q.items():
        model.addConstr(qvar >= sa1[(a,a2)] + sa2[(a,a2)] + x.get((a,j),0) + x.get((a2,j),0) - 3)
    
    # Capacity simplified loop
    for g in G:
        for j in Jg[g]:
            acts = idx["A_g"].get(g, [])
            for a in acts:
                lhs = quicksum(q[(j,a,a2)] for a2 in acts if a2 != a and (j,a,a2) in q)
                model.addConstr(lhs <= 1 - 1) # capacity is 1

    # Deterministic Sequencing & Timing
    for a in A:
        if A_info[a]["is_start"]:
            model.addConstr(b[a] >= t_a[a])
        else:
            prev = pre[a]
            model.addConstr(b[a] >= b[prev] + mean_d[prev])
    
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

    print("Solving Baseline Deterministic Model...")
    model.optimize()

    sol_first_stage = {
        "x": {(a,j): 1 for (a,j) in x if x[(a,j)].X > 0.5},
        "sa1": {(a,a2): 1 for (a,a2) in sa1 if sa1[(a,a2)].X > 0.5},
        "sa2": {(a,a2): 1 for (a,a2) in sa2 if sa2[(a,a2)].X > 0.5},
    }
    return sol_first_stage

# ==========================================
# 3. 評估階段 (Evaluation / Testing Phase)
# ==========================================

def evaluate_solution_on_scenarios(first_stage_sol, test_data, M=1000):
    """
    Monte Carlo Simulation 步驟：
    將固定的第一階段變數 (x, sa1, sa2) 套用到 N' (Test) 個情境中。
    針對每個情境 k，這變成一個簡單的 LP (因為 binary 都固定了)。
    """
    idx = build_index_sets(test_data)
    A, A_info = idx["A"], idx["A_info"]
    scenarios = list(range(test_data["num_scenarios"]))
    t_a = {a: (A_info[a]["scheduled_start"] if A_info[a]["is_start"] else None) for a in A}
    pre = {a: A_info[a]["predecessor"] for a in A}

    # Retrieve fixed decisions
    x_fixed = first_stage_sol["x"]
    sa1_fixed = first_stage_sol["sa1"]
    sa2_fixed = first_stage_sol["sa2"]

    Q_values = []
    print(f"Evaluating solution on {len(scenarios)} testing scenarios...")

    # Reuse model environment for speed? Or create new. Creating new is safer for clarity.
    # For 500 scenarios, creating 500 small models is fast enough.
    
    for k in scenarios:
        # Build a simple LP for this specific scenario
        eval_model = Model(f"Eval_k{k}")
        # eval_model.setParam('OutputFlag', 0)
        
        # Variables: only b[a]
        b = {a: eval_model.addVar(lb=0.0, name=f"b_{a}") for a in A}
        
        # Get duration for THIS scenario
        d_k = {a: float(A_info[a]["durations"][k]) for a in A}

        # Constraints
        # (2b & 2c)
        for a in A:
            if A_info[a]["is_start"]:
                eval_model.addConstr(b[a] >= t_a[a])
            else:
                prev = pre[a]
                eval_model.addConstr(b[a] >= b[prev] + d_k[prev])

        # (2d-2g) Sequencing - NOW THESE ARE LINEAR CONSTRAINTS because sa1/sa2 are constants
        # Identify pairs sharing resources
        # We need to loop through the sa1 keys from the solution
        for (a,a2), val_sa1 in sa1_fixed.items():
            val_sa2 = sa2_fixed.get((a,a2), 0)
            
            # Constraints using constants 0 or 1
            eval_model.addConstr(M * val_sa1 >= b[a] - b[a2] + 1)
            eval_model.addConstr(M * (1 - val_sa1) >= b[a2] - b[a])
            eval_model.addConstr(M * val_sa2 >= b[a2] - b[a] + d_k[a2])
            eval_model.addConstr(M * (1 - val_sa2) >= b[a] - b[a2] - d_k[a2] + 1)

        # Objective: Wait time for scenario k
        terms = []
        for a in A:
            if A_info[a]["is_start"]:
                terms.append(b[a] - t_a[a])
            else:
                terms.append(b[a] - b[pre[a]] - d_k[pre[a]])
        
        eval_model.setObjective(quicksum(terms), GRB.MINIMIZE)
        eval_model.optimize()

        if eval_model.Status == GRB.OPTIMAL:
            Q_values.append(eval_model.ObjVal)
        else:
            # 若因 M 不夠大或其他原因導致 infeasible，給予罰分 (Penalty)
            # 在論文假設 Assumption 2 (資源充足) 下，應該總是可行的
            Q_values.append(10000.0) 

    return Q_values

# ==========================================
# 4. 視覺化 (Visualization)
# ==========================================

def plot_paper_comparison(smilp_Q, baseline_Q):
    smilp_vals = [v for v in smilp_Q if v < 9000]
    baseline_vals = [v for v in baseline_Q if v < 9000]

    smilp_mean = np.mean(smilp_vals)
    baseline_mean = np.mean(baseline_vals)
    
    # VSS Calculation
    vss = baseline_mean - smilp_mean
    improvement_pct = (vss / baseline_mean) * 100 if baseline_mean > 0 else 0

    text_str = (f"Baseline Avg: {baseline_mean:.2f}\n"
                f"SMILP Avg: {smilp_mean:.2f}\n"
                f"VSS (Diff): {vss:.2f}\n"
                f"Improvement: {improvement_pct:.2f}%")

    plt.figure(figsize=(9, 6))
    plt.boxplot([baseline_vals, smilp_vals], labels=["Baseline (Mean Value)", "SMILP (Stochastic)"], 
                showmeans=True, patch_artist=True)
    
    plt.ylabel("Total Patient Waiting Time (mins)")
    plt.title(f"Evaluation on Out-of-Sample Scenarios (N={len(smilp_Q)})")
    
    # Add text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("smilp_paper_result.png")
    print(f"\n=== Final Results ===")
    print(text_str)
    print("Plot saved to smilp_paper_result.png")

# ==========================================
# 5. 主程式 (Main Execution)
# ==========================================
def split_data(full_data, n_train, n_test):
    """
    將一份包含 (n_train + n_test) 個情境的資料集切分成兩份。
    保持 Activities 結構不變，只切分 durations 列表。
    """
    import copy
    
    # 深層複製以避免修改原始資料
    # 注意：如果資料量極大，deepcopy 會慢，但在這裡應該還好
    train_data = copy.deepcopy(full_data)
    test_data = copy.deepcopy(full_data)
    
    total_scenarios = full_data["num_scenarios"]
    if total_scenarios < n_train + n_test:
        raise ValueError(f"Data has {total_scenarios} scenarios, but need {n_train} + {n_test}")

    # 修改 Train Data
    train_data["num_scenarios"] = n_train
    for act in train_data["activities"]:
        # 取前 n_train 個情境
        act["durations"] = act["durations"][:n_train]
        
    # 修改 Test Data
    test_data["num_scenarios"] = n_test
    for act in test_data["activities"]:
        # 取後 n_test 個情境
        # 重要：必須讓 test 的情境索引重新從 0 開始，以配合 evaluate 函式的迴圈
        act["durations"] = act["durations"][n_train : n_train + n_test]

    return train_data, test_data

# ==========================================
# 6. 多次實驗 (Multiple Replications)
# ==========================================

def run_multiple_experiments(
    num_runs=10,
    num_patients=15,
    train_scenarios=200,
    test_scenarios=500,
    base_seed=42,
    time_limit_baseline=3600,
    time_limit_smilp=3600
):
    """
    重複執行完整實驗 num_runs 次，
    回傳每次的 improvement (VSS 與 %)
    """

    results = []

    for r in range(num_runs):
        print("\n" + "="*60)
        print(f" Running Experiment {r+1}/{num_runs}")
        print("="*60)

        # 每次實驗用不同 seed（但可重現）
        seed = base_seed + r * 100

        gen = InstanceGenerator(
            num_patients=num_patients,
            arrival_interval=10,
            random_seed=seed
        )

        total_scenarios = train_scenarios + test_scenarios
        full_data = gen.generate_data(num_scenarios=total_scenarios)
        train_data, test_data = split_data(full_data, train_scenarios, test_scenarios)

        # === Training ===
        print("Solving Baseline...")
        sol_baseline = solve_baseline_training(
            train_data, time_limit=time_limit_baseline
        )

        print("Solving SMILP...")
        sol_smilp = solve_smilp_training(
            train_data, time_limit=time_limit_smilp
        )

        # === Evaluation ===
        q_base = evaluate_solution_on_scenarios(sol_baseline, test_data)
        q_smilp = evaluate_solution_on_scenarios(sol_smilp, test_data)

        # 清掉 infeasible penalty
        base_vals = [v for v in q_base if v < 9000]
        smilp_vals = [v for v in q_smilp if v < 9000]

        base_mean = np.mean(base_vals)
        smilp_mean = np.mean(smilp_vals)

        vss = base_mean - smilp_mean
        improvement_pct = (vss / base_mean) * 100 if base_mean > 0 else 0

        results.append({
            "run": r + 1,
            "baseline_mean": base_mean,
            "smilp_mean": smilp_mean,
            "vss": vss,
            "improvement_pct": improvement_pct
        })

        print(f"Run {r+1} | Baseline={base_mean:.2f}, "
              f"SMILP={smilp_mean:.2f}, "
              f"VSS={vss:.2f}, "
              f"Improvement={improvement_pct:.2f}%")

    return results

def plot_improvement_variability(results):
    improvements = [r["improvement_pct"] for r in results]

    mean_imp = np.mean(improvements)
    std_imp = np.std(improvements)

    plt.figure(figsize=(9, 6))

    # Boxplot
    plt.boxplot(improvements, widths=0.4, showmeans=True)

    # Scatter (jitter)
    x = np.random.normal(1, 0.02, size=len(improvements))
    plt.scatter(x, improvements, alpha=0.7)

    plt.ylabel("Improvement over Baseline (%)")
    plt.title("Variability of SMILP Improvement over 10 Replications")

    text = (f"Mean Improvement: {mean_imp:.2f}%\n"
            f"Std Dev: {std_imp:.2f}%")

    plt.text(
        0.7, max(improvements)*0.95,
        text,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6)
    )

    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("improvement_variability.png", dpi=200)
    # plt.show()

    print("\n=== Improvement Statistics ===")
    print(text)
    print("Plot saved to improvement_variability.png")

if __name__ == "__main__":
    results = run_multiple_experiments(
        num_runs=10,
        num_patients=15,
        train_scenarios=100,
        test_scenarios=500,
        base_seed=42
    )

    plot_improvement_variability(results)
