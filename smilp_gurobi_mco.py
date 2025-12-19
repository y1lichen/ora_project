# smilp_paper_compliant.py
from gurobipy import Model, GRB, quicksum
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy

# 假設 instance_generator.py 與此檔案在同一目錄
from instance_generator import InstanceGenerator 

SLACK_PENALTY = 1000
# ==========================================
# 1. 資料預處理 (Data Preprocessing)
# ==========================================
def build_index_sets(data):
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
            if g not in A_g: A_g[g] = []
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
# 2. 模型建構：SAA 求解器
# ==========================================
def solve_saa_model(data, M=50000, time_limit=3600, verbose=False):
    idx = build_index_sets(data)
    A, A_info = idx["A"], idx["A_info"]
    J, Jg, G = idx["J"], idx["Jg"], idx["G"]
    eligible = idx["eligible_j_for_a"]
    num_scenarios = data["num_scenarios"]
    scenarios = list(range(num_scenarios))

    t_a = {a: (A_info[a]["scheduled_start"] if A_info[a]["is_start"] else None) for a in A}
    pre = {a: A_info[a]["predecessor"] for a in A}
    Va_g = {(a,g): 1 for a in A for g in A_info[a]["required_types"]}
    k_j = {j: 1 for j in J}
    
    d = {}
    for a in A:
        for k in scenarios: d[(a,k)] = float(A_info[a]["durations"][k])

    share_pairs = []
    for a in A:
        types_a = set(A_info[a]["required_types"])
        for a2 in A:
            if a == a2: continue
            if len(types_a & set(A_info[a2]["required_types"])) > 0:
                share_pairs.append((a,a2))

    model = Model(f"SAA_N{num_scenarios}")
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)
    
    x = { (a,j): model.addVar(vtype=GRB.BINARY, name=f"x_{a}_{j}") for a in A for j in eligible[a] }
    sa1, sa2, q = {}, {}, {}
    for (a,a2) in share_pairs:
        sa1[(a,a2)] = model.addVar(vtype=GRB.BINARY, name=f"sa1_{a}_{a2}")
        sa2[(a,a2)] = model.addVar(vtype=GRB.BINARY, name=f"sa2_{a}_{a2}")
        types_inter = set(A_info[a]["required_types"]) & set(A_info[a2]["required_types"])
        js = sorted(list(set([j for g in types_inter for j in Jg[g]])))
        for j in js: q[(j,a,a2)] = model.addVar(vtype=GRB.BINARY, name=f"q_{j}_{a}_{a2}")

    b = { (a,k): model.addVar(lb=0.0, name=f"b_{a}_{k}") for a in A for k in scenarios }

    for a in A:
        for g in A_info[a]["required_types"]:
            model.addConstr(quicksum(x.get((a,j),0) for j in Jg[g]) == Va_g[(a,g)])

    for (j,a,a2), qvar in q.items():
        model.addConstr(qvar >= sa1[(a,a2)] + sa2[(a,a2)] + x.get((a,j),0) + x.get((a2,j),0) - 3)

    for g in G:
        for j in Jg[g]:
            acts_g = idx["A_g"].get(g, [])
            for a in acts_g:
                lhs = quicksum(q[(j,a,a2)] for a2 in acts_g if a2 != a and (j,a,a2) in q)
                model.addConstr(lhs <= k_j[j] - 1)

    for k in scenarios:
        for a in A:
            if A_info[a]["is_start"]: model.addConstr(b[(a,k)] >= t_a[a])
            else: model.addConstr(b[(a,k)] >= b[(pre[a],k)] + d[(pre[a],k)])
        for (a,a2) in share_pairs:
            model.addConstr(M * sa1[(a,a2)] >= b[(a,k)] - b[(a2,k)] + 1)
            model.addConstr(M * (1 - sa1[(a,a2)]) >= b[(a2,k)] - b[(a,k)])
            model.addConstr(M * sa2[(a,a2)] >= b[(a2,k)] - b[(a,k)] + d[(a2,k)])
            model.addConstr(M * (1 - sa2[(a,a2)]) >= b[(a,k)] - b[(a2,k)] - d[(a2,k)] + 1)

    Q_k_terms = []
    for k in scenarios:
        terms = []
        for a in A:
            if A_info[a]["is_start"]: terms.append(b[(a,k)] - t_a[a])
            else: terms.append(b[(a,k)] - b[(pre[a],k)] - d[(pre[a],k)])
        Q_k_terms.append(quicksum(terms))
    
    model.setObjective((1.0/num_scenarios) * quicksum(Q_k_terms), GRB.MINIMIZE)
    model.optimize()

    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        return {
            "objective": model.ObjVal,
            "x": {(a,j): round(var.X) for (a,j), var in x.items() if var.X > 0.5},
            "sa1": {(a,a2): round(sa1[(a,a2)].X) for (a,a2) in sa1 if sa1[(a,a2)].X > 0.5},
            "sa2": {(a,a2): round(sa2[(a,a2)].X) for (a,a2) in sa2 if sa2[(a,a2)].X > 0.5}
        }
    return None

# ==========================================
# 3. MCO 函數 (Algorithm 1)
# ==========================================
def run_mco_method(train_data_pool, N0, K, N_prime, epsilon, M, time_limit):
    N = N0
    AOIN = float('inf')
    iteration = 0
    final_sol = None
    
    print("--- Running Monte Carlo Optimization (MCO) ---")
    while AOIN >= epsilon:
        iteration += 1
        print(f"\n--- MCO Iteration {iteration}: N = {N} ---")
        v_N_list, v_N_prime_list = [], []
        
        for k in range(K):
            idx_list = np.random.choice(range(train_data_pool["num_scenarios"]), N + N_prime, replace=False)
            train_sub = copy.deepcopy(train_data_pool)
            eval_sub = copy.deepcopy(train_data_pool)
            train_sub["num_scenarios"], eval_sub["num_scenarios"] = N, N_prime
            
            for act in train_sub["activities"]: act["durations"] = [act["durations"][i] for i in idx_list[:N]]
            for act in eval_sub["activities"]: act["durations"] = [act["durations"][i] for i in idx_list[N:]]

            saa_result = solve_saa_model(train_sub, M=M, time_limit=time_limit)
            if saa_result is None: continue

            # training objective (in-sample SAA)
            v_N_list.append(saa_result["objective"])

            # out-of-sample evaluation (paper-style)
            eval_val = evaluate_solution_on_scenarios_paper_style(
                saa_result, eval_sub, M=M, slack_penalty=SLACK_PENALTY
            )
            v_N_prime_list.append(eval_val)
            final_sol = {"sol": saa_result, "N": N}


        mean_v_N = np.mean(v_N_list)
        mean_v_N_prime = np.mean(v_N_prime_list)
        AOIN = abs(mean_v_N_prime - mean_v_N) / mean_v_N_prime if mean_v_N_prime > 0 else 0
        
        print(f"  > Avg v_N: {mean_v_N:.2f}, Avg v_N': {mean_v_N_prime:.2f}, AOIN: {AOIN:.4f}")
        if AOIN < epsilon: break
        N *= 2
        if N > train_data_pool["num_scenarios"] // 2: break
    return final_sol

# ==========================================
# 4. 評估與 Baseline
# ==========================================
def solve_baseline_training(data, M=50000, time_limit=600):
    idx = build_index_sets(data)
    A, A_info = idx["A"], idx["A_info"]
    J, Jg, G = idx["J"], idx["Jg"], idx["G"]
    eligible = idx["eligible_j_for_a"]
    mean_d = {a: float(A_info[a]["mean_duration"]) for a in A}
    t_a = {a: (A_info[a]["scheduled_start"] if A_info[a]["is_start"] else None) for a in A}
    pre = {a: A_info[a]["predecessor"] for a in A}
    
    model = Model("Baseline_Mean")
    model.setParam('OutputFlag', 0)
    
    x = { (a,j): model.addVar(vtype=GRB.BINARY) for a in A for j in eligible[a] }
    share_pairs = [(a,a2) for a in A for a2 in A if a != a2 and set(A_info[a]["required_types"]) & set(A_info[a2]["required_types"])]
    sa1 = {(a,a2): model.addVar(vtype=GRB.BINARY) for (a,a2) in share_pairs}
    sa2 = {(a,a2): model.addVar(vtype=GRB.BINARY) for (a,a2) in share_pairs}
    b = {a: model.addVar(lb=0.0) for a in A}

    for a in A:
        for g in A_info[a]["required_types"]: model.addConstr(quicksum(x.get((a,j),0) for j in Jg[g]) == 1)
        if A_info[a]["is_start"]: model.addConstr(b[a] >= t_a[a])
        else: model.addConstr(b[a] >= b[pre[a]] + mean_d[pre[a]])

    for (a,a2) in share_pairs:
        model.addConstr(M * sa1[(a,a2)] >= b[a] - b[a2] + 1)
        model.addConstr(M * (1 - sa1[(a,a2)]) >= b[a2] - b[a])
        model.addConstr(M * sa2[(a,a2)] >= b[a2] - b[a] + mean_d[a2])
        model.addConstr(M * (1 - sa2[(a,a2)]) >= b[a] - b[a2] - mean_d[a2] + 1)

    model.setObjective(quicksum(b[a] - (t_a[a] if A_info[a]["is_start"] else b[pre[a]]+mean_d[pre[a]]) for a in A), GRB.MINIMIZE)
    model.optimize()

    return {
        "x": {(a,j): round(var.X) for (a,j), var in x.items() if var.X > 0.5},
        "sa1": {(a,a2): round(sa1[(a,a2)].X) for (a,a2) in sa1 if sa1[(a,a2)].X > 0.5},
        "sa2": {(a,a2): round(sa2[(a,a2)].X) for (a,a2) in sa2 if sa2[(a,a2)].X > 0.5}
    }


def split_data(full_data, n_train, n_test):
    train_data = copy.deepcopy(full_data)
    test_data = copy.deepcopy(full_data)
    train_data["num_scenarios"], test_data["num_scenarios"] = n_train, n_test
    for act in train_data["activities"]: act["durations"] = act["durations"][:n_train]
    for act in test_data["activities"]: act["durations"] = act["durations"][n_train:n_train+n_test]
    return train_data, test_data

def evaluate_solution_on_scenarios_paper_style(sol, data, M=50000, slack_penalty=240):
    """
    Paper-consistent evaluation:
    - All scenarios are included
    - Infeasibility is absorbed via slack variables
    - Objective is unconditional expectation
    """
    idx = build_index_sets(data)
    A, A_info = idx["A"], idx["A_info"]
    scenarios = range(data["num_scenarios"])
    sa1_f, sa2_f = sol["sa1"], sol["sa2"]

    Q_vals = []

    for k in scenarios:
        m = Model()
        m.setParam("OutputFlag", 0)

        b = {a: m.addVar(lb=0.0) for a in A}
        r = {a: m.addVar(lb=0.0) for a in A}  # slack
        d_k = {a: float(A_info[a]["durations"][k]) for a in A}

        # precedence
        for a in A:
            if A_info[a]["is_start"]:
                m.addConstr(b[a] >= A_info[a]["scheduled_start"])
            else:
                m.addConstr(
                    b[a] + r[a] >= b[A_info[a]["predecessor"]] + d_k[A_info[a]["predecessor"]]
                )

        # sequencing
        for (a, a2), v1 in sa1_f.items():
            v2 = sa2_f.get((a, a2), 0)
            m.addConstr(M * v1 >= b[a] - b[a2] + 1)
            m.addConstr(M * (1 - v1) >= b[a2] - b[a])
            m.addConstr(M * v2 >= b[a2] - b[a] + d_k[a2])
            m.addConstr(M * (1 - v2) >= b[a] - b[a2] - d_k[a2] + 1)

        # objective: waiting + slack
        m.setObjective(
            quicksum(
                b[a]
                - (
                    A_info[a]["scheduled_start"]
                    if A_info[a]["is_start"]
                    else b[A_info[a]["predecessor"]] + d_k[A_info[a]["predecessor"]]
                )
                for a in A
            )
            + slack_penalty * quicksum(r.values()),
            GRB.MINIMIZE,
        )

        m.optimize()

        if m.Status == GRB.OPTIMAL:
            Q_vals.append(m.ObjVal)
        else:
            # paper-style: infeasible or no incumbent → penalize
            Q_vals.append(SLACK_PENALTY)


    return np.mean(Q_vals)


if __name__ == "__main__":
    NUM_PATIENTS = 15
    TRAIN_SIZE = 300 
    TEST_SIZE = 500
    
    # 增加 Big-M 以確保在高波動情境下仍有可行解
    BIG_M = 100000 
    
    gen = InstanceGenerator(num_patients=NUM_PATIENTS, arrival_interval=10, random_seed=42)
    full_data = gen.generate_data(num_scenarios=TRAIN_SIZE + TEST_SIZE)
    train_data, test_data = split_data(full_data, TRAIN_SIZE, TEST_SIZE)

    print("=== Step 1: Running MCO for SMILP ===")
    mco_res = run_mco_method(train_data, N0=30, K=3, N_prime=50, epsilon=0.05, M=BIG_M, time_limit=600)
    sol_smilp = mco_res["sol"]

    print("\n=== Step 2: Solving Baseline ===")
    sol_base = solve_baseline_training(train_data, M=BIG_M)

    print("\n=== Step 3: Final Comparison ===")
    v_smilp = evaluate_solution_on_scenarios_paper_style(sol_smilp, test_data, M=BIG_M)
    v_base  = evaluate_solution_on_scenarios_paper_style(sol_base,  test_data, M=BIG_M)

    VSS = v_base - v_smilp
    ratio = VSS / v_base * 100

    print("\n=== Paper-style Evaluation ===")
    print(f"Baseline expected obj: {v_base:.2f}")
    print(f"SMILP    expected obj: {v_smilp:.2f}")
    print(f"VSS: {VSS:.2f}")
    print(f"Improvement ratio: {ratio:.2f}%")

    plt.boxplot(
        [v_base, v_smilp],
        labels=["Baseline", "SMILP"],
    )
    plt.ylabel("Waiting Time (Feasible Only)")
    plt.title("Fair Comparison on Test Set")
    plt.savefig("smilp_mco_comparison.png")

