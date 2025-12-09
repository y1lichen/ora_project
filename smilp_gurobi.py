# smilp_gurobi.py
# Requires gurobipy installed and a valid Gurobi license.
from gurobipy import Model, GRB, quicksum
import math
import copy
from instance_generator import InstanceGenerator
import matplotlib.pyplot as plt


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

    # For each activity a, determine which resource types it requires (A(g) sets)
    A_g = {g: [] for g in G}
    for act in activities:
        for g in act["required_types"]:
            if g not in A_g:
                A_g[g] = []
            A_g[g].append(act["id"])

    # Build useful helper: for each activity a, list eligible resource units j (whose type in required_types)
    eligible_j_for_a = {}
    for act in activities:
        a = act["id"]
        eligible = []
        for g in act["required_types"]:
            if g not in Jg:
                raise ValueError(f"Activity {a} requires unknown resource type {g}")
            eligible.extend(Jg[g])
        # remove duplicates
        eligible_j_for_a[a] = list(sorted(set(eligible)))

    return {
        "A": A,
        "A_info": act_by_id,
        "J": J,
        "G": G,
        "Jg": Jg,
        "A_g": A_g,
        "eligible_j_for_a": eligible_j_for_a
    }

def build_smilp_model(data, M=1000, time_limit=None, verbose=True):
    """
    Build and solve the two-stage SMILP via SAA deterministic equivalent:
    - First-stage binaries: x[a,j], sa1[a,a2], sa2[a,a2], q[j,a,a2]
    - Second-stage continuous: b[a,k] for each scenario k
    Objective: minimize (1/num_scenarios) * sum_k Q_k
    Returns: model, solution dict with first-stage variables and objective info.
    """
    idx = build_index_sets(data)
    A = idx["A"]
    A_info = idx["A_info"]
    J = idx["J"]
    G = idx["G"]
    Jg = idx["Jg"]
    eligible = idx["eligible_j_for_a"]
    num_scenarios = data["num_scenarios"]
    scenarios = list(range(num_scenarios))

    # Va_g: number of resources of type g required by activity a
    # From your data "required_types" list, assume each listed type needs 1 unit
    Va_g = {}
    for a in A:
        req_types = A_info[a]["required_types"]
        for g in req_types:
            Va_g[(a,g)] = 1  # if in future you have counts, change here

    # resource capacities k_j: each resource unit has capacity 1
    k_j = {j: 1 for j in J}

    # scheduled times t_a for initial activities; None otherwise
    t_a = {a: (A_info[a]["scheduled_start"] if A_info[a]["is_start"] else None) for a in A}

    # predecessor mapping pre[a] = predecessor activity id or None
    pre = {a: A_info[a]["predecessor"] for a in A}

    # durations per scenario d[a,k]
    d = {}
    for a in A:
        dur_list = A_info[a]["durations"]
        if len(dur_list) != num_scenarios:
            raise ValueError(f"Activity {a} durations length {len(dur_list)} != num_scenarios {num_scenarios}")
        for k in scenarios:
            d[(a,k)] = float(dur_list[k])

    # Build pairs (a,a') that share at least one resource type (used for s1, s2, q constraints)
    share_pairs = []
    for a in A:
        types_a = set(A_info[a]["required_types"])
        for a2 in A:
            if a == a2:
                continue
            types_a2 = set(A_info[a2]["required_types"])
            if len(types_a & types_a2) > 0:
                share_pairs.append((a,a2))
    # We'll also need mapping for each g of activity pairs within A(g)
    pairs_by_g = {g: [] for g in G}
    for g in G:
        acts = idx["A_g"].get(g, [])
        for i in range(len(acts)):
            for j2 in range(len(acts)):
                if acts[i] == acts[j2]:
                    continue
                pairs_by_g[g].append((acts[i], acts[j2]))

    # --- Start building model ---
    model = Model("SMILP_SAA")
    if time_limit is not None:
        model.setParam('TimeLimit', time_limit)
    if not verbose:
        model.setParam('OutputFlag', 0)

    # First-stage variables
    x = {}      # x[a,j] binary assignment if resource j serves activity a (eligible j)
    for a in A:
        for j in eligible[a]:
            x[(a,j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{a}_{j}")

    sa1 = {}    # sa1[a,a2] binary
    sa2 = {}    # sa2[a,a2] binary
    for (a,a2) in share_pairs:
        sa1[(a,a2)] = model.addVar(vtype=GRB.BINARY, name=f"sa1_{a}_{a2}")
        sa2[(a,a2)] = model.addVar(vtype=GRB.BINARY, name=f"sa2_{a}_{a2}")

    q = {}      # q[j,a,a2] binary only when j in intersection of eligible units for both activities
    for (a,a2) in share_pairs:
        # j must be a resource unit whose type is in intersection of types
        types_inter = set(A_info[a]["required_types"]) & set(A_info[a2]["required_types"])
        js = []
        for g in types_inter:
            js.extend(Jg[g])
        js = list(sorted(set(js)))
        for j in js:
            q[(j,a,a2)] = model.addVar(vtype=GRB.BINARY, name=f"q_{j}_{a}_{a2}")

    model.update()

    # Second-stage variables: start times b[a,k] continuous >=0
    b = {}
    for a in A:
        for k in scenarios:
            b[(a,k)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"b_{a}_{k}")

    model.update()

    # ---------- First-stage constraints ----------
    # (1b) Va,g = sum_{j in Jg} x_a_j  for all a, g in required_types of a
    for a in A:
        for g in A_info[a]["required_types"]:
            lhs = quicksum(x[(a,j)] for j in Jg[g] if (a,j) in x)
            model.addConstr(lhs == Va_g[(a,g)], name=f"assign_count_a{a}_g{g}")

    # (1c) q_j_a_a' >= sa1_a_a' + sa2_a_a' + x_a_j + x_a'_j - 3
    for (j,a,a2), qvar in q.items():
        # sa1 and sa2 exist for pair
        model.addConstr(qvar >= sa1[(a,a2)] + sa2[(a,a2)] + x.get((a,j), 0) + x.get((a2,j), 0) - 3,
                        name=f"q_lin1_{j}_{a}_{a2}")

    # (1d) sum_{a' != a} q_j_a_a' <= k_j - 1   for each g, j in Jg, a in A(g)
    # Implement for each resource unit j: for each a that requires j's type
    for g in G:
        for j in Jg[g]:
            # all activities a in A(g)
            acts = idx["A_g"].get(g, [])
            for a in acts:
                lhs = quicksum(q[(j,a,a2)] for a2 in acts if a2 != a and (j,a,a2) in q)
                model.addConstr(lhs <= k_j[j] - 1, name=f"cap_q_j{j}_a{a}")

    # ---------- Second-stage constraints (for each scenario) ----------
    # (2b) ba - ta >= 0 for a in A0 (initial activities)
    for a in A:
        if A_info[a]["is_start"]:
            ta = t_a[a]
            if ta is None:
                raise ValueError(f"Activity {a} marked is_start but scheduled_start missing")
            for k in scenarios:
                model.addConstr(b[(a,k)] - ta >= 0, name=f"init_time_a{a}_k{k}")

    # (2c) ba - b_pre - d_pre >= 0 for a in A1
    for a in A:
        if not A_info[a]["is_start"]:
            prev = pre[a]
            if prev is None:
                raise ValueError(f"Activity {a} not start but predecessor missing")
            for k in scenarios:
                model.addConstr(b[(a,k)] - b[(prev,k)] - d[(prev,k)] >= 0, name=f"succ_time_a{a}_k{k}")

    # (2d)-(2g) big-M sequencing constraints for activities sharing resource types
    # For each g and pairs (a,a') in A(g) apply (2d)-(2g)
    for g in G:
        acts = idx["A_g"].get(g, [])
        for a in acts:
            for a2 in acts:
                if a == a2:
                    continue
                for k in scenarios:
                    # (2d) M * sa1 >= b_a - b_a' + 1
                    model.addConstr(M * sa1[(a,a2)] >= b[(a,k)] - b[(a2,k)] + 1,
                                    name=f"2d_g{g}_a{a}_a2{a2}_k{k}")
                    # (2e) M*(1 - sa1) >= b_a' - b_a
                    model.addConstr(M * (1 - sa1[(a,a2)]) >= b[(a2,k)] - b[(a,k)],
                                    name=f"2e_g{g}_a{a}_a2{a2}_k{k}")
                    # (2f) M * sa2 >= b_a' - b_a + d_a'
                    model.addConstr(M * sa2[(a,a2)] >= b[(a2,k)] - b[(a,k)] + d[(a2,k)],
                                    name=f"2f_g{g}_a{a}_a2{a2}_k{k}")
                    # (2g) M*(1 - sa2) >= b_a - b_a' - d_a' + 1
                    model.addConstr(M * (1 - sa2[(a,a2)]) >= b[(a,k)] - b[(a2,k)] - d[(a2,k)] + 1,
                                    name=f"2g_g{g}_a{a}_a2{a2}_k{k}")

    # ---------- Objective: average over scenarios of waiting time Q(x,ξ_k) ----------
    # For each scenario k, Q_k = sum_{a in A0} (b_a - t_a) + sum_{a in A1} (b_a - b_pre - d_pre)
    Q_per_k = {}
    for k in scenarios:
        terms = []
        for a in A:
            if A_info[a]["is_start"]:
                terms.append(b[(a,k)] - t_a[a])
            else:
                prev = pre[a]
                terms.append(b[(a,k)] - b[(prev,k)] - d[(prev,k)])
        Q_per_k[k] = quicksum(terms)

    # Average
    objective = (1.0 / num_scenarios) * quicksum(Q_per_k[k] for k in scenarios)
    model.setObjective(objective, GRB.MINIMIZE)

    # Solve
    model.update()
    model.optimize()

    # Retrieve first-stage variable values and objective
    sol = {
        "obj": model.ObjVal if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT or model.Status == GRB.OPTIMAL else None,
        "x": {(a,j): (x[(a,j)].X if (a,j) in x else 0) for a in A for j in eligible[a]},
        "sa1": {(a,a2): sa1[(a,a2)].X for (a,a2) in sa1},
        "sa2": {(a,a2): sa2[(a,a2)].X for (a,a2) in sa2},
        "q": {(j,a,a2): q[(j,a,a2)].X for (j,a,a2) in q},
        "Q_per_scenario": {k: (Q_per_k[k].getValue() if hasattr(Q_per_k[k], "getValue") else None) for k in scenarios}
    }

    return model, sol

def build_baseline_mean_model(data, M=1000, time_limit=None, verbose=True):
    """
    Build deterministic baseline model under mean durations.
    Steps:
      1) Build deterministic MILP where durations = mean_duration for each activity.
         Solve for first-stage variables x, sa1, sa2, q and deterministic b[a].
      2) Fix the first-stage variables from step 1, and for each scenario k solve
         the second-stage (only b[a,k] continuous) to compute Q_k.
      3) Return baseline first-stage solution and average Q over scenarios.
    """
    idx = build_index_sets(data)
    A = idx["A"]
    A_info = idx["A_info"]
    J = idx["J"]
    G = idx["G"]
    Jg = idx["Jg"]
    eligible = idx["eligible_j_for_a"]
    num_scenarios = data["num_scenarios"]
    scenarios = list(range(num_scenarios))

    # Va_g set to 1 for each required type
    Va_g = {}
    for a in A:
        for g in A_info[a]["required_types"]:
            Va_g[(a,g)] = 1

    k_j = {j: 1 for j in J}
    t_a = {a: (A_info[a]["scheduled_start"] if A_info[a]["is_start"] else None) for a in A}
    pre = {a: A_info[a]["predecessor"] for a in A}
    # mean durations
    mean_d = {a: float(A_info[a]["mean_duration"]) for a in A}

    # Build deterministic model (single scenario with b[a])
    model = Model("Baseline_mean_det")
    if time_limit is not None:
        model.setParam('TimeLimit', time_limit)
    if not verbose:
        model.setParam('OutputFlag', 0)

    # First-stage vars
    x = {}
    for a in A:
        for j in eligible[a]:
            x[(a,j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{a}_{j}")
    sa1 = {}
    sa2 = {}
    share_pairs = []
    for a in A:
        for a2 in A:
            if a == a2:
                continue
            if len(set(A_info[a]["required_types"]) & set(A_info[a2]["required_types"])) > 0:
                share_pairs.append((a,a2))
                sa1[(a,a2)] = model.addVar(vtype=GRB.BINARY, name=f"sa1_{a}_{a2}")
                sa2[(a,a2)] = model.addVar(vtype=GRB.BINARY, name=f"sa2_{a}_{a2}")
    q = {}
    for (a,a2) in share_pairs:
        types_inter = set(A_info[a]["required_types"]) & set(A_info[a2]["required_types"])
        js = []
        for g in types_inter:
            js.extend(Jg[g])
        js = list(sorted(set(js)))
        for j in js:
            q[(j,a,a2)] = model.addVar(vtype=GRB.BINARY, name=f"q_{j}_{a}_{a2}")
    model.update()

    # deterministic b[a] (single)
    b = {}
    for a in A:
        b[a] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"b_{a}")
    model.update()

    # constraints same structure as SMILP but with mean durations
    # (1b)
    for a in A:
        for g in A_info[a]["required_types"]:
            lhs = quicksum(x[(a,j)] for j in Jg[g] if (a,j) in x)
            model.addConstr(lhs == Va_g[(a,g)], name=f"assign_count_a{a}_g{g}")

    # (1c)
    for (j,a,a2), qvar in q.items():
        model.addConstr(qvar >= sa1[(a,a2)] + sa2[(a,a2)] + x.get((a,j),0) + x.get((a2,j),0) - 3,
                        name=f"q_lin1_{j}_{a}_{a2}")

    # (1d)
    for g in G:
        for j in Jg[g]:
            acts = idx["A_g"].get(g, [])
            for a in acts:
                lhs = quicksum(q[(j,a,a2)] for a2 in acts if a2 != a and (j,a,a2) in q)
                model.addConstr(lhs <= k_j[j] - 1, name=f"cap_q_j{j}_a{a}")

    # (2b) and (2c) with mean durations
    for a in A:
        if A_info[a]["is_start"]:
            ta = t_a[a]
            for a_var in [a]:
                model.addConstr(b[a_var] - ta >= 0, name=f"init_time_a{a}")
        else:
            prev = pre[a]
            model.addConstr(b[a] - b[prev] - mean_d[prev] >= 0, name=f"succ_time_a{a}")

    # (2d)-(2g) with mean durations
    for g in G:
        acts = idx["A_g"].get(g, [])
        for a in acts:
            for a2 in acts:
                if a == a2:
                    continue
                # (2d)
                model.addConstr(M * sa1[(a,a2)] >= b[a] - b[a2] + 1, name=f"2d_g{g}_a{a}_a2{a2}")
                model.addConstr(M * (1 - sa1[(a,a2)]) >= b[a2] - b[a], name=f"2e_g{g}_a{a}_a2{a2}")
                model.addConstr(M * sa2[(a,a2)] >= b[a2] - b[a] + mean_d[a2], name=f"2f_g{g}_a{a}_a2{a2}")
                model.addConstr(M * (1 - sa2[(a,a2)]) >= b[a] - b[a2] - mean_d[a2] + 1, name=f"2g_g{g}_a{a}_a2{a2}")

    # Objective: total waiting under mean durations
    terms = []
    for a in A:
        if A_info[a]["is_start"]:
            terms.append(b[a] - t_a[a])
        else:
            p = pre[a]
            terms.append(b[a] - b[p] - mean_d[p])
    model.setObjective(quicksum(terms), GRB.MINIMIZE)

    model.update()
    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        print("Baseline deterministic model did not solve optimally; status:", model.Status)

    # Extract first-stage solution
    x_sol = {(a,j): x[(a,j)].X for (a,j) in x}
    sa1_sol = {(a,a2): sa1[(a,a2)].X for (a,a2) in sa1}
    sa2_sol = {(a,a2): sa2[(a,a2)].X for (a,a2) in sa2}
    q_sol = {(j,a,a2): q[(j,a,a2)].X for (j,a,a2) in q}

    # Evaluation: fix these first-stage binaries, solve second-stage for each scenario separately (only b vars)
    # We'll create a small LP for each scenario k
    Q_vals = []
    for k in scenarios:
        eval_model = Model(f"baseline_eval_k{k}")
        eval_model.setParam('OutputFlag', 0)
        # b[a,k] continuous
        b_k = {a: eval_model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"b_{a}") for a in A}
        eval_model.update()

        # (2b)
        for a in A:
            if A_info[a]["is_start"]:
                eval_model.addConstr(b_k[a] - t_a[a] >= 0, name=f"init_a{a}_k{k}")
            else:
                prev = pre[a]
                eval_model.addConstr(b_k[a] - b_k[prev] - data["activities"][0]["durations"][k] >= 0, name=f"succ_a{a}_k{k}")
                # Note: above line is placeholder; below we'll use actual d[(prev,k)]
        # But we need to use correct durations per activity/version:
        # Rebuild with correct d[(a,k)]
        eval_model.remove(eval_model.getConstrs())  # remove possibly bad constraints; rebuild correctly
        for a in A:
            if A_info[a]["is_start"]:
                eval_model.addConstr(b_k[a] - t_a[a] >= 0, name=f"init_a{a}_k{k}")
            else:
                prev = pre[a]
                duration_prev = float(A_info[prev]["durations"][k])
                eval_model.addConstr(b_k[a] - b_k[prev] - duration_prev >= 0, name=f"succ_a{a}_k{k}")

        # sequencing constraints (2d)-(2g) using fixed sa1/sa2 values from deterministic solution
        for g in G:
            acts = idx["A_g"].get(g, [])
            for a in acts:
                for a2 in acts:
                    if a == a2:
                        continue
                    val_sa1 = sa1_sol.get((a,a2), 0)
                    val_sa2 = sa2_sol.get((a,a2), 0)
                    # implement as inequalities (no binaries)
                    eval_model.addConstr(M * val_sa1 >= b_k[a] - b_k[a2] + 1, name=f"eval_2d_g{g}_a{a}_a2{a2}_k{k}")
                    eval_model.addConstr(M * (1 - val_sa1) >= b_k[a2] - b_k[a], name=f"eval_2e_g{g}_a{a}_a2{a2}_k{k}")
                    duration_a2 = float(A_info[a2]["durations"][k])
                    eval_model.addConstr(M * val_sa2 >= b_k[a2] - b_k[a] + duration_a2, name=f"eval_2f_g{g}_a{a}_a2{a2}_k{k}")
                    eval_model.addConstr(M * (1 - val_sa2) >= b_k[a] - b_k[a2] - duration_a2 + 1, name=f"eval_2g_g{g}_a{a}_a2{a2}_k{k}")

        # Objective Q_k
        terms_k = []
        for a in A:
            if A_info[a]["is_start"]:
                terms_k.append(b_k[a] - t_a[a])
            else:
                prev = pre[a]
                duration_prev = float(A_info[prev]["durations"][k])
                terms_k.append(b_k[a] - b_k[prev] - duration_prev)
        eval_model.setObjective(quicksum(terms_k), GRB.MINIMIZE)
        eval_model.update()
        eval_model.optimize()

        if eval_model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
            Q_vals.append(eval_model.ObjVal)
        else:
            print(f"Eval scenario k {k} not optimal; status {eval_model.Status}")
            Q_vals.append(10000) # penalty


    v_base = sum(Q_vals) / len(Q_vals)

    return {
        "first_stage": {
            "x": x_sol,
            "sa1": sa1_sol,
            "sa2": sa2_sol,
            "q": q_sol
        },
        "v_base": v_base,
        "Q_vals": Q_vals
    }

def plot_comparison_boxplot(smilp_Q, baseline_Q, infeasible_threshold=9000):
    """
    smilp_Q: dict {k: Q_k}
    baseline_Q: list [Q_k]
    infeasible_threshold: values >= this will be removed
    """

    # 轉成 list
    smilp_vals = [v for v in smilp_Q.values() if v < infeasible_threshold]
    baseline_vals = [v for v in baseline_Q if v < infeasible_threshold]

    # 檢查是否有剩資料
    if len(smilp_vals) == 0 or len(baseline_vals) == 0:
        print("Warning: no feasible scenario values to plot.")
        return

    plt.figure(figsize=(8, 5))
    plt.boxplot([smilp_vals, baseline_vals], labels=["SMILP", "Baseline"], showmeans=True)

    plt.ylabel("Scenario Q value")
    plt.title("Comparison of SMILP vs Baseline (Feasible Scenarios Only)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    # plt.show()
    plt.savefig("smilp_vs_baseline_boxplot.png")
# ---------------------------
# Example usage with your provided data dict:
if __name__ == "__main__":
    generator = InstanceGenerator(num_patients=10, arrival_interval=10, random_seed=42)
    data = generator.generate_data(num_scenarios=30)
    # Solve SMILP SAA deterministic equivalent
    print("Solving SMILP (SAA deterministic equivalent)...")
    model, sol = build_smilp_model(data, M=500, time_limit=600, verbose=True)
    print("SMILP Obj:", sol["obj"])

    # Solve baseline deterministic and evaluate baseline
    print("Solving baseline deterministic then evaluating on scenarios...")
    baseline = build_baseline_mean_model(data, M=500, time_limit=600, verbose=True)
    print("Baseline v_base (avg over scenarios):", baseline["v_base"])
    print("Baseline Q per scenario:", baseline["Q_vals"])
    plot_comparison_boxplot(sol["Q_per_scenario"], baseline["Q_vals"])

