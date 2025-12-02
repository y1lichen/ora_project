import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random

def solve_smilp_mco(data, num_saa_samples=10, num_mco_trials=5,
                    time_limit=300, gap=0.05, bigM=10000, random_seed=42):
    """
    SMILP + SAA + MCO
    - num_saa_samples: 每組 SAA sample 數量
    - num_mco_trials: 生成幾組 SAA samples，挑選最佳 assignment
    """

    np.random.seed(random_seed)
    random.seed(random_seed)

    activities = data['activities']
    resources = data['resources']
    num_acts = len(activities)

    # 展開資源實例
    resource_instances = {}
    inst_to_type = {}
    global_inst_id = 0
    for r_name, r_info in resources.items():
        type_id = r_info['id']
        cap = r_info['capacity']
        real_cap = cap if cap < 10 else 1
        resource_instances[type_id] = []
        for k in range(real_cap):
            resource_instances[type_id].append(global_inst_id)
            inst_to_type[global_inst_id] = type_id
            global_inst_id += 1

    # 活動多資源需求
    Va_g = [{} for _ in range(num_acts)]
    for a in activities:
        a_id = a['id']
        for g in a['required_resources']:
            Va_g[a_id][g] = Va_g[a_id].get(g, 0) + 1

    best_wait = float('inf')
    best_assignment = None
    best_order = None

    for trial in range(num_mco_trials):
        print(f"\n[MCO Trial {trial+1}/{num_mco_trials}]")

        # 生成一組 SAA sample
        realized_durations = {}
        b = {}
        for k in range(num_saa_samples):
            realized_durations[k] = {}
            for a in activities:
                a_id = a['id']
                mu_val = a['mean_duration']
                sigma_val = mu_val * 0.3
                phi = np.sqrt(sigma_val**2 + mu_val**2)
                log_mu = np.log(mu_val**2 / phi)
                log_sigma = np.sqrt(np.log(phi**2 / mu_val**2))
                dur = np.random.lognormal(log_mu, log_sigma)
                realized_durations[k][a_id] = max(1, round(dur))
                b[(a_id, k)] = None  # 先 placeholder

        # 建立模型
        m = gp.Model(f"SMILP_MCO_trial_{trial}")
        x, s1, s2 = {}, {}, {}
        conflict_pairs = []

        act_types = [set(Va_g[a['id']].keys()) for a in activities]
        for i in range(num_acts):
            for j in range(i + 1, num_acts):
                shared = act_types[i].intersection(act_types[j])
                if shared:
                    conflict_pairs.append((i, j, shared))

        # 資源 assignment 變數
        for a in activities:
            a_id = a['id']
            for g, req_cnt in Va_g[a_id].items():
                insts = resource_instances.get(g, [])
                for inst in insts:
                    x[(a_id, inst)] = m.addVar(vtype=GRB.BINARY, name=f"x_{a_id}_{inst}")
                m.addConstr(gp.quicksum(x[(a_id, inst)] for inst in insts) == req_cnt,
                            name=f"assign_cnt_a{a_id}_g{g}")

        # 排序變數
        for (i, j, shared) in conflict_pairs:
            s1[(i, j)] = m.addVar(vtype=GRB.BINARY, name=f"s1_{i}_{j}")
            s2[(i, j)] = m.addVar(vtype=GRB.BINARY, name=f"s2_{i}_{j}")
            m.addConstr(s1[(i, j)] + s2[(i, j)] <= 1)
            for g in shared:
                insts = resource_instances.get(g, [])
                for inst in insts:
                    xi = x.get((i, inst), None)
                    xj = x.get((j, inst), None)
                    if xi is not None and xj is not None:
                        m.addConstr(s1[(i, j)] + s2[(i, j)] >= xi + xj - 1)

        # 第二階段開始時間
        for k in range(num_saa_samples):
            for a in activities:
                a_id = a['id']
                b[(a_id, k)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"b_{a_id}_{k}")

        # 前置與不重疊約束
        for k in range(num_saa_samples):
            for a in activities:
                a_id = a['id']
                if a['is_start']:
                    m.addConstr(b[(a_id, k)] >= a['scheduled_start'])
                else:
                    prev_id = a['predecessor']
                    prev_dur = realized_durations[k][prev_id]
                    m.addConstr(b[(a_id, k)] >= b[(prev_id, k)] + prev_dur)

            for (i, j, shared) in conflict_pairs:
                dur_i = realized_durations[k][i]
                dur_j = realized_durations[k][j]
                m.addConstr(b[(j, k)] >= b[(i, k)] + dur_i - bigM * (1 - s1[(i, j)]))
                m.addConstr(b[(i, k)] >= b[(j, k)] + dur_j - bigM * (1 - s2[(i, j)]))

        # 目標: SAA 平均等待
        total_wait = gp.LinExpr()
        for k in range(num_saa_samples):
            for a in activities:
                a_id = a['id']
                if a['is_start']:
                    total_wait += b[(a_id, k)] - a['scheduled_start']
                else:
                    prev_id = a['predecessor']
                    prev_dur = realized_durations[k][prev_id]
                    total_wait += b[(a_id, k)] - (b[(prev_id, k)] + prev_dur)
        total_wait /= num_saa_samples

        m.setObjective(total_wait, GRB.MINIMIZE)

        # 求解
        m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', time_limit)
        m.setParam('MIPGap', gap)
        m.optimize()

        if m.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
            avg_wait = m.ObjVal
            print(f"Trial {trial} average wait: {avg_wait:.2f}")
            if avg_wait < best_wait:
                best_wait = avg_wait
                # 儲存 assignment 與 order
                best_assignment = {k: [] for k in range(num_acts)}
                for (a_id, inst), var in x.items():
                    if var.X > 0.5:
                        best_assignment[a_id].append(inst)
                best_order = {}
                for key, var in s1.items():
                    if var.X > 0.5:
                        best_order[key] = True

    # 返回最終最佳策略（只含第一階段變數）
    class DummyModel:
        def __init__(self, assignment, order):
            self.assignment = assignment
            self.order = order
            self.Status = GRB.OPTIMAL
    return DummyModel(best_assignment, best_order)
