import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve_deterministic_model(data, time_limit=60, bigM=10000, random_seed=42):
    np.random.seed(random_seed)
    activities = data['activities']
    resources = data['resources']
    num_acts = len(activities)
    
    m = gp.Model("Deterministic_MILP_Baseline")
    
    Va_g = [{} for _ in range(num_acts)]
    for a in activities:
        a_id = a['id']
        for g in a['required_resources']:
            Va_g[a_id][g] = Va_g[a_id].get(g, 0) + 1

    resource_instances = {}
    resource_capacities = {}
    global_inst_id = 0
    for r_name, r_info in resources.items():
        type_id = r_info['id']
        cap = r_info['capacity']
        resource_instances[type_id] = []
        resource_capacities[type_id] = cap
        for k in range(cap):
            resource_instances[type_id].append(global_inst_id)
            global_inst_id += 1
    
    x = {}
    for a in activities:
        a_id = a['id']
        for g, req_cnt in Va_g[a_id].items():
            insts = resource_instances.get(g, [])
            for inst in insts:
                x[(a_id, inst)] = m.addVar(vtype=GRB.BINARY, name=f"x_{a_id}_{inst}")
            m.addConstr(gp.quicksum(x[(a_id, inst)] for inst in insts) == req_cnt)
    
    conflict_pairs = []
    act_types = [set(Va_g[a['id']].keys()) for a in activities]
    for i in range(num_acts):
        for j in range(i + 1, num_acts):
            if act_types[i].intersection(act_types[j]):
                conflict_pairs.append((i, j))
    
    s1, s2 = {}, {}
    for (i, j) in conflict_pairs:
        s1[(i, j)] = m.addVar(vtype=GRB.BINARY, name=f"s1_{i}_{j}")
        s2[(i, j)] = m.addVar(vtype=GRB.BINARY, name=f"s2_{i}_{j}")
        m.addConstr(s1[(i, j)] + s2[(i, j)] <= 1)
    
    b = {}
    for a in activities:
        b[a['id']] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
    
    for a in activities:
        if a['is_start']:
            m.addConstr(b[a['id']] >= a['scheduled_start'])
        else:
            prev = a['predecessor']
            m.addConstr(b[a['id']] >= b[prev] + activities[prev]['mean_duration'])
            
    for (i, j) in conflict_pairs:
        di = activities[i]['mean_duration']
        dj = activities[j]['mean_duration']
        m.addConstr(bigM * s1[(i, j)] >= b[i] - b[j] + 1)
        m.addConstr(bigM * (1 - s1[(i, j)]) >= b[j] - b[i])
        m.addConstr(bigM * s2[(i, j)] >= b[j] - b[i] + dj)
        m.addConstr(bigM * (1 - s2[(i, j)]) >= b[i] - b[j] - dj + 1)
    
    total_wait = gp.LinExpr()
    for a in activities:
        if a['is_start']:
            total_wait += b[a['id']] - a['scheduled_start']
        else:
            prev = a['predecessor']
            total_wait += b[a['id']] - (b[prev] + activities[prev]['mean_duration'])
            
    m.setObjective(total_wait, GRB.MINIMIZE)
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', time_limit)
    m.setParam('MIPGap', 0.001)
    m.optimize()
    return m

def evaluate_baseline_mean_value_model(baseline_model, data, validation_scenarios, time_limit=300, bigM=10000):
    """
    Evaluates Baseline using EXTERNAL validation_scenarios.
    validation_scenarios: dict {scenario_index: {activity_id: duration}}
    """
    if baseline_model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
        return float('inf'), []
        
    activities = data['activities']
    resources = data['resources']
    num_acts = len(activities)

    # Extract fixed decisions
    x_sol = {}
    for v in baseline_model.getVars():
        if v.VarName.startswith("x_") and v.X > 0.5:
            parts = v.VarName.split('_')
            x_sol[(int(parts[1]), int(parts[2]))] = True
            
    s1_sol = {}
    s2_sol = {}
    for v in baseline_model.getVars():
        if v.X > 0.5:
            if v.VarName.startswith("s1_"): s1_sol[(int(v.VarName.split('_')[1]), int(v.VarName.split('_')[2]))] = True
            elif v.VarName.startswith("s2_"): s2_sol[(int(v.VarName.split('_')[1]), int(v.VarName.split('_')[2]))] = True

    resource_instances = {}
    global_inst_id = 0
    for r_name, r_info in resources.items():
        type_id = r_info['id']
        cap = r_info['capacity']
        resource_instances[type_id] = []
        for k in range(cap):
            resource_instances[type_id].append(global_inst_id)
            global_inst_id += 1

    objectives = []
    # Loop over the provided scenario indices
    for k, durations in validation_scenarios.items():
        m_eval = gp.Model(f"Base_Eval_{k}")
        b_eval = {a['id']: m_eval.addVar(vtype=GRB.CONTINUOUS, lb=0.0) for a in activities}
        
        for a in activities:
            if a['is_start']:
                m_eval.addConstr(b_eval[a['id']] >= a['scheduled_start'])
            else:
                prev = a['predecessor']
                m_eval.addConstr(b_eval[a['id']] >= b_eval[prev] + durations[prev])
                
        act_types = [set(a['required_resources']) for a in activities]
        for i in range(num_acts):
            for j in range(i + 1, num_acts):
                if act_types[i].intersection(act_types[j]):
                    # Use the fixed s1/s2 from baseline
                    is_s1 = (i, j) in s1_sol
                    is_s2 = (i, j) in s2_sol
                    di, dj = durations[i], durations[j]
                    
                    val_s1 = 1 if is_s1 else 0
                    val_s2 = 1 if is_s2 else 0
                    
                    m_eval.addConstr(bigM * val_s1 >= b_eval[i] - b_eval[j] + 1)
                    m_eval.addConstr(bigM * (1 - val_s1) >= b_eval[j] - b_eval[i])
                    m_eval.addConstr(bigM * val_s2 >= b_eval[j] - b_eval[i] + dj)
                    m_eval.addConstr(bigM * (1 - val_s2) >= b_eval[i] - b_eval[j] - dj + 1)

        total_wait = gp.LinExpr()
        for a in activities:
            if a['is_start']:
                total_wait += b_eval[a['id']] - a['scheduled_start']
            else:
                prev = a['predecessor']
                total_wait += b_eval[a['id']] - (b_eval[prev] + durations[prev])
                
        m_eval.setObjective(total_wait, GRB.MINIMIZE)
        m_eval.setParam('OutputFlag', 0)
        m_eval.optimize()
        if m_eval.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            objectives.append(m_eval.ObjVal)
            
    return np.mean(objectives), objectives