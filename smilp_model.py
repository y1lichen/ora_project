import gurobipy as gp
from gurobipy import GRB

def solve_smilp_model(data):

    m = gp.Model("Stochastic_Patient_Scheduling_SMILP")
    
    activities = data['activities']
    resources = data['resources']
    num_scenarios = data['num_scenarios']
    num_acts = len(activities)
    M = 10000
    
    # 第一階段
    s1 = {}
    s2 = {}
    q = {}

    pairs = []
    for i in range(num_acts):
        for j in range(num_acts):
            if i == j: continue
            act_i = activities[i]
            act_j = activities[j]
            if act_i['resource_type'] == act_j['resource_type']:
                pairs.append((i, j, act_i['resource_type']))
                
    for (i, j, r_type) in pairs:
        s1[i, j] = m.addVar(vtype=GRB.BINARY, name=f"s1_{i}_{j}")
        s2[i, j] = m.addVar(vtype=GRB.BINARY, name=f"s2_{i}_{j}")
        # q: indicates if j is in progress when i starts
        q[i, j] = m.addVar(vtype=GRB.BINARY, name=f"q_{i}_{j}")

    # 第二階段
    b = m.addVars(num_acts, num_scenarios, vtype=GRB.CONTINUOUS, lb=0, name="b")
    

   
    obj_expr = 0
    for n in range(num_scenarios):
        scenario_wait = 0
        for a in activities:
            if a['is_start']:
                # Wait = start - scheduled
                scenario_wait += (b[a['id'], n] - a['scheduled_start'])
            else:
                # Wait = start - (prev_start + prev_duration_n)
                prev_id = a['predecessor']
                prev_dur = activities[prev_id]['durations'][n]
                scenario_wait += (b[a['id'], n] - (b[prev_id, n] + prev_dur))
        obj_expr += scenario_wait
        
    m.setObjective((1.0/num_scenarios) * obj_expr, GRB.MINIMIZE)
    
   
    for n in range(num_scenarios):
        for a in activities:
            if a['is_start']:
                m.addConstr(b[a['id'], n] >= a['scheduled_start'])
            else:
                prev_id = a['predecessor']
                prev_dur = activities[prev_id]['durations'][n]
                m.addConstr(b[a['id'], n] >= b[prev_id, n] + prev_dur)

    for n in range(num_scenarios):
        for (i, j, r_type) in pairs:
            act_i = activities[i] # a
            act_j = activities[j] # a'
            dur_j = act_j['durations'][n]
            
            m.addConstr(b[i, n] >= b[j, n] - M * (1 - s1[i, j]))
        
            m.addConstr(b[j, n] >= b[i, n] + 1 - M * s1[i, j])
            
            m.addConstr(b[i, n] <= b[j, n] + dur_j - 1 + M * (1 - s2[i, j]))
            
            m.addConstr(b[i, n] >= b[j, n] + dur_j - M * s2[i, j])

    for (i, j, r_type) in pairs:
        m.addConstr(q[i, j] >= s1[i, j] + s2[i, j] - 1)
        m.addConstr(q[i, j] <= s1[i, j])
        m.addConstr(q[i, j] <= s2[i, j])
    
    for i in range(num_acts):
        r_type = activities[i]['resource_type']
        cap = [v['capacity'] for k, v in resources.items() if v['id'] == r_type][0]
        
        competitors = [q[i, j] for (p_i, j, p_r) in pairs if p_i == i]
        
        if competitors:
            m.addConstr(gp.quicksum(competitors) <= cap - 1, name=f"cap_{i}")

    # 設定參數
    m.setParam('OutputFlag', 1)
    # m.setParam('MIPGap', 0.01) # 設定 1% Gap
    m.setParam('TimeLimit', 10800)
    m.optimize()
    return m