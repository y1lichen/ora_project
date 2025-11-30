import gurobipy as gp
from gurobipy import GRB

def solve_deterministic_model(data, time_limit=60):
    """
    傳統確定性方法 (Deterministic Method)
    特點：忽略變異，假設所有活動時間都等於平均值 (Mean Duration)。
    注意：為了讓 main.py 能統一提取排程，這裡使用了與 SMILP 相同的變數命名 (s1, s2, x)。
    """
    m = gp.Model("Deterministic_Model")
    
    activities = data['activities']
    resources = data['resources']
    num_acts = len(activities)
    M = 20000 
    
    # --- 1. 資源實例展開 ---
    resource_instances = {} 
    global_res_idx = 0
    for r_name, r_info in resources.items():
        r_type = r_info['id']
        cap = r_info['capacity']
        current_instances = []
        real_cap = cap if cap < 10 else 1 
        for k in range(real_cap):
            current_instances.append(global_res_idx)
            global_res_idx += 1
        resource_instances[r_type] = current_instances

    # --- 變數定義 ---
    x = {} 
    for i in range(num_acts):
        r_type = activities[i]['resource_type']
        valid_insts = resource_instances[r_type]
        r_cap = resources[[n for n, r in resources.items() if r['id'] == r_type][0]]['capacity']
        if r_cap < 10:
            vars_x_i = []
            for inst_id in valid_insts:
                x[i, inst_id] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{inst_id}")
                vars_x_i.append(x[i, inst_id])
            m.addConstr(gp.quicksum(vars_x_i) == 1, name=f"assign_{i}")

    s1 = {}
    s2 = {}
    pairs = []
    for i in range(num_acts):
        for j in range(num_acts):
            if i == j: continue
            if activities[i]['resource_type'] == activities[j]['resource_type']:
                pairs.append((i, j))
                
    for (i, j) in pairs:
        s1[i, j] = m.addVar(vtype=GRB.BINARY, name=f"s1_{i}_{j}")
        s2[i, j] = m.addVar(vtype=GRB.BINARY, name=f"s2_{i}_{j}")
        m.addConstr(s1[i, j] + s2[i, j] <= 1)
        r_type = activities[i]['resource_type']
        r_cap = resources[[n for n, r in resources.items() if r['id'] == r_type][0]]['capacity']
        if r_cap < 10:
            valid_insts = resource_instances[r_type]
            for inst_id in valid_insts:
                m.addConstr(s1[i, j] + s2[i, j] >= x[i, inst_id] + x[j, inst_id] - 1)

    # b: 這裡只需要一個場景 (Deterministic)
    b = m.addVars(num_acts, vtype=GRB.CONTINUOUS, lb=0, name="b")

    # --- 限制式 (使用 Mean Duration) ---
    
    # 1. 流程順序
    for a in activities:
        if a['is_start']:
            m.addConstr(b[a['id']] >= a['scheduled_start'])
        else:
            prev_id = a['predecessor']
            prev_dur = activities[prev_id]['mean_duration'] # 關鍵差異：使用平均值
            m.addConstr(b[a['id']] >= b[prev_id] + prev_dur)
    
    # 2. 資源不重疊
    for (i, j) in pairs:
        r_type = activities[i]['resource_type']
        r_cap = resources[[n for n, r in resources.items() if r['id'] == r_type][0]]['capacity']
        if r_cap >= 10: continue 
        
        dur_i = activities[i]['mean_duration']
        dur_j = activities[j]['mean_duration']
        
        m.addConstr(b[j] >= b[i] + dur_i - M * (1 - s1[i, j]))
        m.addConstr(b[i] >= b[j] + dur_j - M * (1 - s2[i, j]))

    # --- 目標函數 ---
    total_wait = 0
    for a in activities:
        if a['is_start']:
                total_wait += (b[a['id']] - a['scheduled_start'])
        else:
                prev_id = a['predecessor']
                prev_dur = activities[prev_id]['mean_duration']
                total_wait += (b[a['id']] - (b[prev_id] + prev_dur))
    
    m.setObjective(total_wait, GRB.MINIMIZE)
    
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', time_limit)
    m.setParam('MIPGap', 0.001) # 確定性問題通常解得很快，Gap 設小一點
    
    m.optimize()
    return m