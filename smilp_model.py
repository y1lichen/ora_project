import gurobipy as gp
from gurobipy import GRB

def solve_smilp_model(data, time_limit=300, gap=0.05):
    """
    兩階段隨機規劃模型 (SMILP)
    特點：考慮 N 個情境 (Scenarios) 的變異，優化期望等待時間。
    """
    m = gp.Model("SMILP_Model")
    
    activities = data['activities']
    resources = data['resources']
    num_scenarios = data['num_scenarios']
    num_acts = len(activities)
    M = 20000 # Big-M
    
    # --- 1. 資源實例展開 (Resource Instances) ---
    # 將資源從類別 (例如 "Provider", cap=2) 展開為個體 (Provider_0, Provider_1)
    # 這是為了支援論文中的 "Individual Assignment"
    resource_instances = {} # type_id -> list of global_instance_ids
    global_res_idx = 0
    
    for r_name, r_info in resources.items():
        r_type = r_info['id']
        cap = r_info['capacity']
        current_instances = []
        # 簡化：大容量資源(如Intake)視為無限，只建1個虛擬實例；瓶頸資源依容量展開
        real_cap = cap if cap < 10 else 1 
        
        for k in range(real_cap):
            current_instances.append(global_res_idx)
            global_res_idx += 1
        resource_instances[r_type] = current_instances

    # --- 變數定義 ---
    
    # x[i, k]: 指派變數 (Stage 1)
    x = {} 
    for i in range(num_acts):
        r_type = activities[i]['resource_type']
        valid_insts = resource_instances[r_type]
        r_cap = resources[[n for n, r in resources.items() if r['id'] == r_type][0]]['capacity']
        
        # 只有瓶頸資源需要明確指派
        if r_cap < 10:
            vars_x_i = []
            for inst_id in valid_insts:
                x[i, inst_id] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{inst_id}")
                vars_x_i.append(x[i, inst_id])
            # 每個活動必須指派給恰好一個實例
            m.addConstr(gp.quicksum(vars_x_i) == 1, name=f"assign_{i}")

    # s1, s2: 排序變數 (Stage 1)
    s1 = {}
    s2 = {}
    pairs = []
    
    # 找出所有可能衝突的活動對 (同資源類型)
    for i in range(num_acts):
        for j in range(num_acts):
            if i == j: continue
            if activities[i]['resource_type'] == activities[j]['resource_type']:
                pairs.append((i, j))
                
    for (i, j) in pairs:
        s1[i, j] = m.addVar(vtype=GRB.BINARY, name=f"s1_{i}_{j}") # i 在 j 前
        s2[i, j] = m.addVar(vtype=GRB.BINARY, name=f"s2_{i}_{j}") # j 在 i 前
        
        # 互斥：不可能互相在對方前面
        m.addConstr(s1[i, j] + s2[i, j] <= 1)
        
        # 強制排序：若 i 和 j 被指派給同一個實例，則必須決定順序
        r_type = activities[i]['resource_type']
        r_cap = resources[[n for n, r in resources.items() if r['id'] == r_type][0]]['capacity']
        
        if r_cap < 10:
            valid_insts = resource_instances[r_type]
            for inst_id in valid_insts:
                # 限制式邏輯： if x[i] and x[j] then s1+s2 >= 1
                # 轉換為： s1 + s2 >= x[i] + x[j] - 1
                m.addConstr(s1[i, j] + s2[i, j] >= x[i, inst_id] + x[j, inst_id] - 1)

    # b: 開始時間變數 (Stage 2 - Recourse)
    b = m.addVars(num_acts, num_scenarios, vtype=GRB.CONTINUOUS, lb=0, name="b")

    # --- 限制式 ---
    
    for n in range(num_scenarios):
        # 1. 流程順序 (Precedence Constraints)
        for a in activities:
            if a['is_start']:
                m.addConstr(b[a['id'], n] >= a['scheduled_start'])
            else:
                prev_id = a['predecessor']
                prev_dur = activities[prev_id]['durations'][n]
                m.addConstr(b[a['id'], n] >= b[prev_id, n] + prev_dur)
        
        # 2. 資源不重疊 (Resource Constraints with Big-M)
        for (i, j) in pairs:
            r_type = activities[i]['resource_type']
            r_cap = resources[[n for n, r in resources.items() if r['id'] == r_type][0]]['capacity']
            
            # 無限資源不需要 Big-M 互斥
            if r_cap >= 10: continue 
            
            dur_i = activities[i]['durations'][n]
            dur_j = activities[j]['durations'][n]
            
            # Big-M 限制：若 s1=1 (i before j)，則 Start_j >= Start_i + Dur_i
            m.addConstr(b[j, n] >= b[i, n] + dur_i - M * (1 - s1[i, j]))
            
            # Big-M 限制：若 s2=1 (j before i)，則 Start_i >= Start_j + Dur_j
            m.addConstr(b[i, n] >= b[j, n] + dur_j - M * (1 - s2[i, j]))

    # --- 目標函數 ---
    # 最小化 N 個情境的平均總等待時間
    obj_expr = 0
    for n in range(num_scenarios):
        scenario_wait = 0
        for a in activities:
            if a['is_start']:
                 scenario_wait += (b[a['id'], n] - a['scheduled_start'])
            else:
                 prev_id = a['predecessor']
                 prev_dur = activities[prev_id]['durations'][n]
                 scenario_wait += (b[a['id'], n] - (b[prev_id, n] + prev_dur))
        obj_expr += scenario_wait

    m.setObjective((1.0/num_scenarios) * obj_expr, GRB.MINIMIZE)
    
    # 求解參數
    # m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', time_limit)
    m.setParam('MIPGap', gap)
    m.setParam('MIPFocus', 1)
    
    m.optimize()
    return m