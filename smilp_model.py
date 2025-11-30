import gurobipy as gp
from gurobipy import GRB

def solve_smilp_model(data):
    m = gp.Model("Stochastic_Patient_Scheduling_SMILP_Strict")
    
    activities = data['activities']
    resources = data['resources']
    num_scenarios = data['num_scenarios']
    num_acts = len(activities)
    M = 10000 # Big-M constant
    
    # 1. 資源實例展開 (Resource Instances)
    # 將資源從類別 (例如 "Provider", cap=2) 展開為個體 (Provider_0, Provider_1)
    resource_instances = [] 
    res_type_map = {} # type_id -> list of instance_ids
    
    global_res_id = 0
    for r_name, r_info in resources.items():
        r_type = r_info['id']
        capacity = r_info['capacity']
        
        current_instances = []
        # 對於容量非常大的資源(如Intake/Discharge)，通常不需排程限制，此處簡化處理
        # 若 capacity > 10，視為無限資源，只建立一個虛擬 instance 但不加互斥限制
        real_cap = capacity if capacity < 10 else 1 
        
        for k in range(real_cap):
            res_instance = (r_type, global_res_id)
            resource_instances.append(res_instance)
            current_instances.append(global_res_id)
            global_res_id += 1
        res_type_map[r_type] = current_instances

    # --- 變數定義 ---
    # x[i, k]: 活動 i 是否指派給資源實例 k
    x = {} 
    
    # s1[i, j]: 活動 i 是否在 j 之前 (僅針對同資源類型)
    # s2[i, j]: 活動 j 是否在 i 之前
    s1 = {}
    s2 = {}
    
    # b[i, n]: 活動 i 在情境 n 下的開始時間
    b = m.addVars(num_acts, num_scenarios, vtype=GRB.CONTINUOUS, lb=0, name="b")

    # 建立需要檢查衝突的 Activity Pairs (同資源類型的活動)
    pairs = []
    for i in range(num_acts):
        for j in range(num_acts):
            if i == j: continue
            # 只有當兩個活動需要「相同類型的資源」時，才需要決定順序
            if activities[i]['resource_type'] == activities[j]['resource_type']:
                pairs.append((i, j))

    # --- 限制式 ---

    # 1. 指派限制 (Assignment Constraints)
    for i in range(num_acts):
        r_type = activities[i]['resource_type']
        valid_instances = res_type_map.get(r_type, [])
        
        # 如果是無限資源，不需指派變數(或指派給dummy)，這裡假設有限資源才需要 x
        r_cap = resources[[name for name, r in resources.items() if r['id'] == r_type][0]]['capacity']
        
        if r_cap < 10: # 僅針對瓶頸資源 (有限容量)
            vars_x = []
            for r_inst_id in valid_instances:
                x[i, r_inst_id] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{r_inst_id}")
                vars_x.append(x[i, r_inst_id])
            
            # 每個活動必須指派給該類型中的「恰好一個」實例
            m.addConstr(gp.quicksum(vars_x) == 1, name=f"assign_{i}")

    # 2. 順序變數定義與互斥 (Sequencing Variables)
    for (i, j) in pairs:
        s1[i, j] = m.addVar(vtype=GRB.BINARY, name=f"s1_{i}_{j}")
        s2[i, j] = m.addVar(vtype=GRB.BINARY, name=f"s2_{i}_{j}")
        
        # 互斥限制：不可能 i 在 j 前，同時 j 也在 i 前
        m.addConstr(s1[i, j] + s2[i, j] <= 1, name=f"mutex_seq_{i}_{j}")

        # [關鍵修正 1] 強制排序限制 (Force Sequencing)
        # 如果 i 和 j 被指派給「同一個資源實例 k」，則它們必須有先後順序 (s1=1 或 s2=1)
        r_type = activities[i]['resource_type']
        r_cap = resources[[name for name, r in resources.items() if r['id'] == r_type][0]]['capacity']
        
        if r_cap < 10: # 只對瓶頸資源有效
            valid_instances = res_type_map.get(r_type, [])
            for r_inst_id in valid_instances:
                # 若 x[i,k]=1 且 x[j,k]=1，則 s1+s2 >= 1
                m.addConstr(s1[i, j] + s2[i, j] >= x[i, r_inst_id] + x[j, r_inst_id] - 1, 
                           name=f"force_seq_{i}_{j}_{r_inst_id}")

    # 3. 路徑流程限制 (Care Pathway Precedence)
    # 這部分是每個人內部的流程 (Intake -> X-ray)，與資源衝突無關
    for n in range(num_scenarios):
        for a in activities:
            if a['is_start']:
                # 第一個活動受限於到達時間
                m.addConstr(b[a['id'], n] >= a['scheduled_start'], name=f"arrival_{a['id']}_{n}")
            else:
                # 後續活動受限於前一個活動結束
                prev_id = a['predecessor']
                prev_dur = activities[prev_id]['durations'][n]
                m.addConstr(b[a['id'], n] >= b[prev_id, n] + prev_dur, name=f"flow_{a['id']}_{n}")

    # 4. Big-M 資源不重疊限制 (Resource Non-overlap)
    # [關鍵修正 2] 修正 Big-M 邏輯，加入持續時間
    for n in range(num_scenarios):
        for (i, j) in pairs:
            # 只有瓶頸資源需要這個限制
            r_type = activities[i]['resource_type']
            r_cap = resources[[name for name, r in resources.items() if r['id'] == r_type][0]]['capacity']
            if r_cap >= 10: continue

            dur_i = activities[i]['durations'][n]
            dur_j = activities[j]['durations'][n]
            
            # 若 s1=1 (i 在 j 前)，則 Start_j >= Start_i + Duration_i
            m.addConstr(b[j, n] >= b[i, n] + dur_i - M * (1 - s1[i, j]), name=f"bigM_s1_{i}_{j}_{n}")
            
            # 若 s2=1 (j 在 i 前)，則 Start_i >= Start_j + Duration_j
            m.addConstr(b[i, n] >= b[j, n] + dur_j - M * (1 - s2[i, j]), name=f"bigM_s2_{i}_{j}_{n}")

    # --- 目標函數 (Objective) ---
    # 最小化期望總等待時間
    obj_expr = 0
    for n in range(num_scenarios):
        scenario_wait = 0
        for a in activities:
            # 等待時間 = 實際開始時間 - 最早可開始時間
            if a['is_start']:
                ready_time = a['scheduled_start']
            else:
                prev_id = a['predecessor']
                ready_time = b[prev_id, n] + activities[prev_id]['durations'][n] # 這是變數表達式
            
            # Gurobi 會自動處理 (b - expr)
            # 注意：這裡的 "wait" 定義要與論文一致。
            # 通常 Total Wait = Sum(Actual Start - Arrival) - Sum(Service Durations) ? 
            # 或者單純定義為每個步驟的等待: Start - Ready。
            # 這裡使用 (Start - Ready) 加總
            
            # 由於 ready_time 包含變數 b[prev]，直接加總可能會變得很複雜
            # 簡單寫法：Total Time in System - Total Service Time
            # 但為了精確對應 "Wait"，我們用 (Start - Ready)
            
            if a['is_start']:
                 scenario_wait += (b[a['id'], n] - a['scheduled_start'])
            else:
                 prev_id = a['predecessor']
                 prev_dur = activities[prev_id]['durations'][n]
                 scenario_wait += (b[a['id'], n] - (b[prev_id, n] + prev_dur))

        obj_expr += scenario_wait
        
    m.setObjective((1.0/num_scenarios) * obj_expr, GRB.MINIMIZE)
    
    # --- 參數設定 ---
    m.setParam('OutputFlag', 1)
    m.setParam('MIPGap', 0.05) # 5% Gap 就停止，加快速度
    m.setParam('TimeLimit', 600) # 10分鐘限制
    m.optimize()
    
    return m