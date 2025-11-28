import gurobipy as gp
from gurobipy import GRB

def solve_deterministic_model(data):
    """
    傳統方法：忽略變異，僅使用平均時間 (Mean Duration) 求解 MILP
    """
    try:
        m = gp.Model("Deterministic_Patient_Scheduling")
        activities = data['activities']
        resources = data['resources']
        num_acts = len(activities)
        M = 10000
        
        # 變數: b (開始時間)
        b = m.addVars(num_acts, vtype=GRB.CONTINUOUS, name="b", lb=0)
        
        # 為了模擬傳統 Job Shop，我們這裡使用最簡單的 Disjunctive Constraints
        # x[a, k]: 指派給第 k 台機器 (對於 capacity > 1 的資源)
        x = {}
        for a in activities:
            r_type = a['resource_type']
            r_cap = [v['capacity'] for k, v in resources.items() if v['id'] == r_type][0]
            for k in range(r_cap):
                x[a['id'], r_type, k] = m.addVar(vtype=GRB.BINARY, name=f"x_{a['id']}_{k}")

        # 目標: 最小化總等待時間 (基於平均時間)
        total_wait = 0
        for a in activities:
            if a['is_start']:
                wait = b[a['id']] - a['scheduled_start']
            else:
                prev_id = a['predecessor']
                # 關鍵差異：這裡只用 mean_duration
                prev_dur = activities[prev_id]['mean_duration'] 
                wait = b[a['id']] - (b[prev_id] + prev_dur)
            total_wait += wait
            
        m.setObjective(total_wait, GRB.MINIMIZE)
        
        # 限制式
        for a in activities:
            # 1. 資源指派
            r_type = a['resource_type']
            r_cap = [v['capacity'] for k, v in resources.items() if v['id'] == r_type][0]
            m.addConstr(gp.quicksum(x[a['id'], r_type, k] for k in range(r_cap)) == 1)
            
            # 2. 流程順序
            if not a['is_start']:
                prev_id = a['predecessor']
                prev_dur = activities[prev_id]['mean_duration']
                m.addConstr(b[a['id']] >= b[prev_id] + prev_dur)
            else:
                m.addConstr(b[a['id']] >= a['scheduled_start'])

        # 3. 資源不重疊 (Disjunctive)
        for i in range(num_acts):
            for j in range(i + 1, num_acts):
                act1 = activities[i]
                act2 = activities[j]
                if act1['resource_type'] == act2['resource_type']:
                    r_type = act1['resource_type']
                    r_cap = [v['capacity'] for k, v in resources.items() if v['id'] == r_type][0]
                    for k in range(r_cap):
                        # 如果都在機器 k 上，必須有先後
                        seq = m.addVar(vtype=GRB.BINARY, name=f"seq_{i}_{j}_{k}")
                        
                        dur_i = act1['mean_duration']
                        dur_j = act2['mean_duration']
                        
                        # Big-M Constraint
                        lhs1 = b[act1['id']] + dur_i
                        rhs1 = b[act2['id']] + M * (1 - seq) + M * (2 - x[act1['id'], r_type, k] - x[act2['id'], r_type, k])
                        m.addConstr(lhs1 <= rhs1)
                        
                        lhs2 = b[act2['id']] + dur_j
                        rhs2 = b[act1['id']] + M * seq + M * (2 - x[act1['id'], r_type, k] - x[act2['id'], r_type, k])
                        m.addConstr(lhs2 <= rhs2)

        m.setParam('OutputFlag', 0) # 關閉 log 保持乾淨
        m.optimize()
        return m
    except Exception as e:
        print(e)
        return None