import gurobipy as gp
from gurobipy import GRB
import numpy as np

class SMILPSolver:
    def __init__(self, data, time_limit=3600, mip_gap=0.05, baseline_sol=None):
        """
        :param data: 來自 InstanceGenerator 的輸出字典
        :param time_limit: Gurobi 時間限制 (秒)
        :param mip_gap: 收斂 Gap
        """
        self.data = data
        self.patients = data['patients']
        self.activities = data['activities']
        self.resources = data['resources']
        self.num_scenarios = data['num_scenarios']
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.baseline_sol = baseline_sol
        
        # 為了建立約束，我們需要知道每個 Activity 潛在可用的資源集合 J_g
        # 在你的 generator 中，這已經被解析為具體的 ID 列表
        self.act_valid_resources = {act['id']: act['required_resources'] for act in self.activities}
        
        # Big-M (足夠大的數，約為總 horizon 長度)
        self.M = 10000 

        self.model = gp.Model("PatientScheduling_SMILP")
        self.model.setParam('TimeLimit', self.time_limit)
        self.model.setParam('MIPGap', self.mip_gap)
        self.model.setParam('OutputFlag', 1) # 設為 1 可看到求解過程

    def build_and_solve(self):
        m = self.model
        acts = self.activities
        scenarios = range(self.num_scenarios)
        
        # ==========================
        # Stage 1 Variables (Scenario Independent)
        # ==========================
        
        # x[a, j]: Binary, 1 if activity a is served by resource j
        x = {}
        for a in acts:
            for j in self.act_valid_resources[a['id']]:
                x[a['id'], j] = m.addVar(vtype=GRB.BINARY, name=f"x_{a['id']}_{j}")

        # 找出需要排序的 Activity Pairs (a, a')
        # 論文定義: a 和 a' 共享至少一種類型的資源 (這裡是共享具體資源ID)
        competition_pairs = []
        for i, a1 in enumerate(acts):
            for j, a2 in enumerate(acts):
                if i == j: continue
                # 檢查資源需求是否有交集
                res1 = set(self.act_valid_resources[a1['id']])
                res2 = set(self.act_valid_resources[a2['id']])
                if not res1.isdisjoint(res2):
                    competition_pairs.append((a1['id'], a2['id']))

        # s1[a, a']: 1 if a is assigned NO LATER THAN a' (Sequence order)
        s1 = m.addVars(competition_pairs, vtype=GRB.BINARY, name="s1")
        
        # s2[a, a']: 1 if a is assigned before a' ends (Overlapping logic)
        s2 = m.addVars(competition_pairs, vtype=GRB.BINARY, name="s2")

        # q[j, a, a']: 1 if a' is ongoing when a starts on resource j
        # 只針對共享資源 j 的 pair 建立 q
        q_indices = []
        for (a_id, a_prime_id) in competition_pairs:
            common_res = set(self.act_valid_resources[a_id]) & set(self.act_valid_resources[a_prime_id])
            for r_id in common_res:
                q_indices.append((r_id, a_id, a_prime_id))
        
        q = m.addVars(q_indices, vtype=GRB.BINARY, name="q")

        # ==========================
        # Stage 2 Variables (Scenario Dependent)
        # ==========================
        
        # b[a, n]: Start time of activity a in scenario n
        b = m.addVars([a['id'] for a in acts], scenarios, vtype=GRB.CONTINUOUS, lb=0, name="b")

        m.update()

        # ==========================
        # Constraints
        # ==========================

        # --- (1b) Resource Assignment ---
        for a in acts:
            m.addConstr(gp.quicksum(x[a['id'], j] for j in self.act_valid_resources[a['id']]) == 1, 
                        name=f"assign_{a['id']}")

        # --- (1c) q Logic Definition ---
        # q >= s1 + s2 + x_a + x_a' - 3
        for (r_id, a_id, a_prime_id) in q_indices:
            m.addConstr(q[r_id, a_id, a_prime_id] >= 
                        s1[a_id, a_prime_id] + s2[a_id, a_prime_id] + 
                        x[a_id, r_id] + x[a_prime_id, r_id] - 3,
                        name=f"q_def_{r_id}_{a_id}_{a_prime_id}")

        # --- (1d) Resource Capacity ---
        # Sum of q[j, a, a'] <= Capacity - 1
        # Generator 中的 capacity 都是 1，所以 sum(q) <= 0 (即不能重疊)
        for a in acts:
            a_id = a['id']
            for r_id in self.act_valid_resources[a_id]:
                # 找出所有與 a 在資源 r 上競爭的 a'
                competitors = [pair[1] for pair in competition_pairs if pair[0] == a_id and r_id in self.act_valid_resources[pair[1]]]
                
                # 假設 capacity 固定為 1 (依據 generator)
                cap = 1
                if competitors:
                    # 使用 Big-M 放寬：如果 x[a,r]=0 (a沒用這個資源)，則約束無效
                    m.addConstr(gp.quicksum(q[r_id, a_id, a_prime] for a_prime in competitors) <= 
                                cap - 1 + self.M * (1 - x[a_id, r_id]),
                                name=f"capacity_{r_id}_{a_id}")

        # --- Stage 2: Scenario Specific Constraints ---
        total_wait_expr = 0

        for n in scenarios:
            for a in acts:
                a_id = a['id']
                
                # (2b) Initial Activity Start Time
                if a['is_start']:
                    scheduled_t = a['scheduled_start']
                    m.addConstr(b[a_id, n] >= scheduled_t, name=f"start_init_{a_id}_{n}")
                    total_wait_expr += (b[a_id, n] - scheduled_t)
                
                # (2c) Precedence Constraints (同個病人的流程)
                else:
                    pred_id = a['predecessor']
                    pred_act = next(item for item in acts if item["id"] == pred_id)
                    pred_dur = pred_act['durations'][n] # Realized duration
                    
                    m.addConstr(b[a_id, n] >= b[pred_id, n] + pred_dur, name=f"prec_{a_id}_{n}")
                    total_wait_expr += (b[a_id, n] - b[pred_id, n] - pred_dur)

            # (2d) - (2g) Sequencing Constraints (linking s1, s2 with b)
            for (a_id, a_prime_id) in competition_pairs:
                a_prime_act = next(item for item in acts if item["id"] == a_prime_id)
                d_prime_n = a_prime_act['durations'][n]

                # (2d) M * s1 >= b_a - b_a' + 1
                m.addConstr(self.M * s1[a_id, a_prime_id] >= b[a_id, n] - b[a_prime_id, n] + 0.001)
                
                # (2e) M(1 - s1) >= b_a' - b_a
                m.addConstr(self.M * (1 - s1[a_id, a_prime_id]) >= b[a_prime_id, n] - b[a_id, n])

                # (2f) M * s2 >= b_a' - b_a + d_a'
                m.addConstr(self.M * s2[a_id, a_prime_id] >= b[a_prime_id, n] - b[a_id, n] + d_prime_n)

                # (2g) M(1 - s2) >= b_a - b_a' - d_a' + 1
                m.addConstr(self.M * (1 - s2[a_id, a_prime_id]) >= b[a_id, n] - b[a_prime_id, n] - d_prime_n + 0.001)

        # ==========================
        # Objective (3a)
        # ==========================
        # Minimize Expected Total Waiting Time (Sample Average)
        m.setObjective((1.0 / self.num_scenarios) * total_wait_expr, GRB.MINIMIZE)
        # m.setParam('MIPFocus', 1)  # 1 代表專注於尋找可行解 (Feasible Solutions)
        if self.baseline_sol:
            for k, v in self.baseline_sol['x'].items():
                if k in x: x[k].Start = v
            for k, v in self.baseline_sol['s1'].items():
                if k in s1: s1[k].Start = v
            for k, v in self.baseline_sol['s2'].items():
                if k in s2: s2[k].Start = v
        m.optimize()

        if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
            # 提取第一階段決策 (Planning Decisions)
            sol_x = {(k[0], k[1]): v.X for k, v in x.items()}
            sol_s1 = {k: v.X for k, v in s1.items()}
            sol_s2 = {k: v.X for k, v in s2.items()}
            
            return {
                "status": m.status,
                "obj_val": m.objVal,
                "x": sol_x,
                "s1": sol_s1,
                "s2": sol_s2
            }
        else:
            print("Optimization failed.")
            return None