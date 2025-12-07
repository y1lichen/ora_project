import copy
from smilp_model import SMILPSolver

class BaselineSolver:
    def __init__(self, data, time_limit=3600):
        self.original_data = data
        self.time_limit = time_limit
        
    def solve(self):
        # 1. 創建數據副本，將場景數設為 1
        mean_data = copy.deepcopy(self.original_data)
        mean_data['num_scenarios'] = 1
        
        # 2. 將所有 Activity 的 duration 替換為 Mean Duration
        for act in mean_data['activities']:
            # 注意：這裡將 mean_duration 放入列表，模擬單一場景
            act['durations'] = [act['mean_duration']]
            
        # 3. 使用 SMILPSolver 求解 deterministic 版本
        print("--- Solving Baseline (Deterministic Mean Value) ---")
        solver = SMILPSolver(mean_data, time_limit=self.time_limit)
        result = solver.build_and_solve()
        
        return result