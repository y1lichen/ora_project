import numpy as np
import time
import copy
from instance_generator import InstanceGenerator
from smilp_model import SMILPSolver
from evaluate import solve_second_stage_evaluation  # 重用 evaluate 中的函數

class MCOOptimizer:
    def __init__(self, num_patients=15, arrival_interval=10, 
                 N0=20, K=5, N_prime=100, epsilon=0.05, 
                 time_limit=600):
        """
        復現論文 Algorithm 1: MCO Method
        
        :param num_patients: 病人數量
        :param N0: 初始樣本數 (Initial sample size)
        :param K: 重複次數 (Number of replicates)
        :param N_prime: 模擬樣本數 (Simulation sample size N')
        :param epsilon: 停止閾值 (Termination tolerance)
        :param time_limit: Gurobi 求解時間限制
        """
        self.generator = InstanceGenerator(num_patients=num_patients, arrival_interval=arrival_interval)
        self.N = N0
        self.K = K
        self.N_prime = N_prime
        self.epsilon = epsilon
        self.time_limit = time_limit
        
        # 記錄歷程
        self.history = []

    def run(self):
        print(f"Starting MCO Procedure (Patients={self.generator.num_patients}, Epsilon={self.epsilon})...")
        
        AOI = float('inf')
        
        while AOI >= self.epsilon:
            print(f"\n" + "="*50)
            print(f"Iteration with Sample Size N = {self.N}")
            print("="*50)
            
            v_N_list = []      # 存儲 K 次優化的目標值 (Lower Bound)
            v_N_prime_list = [] # 存儲 K 次模擬的目標值 (Upper Bound)
            
            best_sol_in_this_N = None # 暫存這個 N 下最好的解
            
            for k in range(self.K):
                print(f"  Replicate {k+1}/{self.K}...", end=" ", flush=True)
                
                # -------------------------------------------------
                # Step 1: Generate (N + N') Scenarios
                # -------------------------------------------------
                # 生成總數據，包含 N 個訓練用 + N' 個驗證用
                total_samples = self.N + self.N_prime
                data = self.generator.generate_data(num_scenarios=total_samples)
                
                # 切分數據：前 N 個給優化，後 N' 個給模擬
                train_data = copy.deepcopy(data)
                train_data['num_scenarios'] = self.N
                for act in train_data['activities']:
                    act['durations'] = act['durations'][:self.N]
                
                sim_data = copy.deepcopy(data)
                # 模擬數據只保留後 N' 個 duration
                sim_data['num_scenarios'] = self.N_prime
                for act in sim_data['activities']:
                    act['durations'] = act['durations'][self.N:]
                
                # -------------------------------------------------
                # Step 2: Solve SAA (Optimization) -> v_N
                # -------------------------------------------------
                # 這裡可以使用 Warm Start (如果你有 Baseline 解的話，可以傳入)
                # 為了簡化，這裡先裸跑，但建議加上 MIPFocus=1
                solver = SMILPSolver(train_data, time_limit=self.time_limit)
                solver.model.setParam('MIPFocus', 1) 
                solver.model.setParam('OutputFlag', 0) # 關閉 Gurobi 刷屏
                
                res = solver.build_and_solve()
                
                if not res:
                    print("[Failed] Optimization timed out or infeasible.")
                    continue
                
                v_N_k = res['obj_val']
                v_N_list.append(v_N_k)
                
                # -------------------------------------------------
                # Step 3: Evaluate (Simulation) -> v_N'
                # -------------------------------------------------
                # 用剛算出來的 x, s1, s2 去跑 N' 個模擬場景
                sim_wait_sum = 0
                for n in range(self.N_prime):
                    val, _ = solve_second_stage_evaluation(res, sim_data, n)
                    sim_wait_sum += val
                
                v_N_prime_k = sim_wait_sum / self.N_prime
                v_N_prime_list.append(v_N_prime_k)
                
                print(f"Opt(v_N)={v_N_k:.2f}, Sim(v_N')={v_N_prime_k:.2f}")
                
                # 保存最後一輪的第一個解當作 output
                if k == self.K - 1:
                    best_sol_in_this_N = res

            # -------------------------------------------------
            # Step 4: Calculate Statistics
            # -------------------------------------------------
            if not v_N_list:
                print("All replicates failed. Stopping.")
                return None

            avg_v_N = np.mean(v_N_list)       # Lower Bound
            avg_v_N_prime = np.mean(v_N_prime_list) # Upper Bound
            
            # AOI Calculation (論文公式)
            # AOI = | v_N' - v_N | / v_N'
            if avg_v_N_prime > 0:
                AOI = abs(avg_v_N_prime - avg_v_N) / avg_v_N_prime
            else:
                AOI = float('inf')
                
            print(f"\nStats for N={self.N}:")
            print(f"  Avg Optimization Obj (Lower Bound): {avg_v_N:.2f}")
            print(f"  Avg Simulation Obj   (Upper Bound): {avg_v_N_prime:.2f}")
            print(f"  Gap (Overfitting measure):          {avg_v_N_prime - avg_v_N:.2f}")
            print(f"  AOI:                                {AOI:.4f} (Target: {self.epsilon})")
            
            self.history.append({
                'N': self.N,
                'AOI': AOI,
                'v_N': avg_v_N,
                'v_N_prime': avg_v_N_prime
            })
            
            if AOI < self.epsilon:
                print(f"\n>>> CONVERGED! Optimal Sample Size found: N = {self.N}")
                return {
                    "optimal_N": self.N,
                    "final_solution": best_sol_in_this_N,
                    "final_AOI": AOI,
                    "lower_bound": avg_v_N,
                    "upper_bound": avg_v_N_prime
                }
            
            # Update N: 論文說 N <- 2N
            self.N = self.N * 2
            
            # 安全閥：防止 N 爆炸導致算不動
            if self.N > 200:
                print("\n>>> Reached max sample size limit (200). Stopping.")
                return {
                    "optimal_N": self.N, # 雖然沒收斂，但回傳目前最大的
                    "final_solution": best_sol_in_this_N,
                    "final_AOI": AOI,
                    "lower_bound": avg_v_N,
                    "upper_bound": avg_v_N_prime
                }
        
        return None