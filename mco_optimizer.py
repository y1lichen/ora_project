import numpy as np
import copy
import time
from instance_generator import InstanceGenerator
from smilp_model import SMILPSolver
from evaluate import solve_second_stage

class MCOOptimizer:
    def __init__(self, num_patients=10, arrival_interval=10, 
                 initial_n=20, sim_n_prime=100, step_size=20, 
                 replicates_k=5, epsilon=0.05, max_n=200):
        """
        :param initial_n: 初始優化樣本數 N0
        :param sim_n_prime: 模擬(驗證)樣本數 N' (通常遠大於 N)
        :param replicates_k: 每個 N 重複跑 K 次 (K replicates) 來計算統計區間
        :param epsilon: AOI 停止條件閾值
        """
        self.generator = InstanceGenerator(num_patients=num_patients, arrival_interval=arrival_interval)
        
        self.current_n = initial_n
        self.n_prime = sim_n_prime
        self.step_size = step_size
        self.K = replicates_k
        self.epsilon = epsilon
        self.max_n = max_n

    def generate_scenarios(self, n_samples):
        """生成指定數量的場景數據"""
        # 注意: 這裡我們需要一個全新的數據結構，與 generator 保持一致
        # 為了避免重新生成病人路徑(pathway)，我們需要先生成一次基礎結構
        # 然後只變動 duration
        
        # 這裡簡化處理：我們假設每次呼叫 generate_data 都會產生同樣的病人結構
        # 只要 random_seed 控制得當，或者我們修改 generator 讓它分開生成結構和時間
        # *在目前的 generator 實作中，每次 generate_data 都會重骰 pathway*
        # *這在 MCO 中是不對的，MCO 是針對「特定一組病人需求」解決問題*
        
        # 修正策略：先生成一次 Master Data (包含病人、路徑)，只把 durations 設為空或模板
        if not hasattr(self, 'master_data'):
            self.master_data = self.generator.generate_data(num_scenarios=1)
        
        # 複製 master data
        data = copy.deepcopy(self.master_data)
        data['num_scenarios'] = n_samples
        
        # 為每個 Activity 重新採樣 N 個 duration
        for act in data['activities']:
            mean = act['mean_duration']
            var = act['var_duration']
            mu, sigma = self.generator._convert_params_to_lognormal(mean, var)
            # 重新抽樣
            samples = np.maximum(1, np.round(np.random.lognormal(mu, sigma, n_samples)))
            act['durations'] = samples.tolist()
            
        return data

    def run(self):
        print(f"Starting MCO Procedure (Target Epsilon={self.epsilon})...")
        
        while self.current_n <= self.max_n:
            print(f"\n--- Iteration with N = {self.current_n} ---")
            
            lower_bounds = []  # v_N^k (Optimization objective)
            upper_bounds = []  # v_N'^k (Simulation objective)
            
            start_time = time.time()
            
            # 執行 K 次重複實驗 (Replicates)
            for k in range(self.K):
                # 1. Generate Optimization Samples (Size N)
                opt_data = self.generate_scenarios(self.current_n)
                
                # 2. Solve Optimization Problem (SAA)
                # 這裡時間限制設短一點，因為是迭代過程
                solver = SMILPSolver(opt_data, time_limit=120, mip_gap=0.02)
                sol = solver.build_and_solve()
                
                if not sol:
                    print(f"  Replicate {k+1}: Optimization failed.")
                    continue
                
                v_opt = sol['obj_val']
                lower_bounds.append(v_opt)
                
                # 3. Generate Simulation Samples (Size N')
                # 這是用來評估該解在真實情況(更大樣本)下的表現
                sim_data = self.generate_scenarios(self.n_prime)
                
                # 4. Evaluate (Simulation Step)
                # 固定第一階段解 sol['x'], sol['s1']...，計算 N' 個場景的平均等待時間
                sim_vals = []
                for n_idx in range(self.n_prime):
                    val, _ = solve_second_stage(sol, sim_data, n_idx)
                    sim_vals.append(val)
                
                v_sim = np.mean(sim_vals)
                upper_bounds.append(v_sim)
                
                print(f"  Replicate {k+1}: Opt_Obj={v_opt:.2f}, Sim_Obj={v_sim:.2f}")

            # 計算統計界限
            avg_lower = np.mean(lower_bounds)
            avg_upper = np.mean(upper_bounds)
            
            # 計算 AOI (Approximate Optimality Index)
            # AOI = |Upper - Lower| / Upper
            if avg_upper == 0:
                aoi = 0
            else:
                aoi = abs(avg_upper - avg_lower) / avg_upper
            
            elapsed = time.time() - start_time
            print(f"Summary N={self.current_n}: Avg Lower={avg_lower:.2f}, Avg Upper={avg_upper:.2f}")
            print(f"AOI = {aoi:.4f} (Time: {elapsed:.2f}s)")
            
            # 檢查停止條件
            if aoi <= self.epsilon:
                print(f"\nConvergence reached! Optimal Sample Size N = {self.current_n}")
                return {
                    "optimal_N": self.current_n,
                    "final_aoi": aoi,
                    "lower_bound": avg_lower,
                    "upper_bound": avg_upper
                }
            
            # 增加樣本數
            self.current_n += self.step_size

        print("\nMax sample size reached without convergence.")
        return None

if __name__ == "__main__":
    # 測試 MCO
    mco = MCOOptimizer(
        num_patients=20,       # 為了測試速度，病人少一點
        arrival_interval=15, 
        initial_n=10, 
        sim_n_prime=50,       # 模擬樣本數
        step_size=10, 
        replicates_k=3,       # 重複次數
        epsilon=0.05          # 5% gap
    )
    mco.run()