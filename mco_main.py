from mco_optimizer import MCOOptimizer
from instance_generator import InstanceGenerator
from baseline_model import BaselineSolver
from evaluate import solve_second_stage_evaluation, plot_waiting_time_comparison
import numpy as np

def main():
    # ==========================================
    # Phase 1: Run MCO to find optimal N
    # ==========================================
    # 設定參數：根據之前的經驗，設 Patients=15 比較容易跑出結果
    mco = MCOOptimizer(
        num_patients=15, 
        arrival_interval=10, 
        N0=30,          # 從 N=20 開始
        K=3,            # 為了省時間，K設3 (論文是多一點)
        N_prime=100,    # 驗證集大小
        epsilon=0.05,   # 5% Gap 就停止
        time_limit=600  # 每次優化給 10 分鐘
    )
    
    mco_result = mco.run()
    
    if not mco_result:
        print("MCO Procedure failed.")
        return

    optimal_N = mco_result['optimal_N']
    print(f"\n[Result] Optimal Sample Size determined by MCO: N = {optimal_N}")
    
    # ==========================================
    # Phase 2: Final Evaluation (SMILP vs Baseline)
    # ==========================================
    print(f"\nRunning Final Comparison with N={optimal_N}...")
    
    # 1. 產生最終測試數據
    # 使用 MCO 建議的 optimal_N 作為訓練大小
    # 測試集 N_TEST 可以大一點 (例如 200)
    N_TEST = 200
    gen = InstanceGenerator(num_patients=15, arrival_interval=10)
    train_data = gen.generate_data(num_scenarios=optimal_N)
    
    # 2. 解 Baseline
    print("Solving Baseline...")
    base_solver = BaselineSolver(train_data, time_limit=300)
    res_base = base_solver.solve()
    
    # 3. SMILP (直接使用 MCO 跑出來的最後一組解，或者重新訓練)
    # 為了公平，我們重新訓練一次 SMILP，並使用 Baseline 做 Warm Start
    print(f"Solving SMILP (N={optimal_N})...")
    from smilp_model import SMILPSolver
    smilp_solver = SMILPSolver(train_data, time_limit=1800, baseline_sol=res_base)
    smilp_solver.model.setParam('MIPFocus', 1)
    res_smilp = smilp_solver.build_and_solve()
    
    # 4. Out-of-Sample Test
    print("Generating Test Data...")
    test_data = train_data.copy()
    test_data['num_scenarios'] = N_TEST
    for a in test_data['activities']:
        mu, sigma = gen._convert_params_to_lognormal(a['mean_duration'], a['var_duration'])
        # 這裡你可以選擇是否放大變異數做壓力測試
        # sigma = sigma * 1.5 
        new_samples = np.maximum(1, np.round(np.random.lognormal(mu, sigma, N_TEST)))
        a['durations'] = new_samples.tolist()
        
    print("Evaluating...")
    smilp_waits = []
    base_waits = []
    
    for n in range(N_TEST):
        val_s, _ = solve_second_stage_evaluation(res_smilp, test_data, n)
        val_b, _ = solve_second_stage_evaluation(res_base, test_data, n)
        smilp_waits.append(val_s)
        base_waits.append(val_b)
        
    # 畫圖
    plot_waiting_time_comparison(smilp_waits, base_waits)
    
    avg_s = np.mean(smilp_waits)
    avg_b = np.mean(base_waits)
    
    print("\n" + "="*40)
    print(f"FINAL COMPARISON (N={optimal_N})")
    print("="*40)
    print(f"SMILP Avg Wait:    {avg_s:.2f}")
    print(f"Baseline Avg Wait: {avg_b:.2f}")
    print(f"VSS:               {avg_b - avg_s:.2f}")
    print(f"Improvement:       {((avg_b - avg_s)/avg_b)*100:.2f}%")

if __name__ == "__main__":
    main()