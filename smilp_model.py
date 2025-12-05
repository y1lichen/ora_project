import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random

def solve_smilp_mco(data, N0=5, N_prime=20, K=3, epsilon=0.05,
                         time_limit=300, gap=0.05, bigM=10000, random_seed=42):
    """
    Implements SMILP + SAA + MCO.
    Returns the result object containing the model, solution, bounds, AND the validation scenarios used.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    activities = data['activities']
    resources = data['resources']
    num_acts = len(activities)

    # 1. Resource Setup
    resource_instances = {}
    resource_capacities = {}
    global_inst_id = 0
    
    for r_name, r_info in resources.items():
        type_id = r_info['id']
        cap = r_info['capacity']
        resource_instances[type_id] = []
        resource_capacities[type_id] = cap
        for k in range(cap):
            resource_instances[type_id].append(global_inst_id)
            global_inst_id += 1

    Va_g = [{} for _ in range(num_acts)]
    for a in activities:
        a_id = a['id']
        for g in a['required_resources']:
            Va_g[a_id][g] = Va_g[a_id].get(g, 0) + 1

    # Helper: Generate duration samples
    def generate_duration_samples(num_samples):
        durations = {}
        for k in range(num_samples):
            durations[k] = {}
            for a in activities:
                a_id = a['id']
                mu_val = a['mean_duration']
                sigma_val = mu_val * 0.3
                phi = np.sqrt(sigma_val**2 + mu_val**2)
                log_mu = np.log(mu_val**2 / phi)
                log_sigma = np.sqrt(np.log(phi**2 / mu_val**2))
                dur = np.random.lognormal(log_mu, log_sigma)
                durations[k][a_id] = max(1, round(dur))
        return durations

    # Helper: Build Model
    def build_saa_model(durations_dict, num_scenarios, trial_id):
        m = gp.Model(f"SMILP_SAA_{trial_id}")
        x, s1, s2, q = {}, {}, {}, {}
        b = {}

        # First Stage
        for a in activities:
            a_id = a['id']
            for g, req_cnt in Va_g[a_id].items():
                insts = resource_instances.get(g, [])
                for inst in insts:
                    x[(a_id, inst)] = m.addVar(vtype=GRB.BINARY, name=f"x_{a_id}_{inst}")
                m.addConstr(gp.quicksum(x[(a_id, inst)] for inst in insts) == req_cnt)

        conflict_pairs = []
        act_types = [set(Va_g[a['id']].keys()) for a in activities]
        for i in range(num_acts):
            for j in range(i + 1, num_acts):
                shared = act_types[i].intersection(act_types[j])
                if shared:
                    conflict_pairs.append((i, j, shared))

        for g, insts in resource_instances.items():
            acts_with_g = [a['id'] for a in activities if g in Va_g[a['id']]]
            k_j = resource_capacities[g]
            for inst in insts:
                for i_idx in range(len(acts_with_g)):
                    for j_idx in range(i_idx + 1, len(acts_with_g)):
                        a, a_prime = acts_with_g[i_idx], acts_with_g[j_idx]
                        key = (a, a_prime)
                        if key not in s1:
                            s1[key] = m.addVar(vtype=GRB.BINARY, name=f"s1_{a}_{a_prime}")
                            s2[key] = m.addVar(vtype=GRB.BINARY, name=f"s2_{a}_{a_prime}")
                            m.addConstr(s1[key] + s2[key] <= 1)
                        q[(g, inst, a, a_prime)] = m.addVar(vtype=GRB.BINARY)
                        m.addConstr(q[(g, inst, a, a_prime)] >= s1[key] + s2[key] + x[(a, inst)] + x[(a_prime, inst)] - 3)
        
        for g, insts in resource_instances.items():
            acts_with_g = [a['id'] for a in activities if g in Va_g[a['id']]]
            k_j = resource_capacities[g]
            for inst in insts:
                for a in acts_with_g:
                    q_sum = gp.LinExpr()
                    for a_prime in acts_with_g:
                        if a_prime == a: continue
                        key_pair = (min(a, a_prime), max(a, a_prime))
                        if (g, inst, key_pair[0], key_pair[1]) in q:
                            q_sum += q[(g, inst, key_pair[0], key_pair[1])]
                    m.addConstr(q_sum <= k_j - 1)

        # Second Stage
        for scenario_n in range(num_scenarios):
            durations = durations_dict[scenario_n]
            for a in activities:
                b[(a['id'], scenario_n)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"b_{a['id']}_{scenario_n}")
            
            for a in activities:
                a_id = a['id']
                if a['is_start']:
                    m.addConstr(b[(a_id, scenario_n)] >= a['scheduled_start'])
                else:
                    prev_id = a['predecessor']
                    prev_dur = durations[prev_id]
                    m.addConstr(b[(a_id, scenario_n)] >= b[(prev_id, scenario_n)] + prev_dur)
            
            for (a, a_prime, shared_res) in conflict_pairs:
                dur_a = durations[a]
                dur_a_prime = durations[a_prime]
                m.addConstr(bigM * s1[(a, a_prime)] >= b[(a, scenario_n)] - b[(a_prime, scenario_n)] + 1)
                m.addConstr(bigM * (1 - s1[(a, a_prime)]) >= b[(a_prime, scenario_n)] - b[(a, scenario_n)])
                m.addConstr(bigM * s2[(a, a_prime)] >= b[(a_prime, scenario_n)] - b[(a, scenario_n)] + dur_a_prime)
                m.addConstr(bigM * (1 - s2[(a, a_prime)]) >= b[(a, scenario_n)] - b[(a_prime, scenario_n)] - dur_a_prime + 1)

        total_wait = gp.LinExpr()
        for scenario_n in range(num_scenarios):
            durations = durations_dict[scenario_n]
            for a in activities:
                a_id = a['id']
                if a['is_start']:
                    total_wait += b[(a_id, scenario_n)] - a['scheduled_start']
                else:
                    prev_id = a['predecessor']
                    prev_dur = durations[prev_id]
                    total_wait += b[(a_id, scenario_n)] - (b[(prev_id, scenario_n)] + prev_dur)
        
        m.setObjective(total_wait / num_scenarios, GRB.MINIMIZE)
        return m, x, s1, s2, q, b

    # ========== MCO ALGORITHM ==========
    print(f"\n[MCO] Starting: N0={N0}, N'={N_prime}, K={K}, ε={epsilon}")
    N = N0
    aoi_history = []
    
    # Store the validation scenarios from the best iteration to pass to Baseline
    final_validation_scenarios = None 
    final_v_N_bar = 0
    final_v_Np_bar = 0
    optimal_N = N

    for mco_iter in range(10):
        print(f"[MCO Iter {mco_iter+1}] N={N}")
        v_N_reps = []
        v_Np_reps = []
        
        # We need to temporarily store validation scenarios for this iteration
        # In case this iteration converges, we use its LAST replicate's scenarios (or average them? 
        # Typically we just need A set of scenarios. Let's store the last replicate's set for simplicity)
        last_replicate_scenarios = None

        for k in range(K):
            all_durations = generate_duration_samples(N + N_prime)
            opt_durations = {i: all_durations[i] for i in range(N)}
            sim_durations = {i: all_durations[N+i] for i in range(N_prime)}
            
            # Optimization
            m_opt, x_opt, s1_opt, _, _, _ = build_saa_model(opt_durations, N, f"opt_{mco_iter}_{k}")
            m_opt.setParam('OutputFlag', 0)
            m_opt.setParam('TimeLimit', time_limit)
            m_opt.setParam('MIPGap', gap)
            m_opt.optimize()
            
            if m_opt.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]: continue
            v_N_reps.append(m_opt.ObjVal)
            
            # Extract Solution
            x_sol = {}
            for (aid, inst), v in x_opt.items():
                if v.X > 0.5: x_sol.setdefault(aid, []).append(inst)
            s1_sol = {key: True for key, v in s1_opt.items() if v.X > 0.5}

            # Validation
            m_sim, x_sim, s1_sim, _, _, _ = build_saa_model(sim_durations, N_prime, f"sim_{mco_iter}_{k}")
            for (aid, inst), v in x_sim.items():
                m_sim.addConstr(v == (1 if aid in x_sol and inst in x_sol[aid] else 0))
            for key, v in s1_sim.items():
                m_sim.addConstr(v == (1 if key in s1_sol else 0))
                
            m_sim.setParam('OutputFlag', 0)
            m_sim.setParam('TimeLimit', time_limit)
            m_sim.setParam('MIPGap', gap)
            m_sim.optimize()
            
            if m_sim.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
                v_Np_reps.append(m_sim.ObjVal)
                last_replicate_scenarios = sim_durations # Store this set

        if not v_N_reps or not v_Np_reps:
            N += N0
            continue

        v_N_bar = np.mean(v_N_reps)
        v_Np_bar = np.mean(v_Np_reps)
        aoi = abs(v_Np_bar - v_N_bar) / v_Np_bar if v_Np_bar != 0 else 0
        aoi_history.append(aoi)
        
        print(f"  AOI: {aoi:.6f} (LB:{v_N_bar:.2f}, UB:{v_Np_bar:.2f})")

        if aoi <= epsilon:
            print(f"  Converged! Optimal N={N}")
            optimal_N = N
            final_v_N_bar = v_N_bar
            final_v_Np_bar = v_Np_bar
            # Capture the validation scenarios from the simulation stage
            # IMPORTANT: The scenarios used to calculate v_Np_bar are aggregated.
            # To be strictly fair, we should output the scenarios from the LAST successful replicate 
            # and use those for Baseline, or generate a fresh set of N_prime for final comparison.
            # Let's use the last replicate's scenarios to be consistent.
            final_validation_scenarios = last_replicate_scenarios
            break
        N += N0
    
    if final_validation_scenarios is None:
        # If no convergence, use the last generated ones
        final_validation_scenarios = last_replicate_scenarios
        optimal_N = N
        if v_N_reps: final_v_N_bar = np.mean(v_N_reps)
        if v_Np_reps: final_v_Np_bar = np.mean(v_Np_reps)

    # Final Solve for Visualization (Single Model)
    print(f"[Final] Solving optimal model (N={optimal_N}) for exporting schedule...")
    final_durations = generate_duration_samples(optimal_N)
    m_final, x_f, s1_f, _, _, _ = build_saa_model(final_durations, optimal_N, "FINAL")
    m_final.setParam('OutputFlag', 0)
    m_final.setParam('TimeLimit', time_limit)
    m_final.optimize()

    class SMILPResult:
        def __init__(self, model, opt_N, v_N, v_Np, aoi, val_scenarios):
            self.final_model = model # Provide the model for extracting schedule
            self.optimal_sample_size = opt_N
            self.lower_bound = v_N
            self.upper_bound = v_Np
            self.aoi_history = aoi
            self.validation_scenarios = val_scenarios # Export scenarios

    return SMILPResult(m_final, optimal_N, final_v_N_bar, final_v_Np_bar, aoi_history, final_validation_scenarios)