import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random

def solve_smilp_mco(data, N0=5, N_prime=20, K=3, epsilon=0.05,
                         time_limit=300, gap=0.05, bigM=10000, random_seed=42):
    """
    Implements the complete SMILP + SAA + MCO algorithm from the paper.
    
    Paper formulation:
    - First stage: Minimize E_ξ[Q(x, s1, s2, q, ξ)] (Eq. 1a)
    - Constraints: (1b) Resource assignment, (1c) Indicator logic, (1d) Capacity
    - Second stage: Minimize waiting time for each scenario (Eq. 2a)
    - Constraints: (2b-2g) Precedence and non-overlap with big-M
    - MCO procedure: Determines sample size N with AOI convergence check
    
    Parameters:
    - N0: Initial SAA sample size (smaller than N_prime)
    - N_prime: Simulation sample size for validation
    - K: Number of replicates in MCO
    - epsilon: Convergence tolerance for AOI
    - time_limit, gap: Gurobi parameters
    - bigM: Big-M parameter for constraints (2d-2g)
    - random_seed: Reproducibility
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    activities = data['activities']
    resources = data['resources']
    num_acts = len(activities)

    # Expand resource instances: J_g = {1, 2, ..., k_j}
    resource_instances = {}
    inst_to_type = {}
    resource_capacities = {}
    global_inst_id = 0
    
    for r_name, r_info in resources.items():
        type_id = r_info['id']
        cap = r_info['capacity']
        resource_instances[type_id] = []
        resource_capacities[type_id] = cap
        
        for k in range(cap):
            resource_instances[type_id].append(global_inst_id)
            inst_to_type[global_inst_id] = type_id
            global_inst_id += 1

    # Map activities to resources: A(g) = {activities requiring resource g}
    Va_g = [{} for _ in range(num_acts)]
    for a in activities:
        a_id = a['id']
        for g in a['required_resources']:
            Va_g[a_id][g] = Va_g[a_id].get(g, 0) + 1

    # Helper: Generate duration samples
    def generate_duration_samples(num_samples):
        """Generate num_samples duration realizations for all activities"""
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

    # Helper: Build SMILP + SAA model with scenario-dependent constraints
    def build_saa_model(durations_dict, num_scenarios, trial_id):
        """
        Build SAA formulation (Eq. 3a-3c) combining:
        - First-stage constraints (1b-1d): shared across all scenarios
        - Second-stage constraints (2b-2g): replicated for each scenario n=1..N
        """
        m = gp.Model(f"SMILP_SAA_{trial_id}")
        x, s1, s2, q = {}, {}, {}, {}
        b = {}

        # ========== FIRST STAGE DECISIONS ==========
        
        # Constraint (1b): Va,g = sum_{j in Jg} xa_j
        # Resources: for each activity a and resource type g it needs,
        # assign it to exactly Va,g instances of that resource
        for a in activities:
            a_id = a['id']
            for g, req_cnt in Va_g[a_id].items():
                insts = resource_instances.get(g, [])
                for inst in insts:
                    x[(a_id, inst)] = m.addVar(vtype=GRB.BINARY, name=f"x_{a_id}_{inst}")
                m.addConstr(gp.quicksum(x[(a_id, inst)] for inst in insts) == req_cnt,
                            name=f"constr_1b_a{a_id}_g{g}")

        # Build activity pairs that share resources (for constraints 1c, 1d, 2d-2g)
        conflict_pairs = []
        act_types = [set(Va_g[a['id']].keys()) for a in activities]
        for i in range(num_acts):
            for j in range(i + 1, num_acts):
                shared = act_types[i].intersection(act_types[j])
                if shared:
                    conflict_pairs.append((i, j, shared))

        # Constraints (1c) & (1d): Sequencing and capacity
        # For each resource instance j of type g:
        for g, insts in resource_instances.items():
            acts_with_g = [a['id'] for a in activities if g in Va_g[a['id']]]
            k_j = resource_capacities[g]  # Capacity of resource type g
            
            for inst in insts:
                for i_idx in range(len(acts_with_g)):
                    for j_idx in range(i_idx + 1, len(acts_with_g)):
                        a, a_prime = acts_with_g[i_idx], acts_with_g[j_idx]
                        key = (a, a_prime)
                        
                        # Create s1, s2 variables (only once per activity pair)
                        if key not in s1:
                            s1[key] = m.addVar(vtype=GRB.BINARY, name=f"s1_{a}_{a_prime}")
                            s2[key] = m.addVar(vtype=GRB.BINARY, name=f"s2_{a}_{a_prime}")
                            m.addConstr(s1[key] + s2[key] <= 1, name=f"s_exclusive_{a}_{a_prime}")
                        
                        # Constraint (1c): q_j_{a,a'} >= s1 + s2 + xa_j + xa'_j - 3
                        q[(g, inst, a, a_prime)] = m.addVar(vtype=GRB.BINARY, 
                                                             name=f"q_{g}_{inst}_{a}_{a_prime}")
                        m.addConstr(q[(g, inst, a, a_prime)] >= s1[key] + s2[key] + 
                                   x[(a, inst)] + x[(a_prime, inst)] - 3,
                                   name=f"constr_1c_g{g}_j{inst}_{a}_{a_prime}")
        
        # Constraint (1d): sum_{a' in A(g), a'!=a} q_j_{a,a'} <= k_j - 1
        for g, insts in resource_instances.items():
            acts_with_g = [a['id'] for a in activities if g in Va_g[a['id']]]
            k_j = resource_capacities[g]
            
            for inst in insts:
                for a in acts_with_g:
                    # Sum over all pairs involving activity a (in either direction)
                    q_sum = gp.LinExpr()
                    for a_prime in acts_with_g:
                        if a_prime == a:
                            continue
                        # Get the q variable (created with smaller index first)
                        key_pair = (min(a, a_prime), max(a, a_prime))
                        if (g, inst, key_pair[0], key_pair[1]) in q:
                            q_sum += q[(g, inst, key_pair[0], key_pair[1])]
                    
                    m.addConstr(q_sum <= k_j - 1,
                               name=f"constr_1d_g{g}_j{inst}_a{a}")

        # ========== SECOND STAGE DECISIONS (per scenario) ==========
        # Replicate constraints (2b-2g) for each scenario n
        
        for scenario_n in range(num_scenarios):
            durations = durations_dict[scenario_n]
            
            # Create start time variables for this scenario
            for a in activities:
                a_id = a['id']
                b[(a_id, scenario_n)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, 
                                                  name=f"b_{a_id}_{scenario_n}")
            
            # Constraints (2b-2c): Precedence constraints
            for a in activities:
                a_id = a['id']
                if a['is_start']:
                    # (2b): ba >= ta for a in A0 (start activities)
                    m.addConstr(b[(a_id, scenario_n)] >= a['scheduled_start'],
                               name=f"constr_2b_a{a_id}_n{scenario_n}")
                else:
                    # (2c): ba >= b_pre(a) + d_pre(a) for a in A1
                    prev_id = a['predecessor']
                    prev_dur = durations[prev_id]
                    m.addConstr(b[(a_id, scenario_n)] >= b[(prev_id, scenario_n)] + prev_dur,
                               name=f"constr_2c_a{a_id}_n{scenario_n}")
            
            # Constraints (2d-2g): Non-overlap with big-M
            for (a, a_prime, shared_res) in conflict_pairs:
                dur_a = durations[a]
                dur_a_prime = durations[a_prime]
                
                # (2d): M*s1_{a,a'} >= ba - ba' + 1
                m.addConstr(bigM * s1[(a, a_prime)] >= b[(a, scenario_n)] - b[(a_prime, scenario_n)] + 1,
                           name=f"constr_2d_a{a}_ap{a_prime}_n{scenario_n}")
                
                # (2e): M*(1 - s1_{a,a'}) >= ba' - ba
                m.addConstr(bigM * (1 - s1[(a, a_prime)]) >= b[(a_prime, scenario_n)] - b[(a, scenario_n)],
                           name=f"constr_2e_a{a}_ap{a_prime}_n{scenario_n}")
                
                # (2f): M*s2_{a,a'} >= ba' - ba + da'
                m.addConstr(bigM * s2[(a, a_prime)] >= b[(a_prime, scenario_n)] - b[(a, scenario_n)] + dur_a_prime,
                           name=f"constr_2f_a{a}_ap{a_prime}_n{scenario_n}")
                
                # (2g): M*(1 - s2_{a,a'}) >= ba - ba' - da' + 1
                m.addConstr(bigM * (1 - s2[(a, a_prime)]) >= b[(a, scenario_n)] - b[(a_prime, scenario_n)] - dur_a_prime + 1,
                           name=f"constr_2g_a{a}_ap{a_prime}_n{scenario_n}")

        # ========== OBJECTIVE FUNCTION (3a) ==========
        # min (1/N) * sum_n [ sum_{a in A0} (b_n_a - ta) + sum_{a in A1} (b_n_a - b_n_pre(a) - d_n_pre(a)) ]
        
        total_wait = gp.LinExpr()
        for scenario_n in range(num_scenarios):
            durations = durations_dict[scenario_n]
            for a in activities:
                a_id = a['id']
                if a['is_start']:
                    # A0 activities: waiting = ba - ta
                    total_wait += b[(a_id, scenario_n)] - a['scheduled_start']
                else:
                    # A1 activities: waiting = ba - (b_pre(a) + d_pre(a))
                    prev_id = a['predecessor']
                    prev_dur = durations[prev_id]
                    total_wait += b[(a_id, scenario_n)] - (b[(prev_id, scenario_n)] + prev_dur)
        
        avg_wait = total_wait / num_scenarios  # (3a): average over scenarios
        m.setObjective(avg_wait, GRB.MINIMIZE)

        return m, x, s1, s2, q, b

    # ========== MCO ALGORITHM (Algorithm 1 from paper) ==========
    print("\n[MCO Algorithm] Starting sample size optimization...")
    print(f"Initial sample size N0={N0}, Validation size N'={N_prime}, "
          f"Replicates K={K}, Convergence tolerance ε={epsilon}")

    N = N0
    aoi_history = []
    v_N_vals = []
    v_Np_vals = []

    for mco_iter in range(10):  # Maximum 10 iterations to find appropriate N
        print(f"\n[MCO Iteration {mco_iter + 1}] Current sample size N={N}")
        
        v_N_replicate = []  # Lower bounds (optimization stage)
        v_Np_replicate = []  # Upper bounds (simulation stage)

        for replicate_k in range(K):
            print(f"  Replicate {replicate_k + 1}/{K}...")
            
            # Generate N + N' samples total
            all_durations = generate_duration_samples(N + N_prime)
            
            # === OPTIMIZATION STAGE: Solve SAA with first N scenarios ===
            opt_durations = {k: all_durations[k] for k in range(N)}
            m_opt, x_opt, s1_opt, s2_opt, q_opt, b_opt = build_saa_model(opt_durations, N, 
                                                                           f"opt_iter{mco_iter}_rep{replicate_k}")
            
            m_opt.setParam('OutputFlag', 0)
            m_opt.setParam('TimeLimit', time_limit)
            m_opt.setParam('MIPGap', gap)
            m_opt.optimize()
            
            if m_opt.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
                print(f"    Optimization stage failed. Status: {m_opt.Status}")
                continue
            
            v_k_N = m_opt.ObjVal  # Lower bound
            v_N_replicate.append(v_k_N)
            
            # Extract first-stage solution
            x_solution = {}
            for (a_id, inst), var in x_opt.items():
                if var.X > 0.5:
                    if a_id not in x_solution:
                        x_solution[a_id] = []
                    x_solution[a_id].append(inst)
            
            s1_solution = {}
            for key, var in s1_opt.items():
                if var.X > 0.5:
                    s1_solution[key] = True
            
            # === SIMULATION STAGE: Validate with N' new scenarios, fixed first-stage ===
            sim_durations = {k: all_durations[N + k] for k in range(N_prime)}
            
            # Build validation model with fixed x, s1, s2
            m_sim, x_sim, s1_sim, s2_sim, q_sim, b_sim = build_saa_model(sim_durations, N_prime, 
                                                        f"sim_iter{mco_iter}_rep{replicate_k}")
            
            # Fix first-stage variables in m_sim to optimization solution
            for (a_id, inst), var in x_sim.items():
                if a_id in x_solution and inst in x_solution[a_id]:
                    m_sim.addConstr(var == 1)
                else:
                    m_sim.addConstr(var == 0)
            
            for key, var in s1_sim.items():
                if key in s1_solution:
                    m_sim.addConstr(var == 1)
                else:
                    m_sim.addConstr(var == 0)
            
            m_sim.setParam('OutputFlag', 0)
            m_sim.setParam('TimeLimit', time_limit)
            m_sim.setParam('MIPGap', gap)
            m_sim.optimize()
            
            if m_sim.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
                print(f"    Simulation stage failed. Status: {m_sim.Status}")
                continue
            
            v_k_Np = m_sim.ObjVal  # Upper bound
            v_Np_replicate.append(v_k_Np)
        
        if not v_N_replicate or not v_Np_replicate:
            print("  Warning: Some replicates failed. Retrying with larger N...")
            N += N0
            continue
        
        # Compute average bounds and AOI
        v_N_bar = np.mean(v_N_replicate)  # Average lower bound
        v_Np_bar = np.mean(v_Np_replicate)  # Average upper bound
        
        v_N_vals.append(v_N_bar)
        v_Np_vals.append(v_Np_bar)
        
        # AOI = (|v_N' - v_N|) / v_N'
        aoi = abs(v_Np_bar - v_N_bar) / v_Np_bar if v_Np_bar != 0 else 0
        aoi_history.append(aoi)
        
        print(f"  v_N (lower bound): {v_N_bar:.4f}")
        print(f"  v_N' (upper bound): {v_Np_bar:.4f}")
        print(f"  AOI: {aoi:.6f}")
        
        # Check convergence
        if aoi <= epsilon:
            print(f"\n[MCO Convergence] AOI={aoi:.6f} <= ε={epsilon}")
            print(f"Optimal sample size: N={N}")
            optimal_N = N
            final_v_N_bar = v_N_bar
            final_v_Np_bar = v_Np_bar
            break
        
        N += N0

    else:
        print(f"\n[MCO Warning] Did not converge within 10 iterations.")
        optimal_N = N
        final_v_N_bar = v_N_vals[-1] if v_N_vals else 0
        final_v_Np_bar = v_Np_vals[-1] if v_Np_vals else 0

    # ========== FINAL SOLVE with optimal N ==========
    print(f"\n[Final Solve] Solving SAA with optimal sample size N={optimal_N}...")
    final_durations = generate_duration_samples(optimal_N)
    m_final, x_final, s1_final, s2_final, q_final, b_final = build_saa_model(final_durations, 
                                                                               optimal_N, "final")
    
    m_final.setParam('OutputFlag', 0)
    m_final.setParam('TimeLimit', time_limit)
    m_final.setParam('MIPGap', gap)
    m_final.optimize()

    # Package results
    best_assignment = {k: [] for k in range(num_acts)}
    if m_final.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
        for (a_id, inst), var in x_final.items():
            if var.X > 0.5:
                best_assignment[a_id].append(inst)
    
    best_order = {}
    for key, var in s1_final.items():
        if var.X > 0.5:
            best_order[key] = True

    class SMILPResult:
        def __init__(self, assignment, order, status, opt_N, v_N_bar, v_Np_bar, aoi_hist):
            self.assignment = assignment
            self.order = order
            self.Status = status
            self.optimal_sample_size = opt_N
            self.lower_bound = v_N_bar
            self.upper_bound = v_Np_bar
            self.aoi_history = aoi_hist
            self.final_objective = m_final.ObjVal if m_final.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT] else None

    return SMILPResult(best_assignment, best_order, m_final.Status, optimal_N, 
                       final_v_N_bar, final_v_Np_bar, aoi_history)
