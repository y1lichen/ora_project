import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve_deterministic_model(data, time_limit=60, bigM=10000, random_seed=42):
    """
    Baseline Mean Value Model (Deterministic MILP) from paper.
    
    Paper reference: "For model evaluation, we consider a baseline mean value
    model whose first-stage solution is derived by the deterministic MILP model 
    under the mean duration setting."
    
    This solves the deterministic counterpart of the SMILP using mean durations,
    producing first-stage decisions (x, s1, s2) that will be fixed in SAA evaluation.
    
    Parameters:
    - data: Instance data from InstanceGenerator
    - time_limit: Gurobi time limit
    - bigM: Big-M parameter for non-overlap constraints
    - random_seed: Reproducibility
    
    Returns:
    - m: Gurobi model with optimal first-stage decisions
    """
    np.random.seed(random_seed)
    
    activities = data['activities']
    resources = data['resources']
    num_acts = len(activities)
    
    m = gp.Model("Deterministic_MILP_Baseline")
    
    # ========== RESOURCE INSTANCE EXPANSION ==========
    resource_instances = {}
    resource_capacities = {}
    global_inst_id = 0
    
    for r_name, r_info in resources.items():
        r_type = r_info['id']
        cap = r_info['capacity']
        resource_instances[r_type] = []
        resource_capacities[r_type] = cap
        
        for k in range(cap):
            resource_instances[r_type].append(global_inst_id)
            global_inst_id += 1
    
    # ========== FIRST-STAGE VARIABLES ==========
    # x_{a,j}: binary assignment of activity a to resource instance j
    x = {}
    for a in activities:
        a_id = a['id']
        for g in a['required_resources']:
            insts = resource_instances.get(g, [])
            for inst in insts:
                x[(a_id, inst)] = m.addVar(vtype=GRB.BINARY, name=f"x_{a_id}_{inst}")
            # Constraint (1b): Va,g = sum_{j in Jg} xa_j (only one resource per activity in deterministic)
            m.addConstr(gp.quicksum(x[(a_id, inst)] for inst in insts) == 1,
                       name=f"constr_1b_det_a{a_id}_g{g}")
    
    # Find activity pairs that share resources
    conflict_pairs = []
    act_types = [set(a['required_resources']) for a in activities]
    for i in range(num_acts):
        for j in range(i + 1, num_acts):
            if act_types[i].intersection(act_types[j]):
                conflict_pairs.append((i, j))
    
    # s1_{a,a'}, s2_{a,a'}: sequencing variables (first-stage)
    s1, s2 = {}, {}
    for (i, j) in conflict_pairs:
        s1[(i, j)] = m.addVar(vtype=GRB.BINARY, name=f"s1_{i}_{j}")
        s2[(i, j)] = m.addVar(vtype=GRB.BINARY, name=f"s2_{i}_{j}")
        m.addConstr(s1[(i, j)] + s2[(i, j)] <= 1, name=f"s_excl_{i}_{j}")
    
    # ========== SECOND-STAGE VARIABLES (DETERMINISTIC) ==========
    # b_a: start time of activity a (using mean durations only)
    b = {}
    for a in activities:
        a_id = a['id']
        b[a_id] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"b_{a_id}")
    
    # ========== CONSTRAINTS ==========
    
    # Precedence constraints (2b-2c, using mean durations)
    for a in activities:
        a_id = a['id']
        if a['is_start']:
            # (2b): ba >= ta for start activities
            m.addConstr(b[a_id] >= a['scheduled_start'],
                       name=f"constr_2b_det_a{a_id}")
        else:
            # (2c): ba >= b_pre(a) + d_pre(a) [using mean]
            prev_id = a['predecessor']
            prev_dur = activities[prev_id]['mean_duration']
            m.addConstr(b[a_id] >= b[prev_id] + prev_dur,
                       name=f"constr_2c_det_a{a_id}")
    
    # Non-overlap constraints (2d-2g, using big-M with mean durations)
    for (i, j) in conflict_pairs:
        dur_i = activities[i]['mean_duration']  # Use mean duration
        dur_j = activities[j]['mean_duration']
        
        # (2d): M*s1_{i,j} >= b_i - b_j + 1
        m.addConstr(bigM * s1[(i, j)] >= b[i] - b[j] + 1,
                   name=f"constr_2d_det_{i}_{j}")
        
        # (2e): M*(1 - s1_{i,j}) >= b_j - b_i
        m.addConstr(bigM * (1 - s1[(i, j)]) >= b[j] - b[i],
                   name=f"constr_2e_det_{i}_{j}")
        
        # (2f): M*s2_{i,j} >= b_j - b_i + d_j
        m.addConstr(bigM * s2[(i, j)] >= b[j] - b[i] + dur_j,
                   name=f"constr_2f_det_{i}_{j}")
        
        # (2g): M*(1 - s2_{i,j}) >= b_i - b_j - d_j + 1
        m.addConstr(bigM * (1 - s2[(i, j)]) >= b[i] - b[j] - dur_j + 1,
                   name=f"constr_2g_det_{i}_{j}")
    
    # ========== OBJECTIVE FUNCTION ==========
    # Minimize total waiting time (using mean durations)
    total_wait = gp.LinExpr()
    for a in activities:
        a_id = a['id']
        if a['is_start']:
            # A0 activities: waiting = ba - ta
            total_wait += b[a_id] - a['scheduled_start']
        else:
            # A1 activities: waiting = ba - (b_pre(a) + d_pre(a))
            prev_id = a['predecessor']
            prev_dur = activities[prev_id]['mean_duration']
            total_wait += b[a_id] - (b[prev_id] + prev_dur)
    
    m.setObjective(total_wait, GRB.MINIMIZE)
    
    # ========== SOLVE ==========
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', time_limit)
    m.setParam('MIPGap', 0.001)  # Deterministic problem, tight gap
    
    m.optimize()
    
    return m


def evaluate_baseline_mean_value_model(baseline_model, data, K=5, 
                                       time_limit=300, gap=0.05, bigM=10000, 
                                       random_seed=42):
    """
    Paper evaluation procedure for baseline mean value model.
    
    Paper reference: "For the second stage, we run the SAA model in Formulation (3)
    for K iterations and obtain K objective values. In all K replicates, the 
    first-stage variables are fixed with the mean value model solution, while at 
    each iteration, we change the duration scenario with a different scenario from 
    the MCO module's last iteration. The average of baseline mean value solution 
    objective value is denoted as v̄base_N'."
    
    This evaluates the baseline model by:
    1. Extracting first-stage decisions from the deterministic model
    2. Running K SAA evaluations with fixed first-stage decisions
    3. Using K different duration scenarios
    4. Computing average objective value (v̄base_N')
    
    Parameters:
    - baseline_model: Solved deterministic MILP model
    - data: Instance data
    - K: Number of evaluation replicates
    - time_limit, gap, bigM: Gurobi parameters
    - random_seed: Reproducibility
    
    Returns:
    - avg_obj_value: Average objective value v̄base_N'
    - objectives: List of K objective values
    """
    np.random.seed(random_seed)
    
    if baseline_model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
        print("Baseline model not solved successfully.")
        return None, []
    
    activities = data['activities']
    resources = data['resources']
    num_acts = len(activities)
    
    # Extract first-stage solution from baseline model
    x_solution = {}
    for v in baseline_model.getVars():
        if v.VarName.startswith("x_") and v.X > 0.5:
            parts = v.VarName.split('_')
            a_id, inst_id = int(parts[1]), int(parts[2])
            x_solution[(a_id, inst_id)] = True
    
    s1_solution = {}
    for v in baseline_model.getVars():
        if v.VarName.startswith("s1_") and v.X > 0.5:
            parts = v.VarName.split('_')
            i, j = int(parts[1]), int(parts[2])
            s1_solution[(i, j)] = True
    
    s2_solution = {}
    for v in baseline_model.getVars():
        if v.VarName.startswith("s2_") and v.X > 0.5:
            parts = v.VarName.split('_')
            i, j = int(parts[1]), int(parts[2])
            s2_solution[(i, j)] = True
    
    # Expand resource instances
    resource_instances = {}
    resource_capacities = {}
    global_inst_id = 0
    
    for r_name, r_info in resources.items():
        r_type = r_info['id']
        cap = r_info['capacity']
        resource_instances[r_type] = []
        resource_capacities[r_type] = cap
        
        for k in range(cap):
            resource_instances[r_type].append(global_inst_id)
            global_inst_id += 1
    
    objectives = []
    
    for replicate_k in range(K):
        print(f"  Baseline evaluation replicate {replicate_k + 1}/{K}...")
        
        # Generate duration samples for this replicate
        durations_scenario = {}
        for a in activities:
            a_id = a['id']
            mu_val = a['mean_duration']
            sigma_val = mu_val * 0.3
            phi = np.sqrt(sigma_val**2 + mu_val**2)
            log_mu = np.log(mu_val**2 / phi)
            log_sigma = np.sqrt(np.log(phi**2 / mu_val**2))
            dur = np.random.lognormal(log_mu, log_sigma)
            durations_scenario[a_id] = max(1, round(dur))
        
        # Build SAA evaluation model with fixed first-stage
        m_eval = gp.Model(f"Baseline_Eval_{replicate_k}")
        
        # Create variables but fix first-stage
        x_eval = {}
        for a in activities:
            a_id = a['id']
            for g in a['required_resources']:
                insts = resource_instances.get(g, [])
                for inst in insts:
                    x_eval[(a_id, inst)] = m_eval.addVar(vtype=GRB.BINARY, 
                                                          name=f"x_{a_id}_{inst}")
                    # Fix to baseline solution
                    if (a_id, inst) in x_solution:
                        m_eval.addConstr(x_eval[(a_id, inst)] == 1)
                    else:
                        m_eval.addConstr(x_eval[(a_id, inst)] == 0)
        
        # Sequencing variables (fixed to baseline)
        s1_eval = {}
        s2_eval = {}
        act_types = [set(a['required_resources']) for a in activities]
        for i in range(num_acts):
            for j in range(i + 1, num_acts):
                if act_types[i].intersection(act_types[j]):
                    s1_eval[(i, j)] = m_eval.addVar(vtype=GRB.BINARY, name=f"s1_{i}_{j}")
                    s2_eval[(i, j)] = m_eval.addVar(vtype=GRB.BINARY, name=f"s2_{i}_{j}")
                    m_eval.addConstr(s1_eval[(i, j)] + s2_eval[(i, j)] <= 1)
                    
                    # Fix to baseline solution
                    if (i, j) in s1_solution:
                        m_eval.addConstr(s1_eval[(i, j)] == 1)
                    else:
                        m_eval.addConstr(s1_eval[(i, j)] == 0)
                    
                    if (i, j) in s2_solution:
                        m_eval.addConstr(s2_eval[(i, j)] == 1)
                    else:
                        m_eval.addConstr(s2_eval[(i, j)] == 0)
        
        # Start time variables (second stage, free)
        b_eval = {}
        for a in activities:
            a_id = a['id']
            b_eval[a_id] = m_eval.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"b_{a_id}")
        
        # Precedence constraints (2b-2c) with realized durations
        for a in activities:
            a_id = a['id']
            if a['is_start']:
                m_eval.addConstr(b_eval[a_id] >= a['scheduled_start'])
            else:
                prev_id = a['predecessor']
                prev_dur = durations_scenario[prev_id]
                m_eval.addConstr(b_eval[a_id] >= b_eval[prev_id] + prev_dur)
        
        # Non-overlap constraints (2d-2g) with realized durations
        for i in range(num_acts):
            for j in range(i + 1, num_acts):
                if act_types[i].intersection(act_types[j]):
                    dur_i = durations_scenario[i]
                    dur_j = durations_scenario[j]
                    
                    m_eval.addConstr(bigM * s1_eval[(i, j)] >= b_eval[i] - b_eval[j] + 1)
                    m_eval.addConstr(bigM * (1 - s1_eval[(i, j)]) >= b_eval[j] - b_eval[i])
                    m_eval.addConstr(bigM * s2_eval[(i, j)] >= b_eval[j] - b_eval[i] + dur_j)
                    m_eval.addConstr(bigM * (1 - s2_eval[(i, j)]) >= b_eval[i] - b_eval[j] - dur_j + 1)
        
        # Objective: minimize waiting time with this scenario's durations
        total_wait = gp.LinExpr()
        for a in activities:
            a_id = a['id']
            if a['is_start']:
                total_wait += b_eval[a_id] - a['scheduled_start']
            else:
                prev_id = a['predecessor']
                prev_dur = durations_scenario[prev_id]
                total_wait += b_eval[a_id] - (b_eval[prev_id] + prev_dur)
        
        m_eval.setObjective(total_wait, GRB.MINIMIZE)
        
        m_eval.setParam('OutputFlag', 0)
        m_eval.setParam('TimeLimit', time_limit)
        m_eval.setParam('MIPGap', gap)
        
        m_eval.optimize()
        
        if m_eval.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
            objectives.append(m_eval.ObjVal)
        else:
            print(f"    Warning: Evaluation replicate {replicate_k + 1} failed.")
    
    if objectives:
        avg_obj_value = np.mean(objectives)
    else:
        avg_obj_value = float('inf')
    
    print(f"Baseline mean value model evaluation: v̄base_N' = {avg_obj_value:.4f}")
    
    return avg_obj_value, objectives