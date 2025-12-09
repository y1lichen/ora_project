import gurobipy as gp
from gurobipy import GRB
import numpy as np


class SMILPSolver:
    """
    Fully aligned with the Two-Stage Stochastic MILP model in Han et al. (2008).
    With corrected resource structure and no TypeError.
    """

    def __init__(self, data, time_limit=1800, mip_gap=0.05):

        self.data = data
        self.activities = data["activities"]

        # ★ resources = list of resource IDs, e.g. [0,1,2]
        self.resources = data["resources"]

        # ★ resource capacities (dict: {resource_id: capacity})
        # If your generator uses capacity=1 for all, we create default:
        if "resource_capacity" in data:
            self.resource_capacity = data["resource_capacity"]
        else:
            # default capacity = 1 for all
            self.resource_capacity = {rid: 1 for rid in self.resources}

        self.num_scenarios = data["num_scenarios"]

        # mapping: for each activity, list of valid resource IDs
        self.act_valid_resources = {
            act["id"]: act["required_resources"] for act in self.activities
        }

        # Big-M
        self.M = data.get("bigM", 10000)

        # Gurobi model
        self.model = gp.Model("SMILP")
        self.model.setParam("TimeLimit", time_limit)
        self.model.setParam("MIPGap", mip_gap)
        self.model.setParam("OutputFlag", 1)

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------
    def build_and_solve(self):

        m = self.model
        acts = self.activities
        scenarios = range(self.num_scenarios)

        # ------------------------------------------------------
        # Stage 1 Variables
        # ------------------------------------------------------

        # x[a, j]: assignment (binary)
        x = {}
        for a in acts:
            a_id = a["id"]
            for r in self.act_valid_resources[a_id]:
                x[(a_id, r)] = m.addVar(vtype=GRB.BINARY, name=f"x[{a_id},{r}]")

        # competition activity pairs (a,a')
        competition_pairs = []
        for i, a1 in enumerate(acts):
            for j, a2 in enumerate(acts):
                if i == j:
                    continue
                R1 = set(self.act_valid_resources[a1["id"]])
                R2 = set(self.act_valid_resources[a2["id"]])
                if not R1.isdisjoint(R2):
                    competition_pairs.append((a1["id"], a2["id"]))

        # s1[a,a’]
        s1 = m.addVars(competition_pairs, vtype=GRB.BINARY, name="s1")

        # s2[a,a’]
        s2 = m.addVars(competition_pairs, vtype=GRB.BINARY, name="s2")

        # q[j,a,a’]
        q_index = []
        for (a, a2) in competition_pairs:
            common = set(self.act_valid_resources[a]) & set(self.act_valid_resources[a2])
            for r in common:
                q_index.append((r, a, a2))

        q = m.addVars(q_index, vtype=GRB.BINARY, name="q")

        m.update()

        # ------------------------------------------------------
        # Stage 2 Variables: b[a, scenario]
        # ------------------------------------------------------
        b = m.addVars(
            [a["id"] for a in acts],
            scenarios,
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name="b",
        )

        # ------------------------------------------------------
        # Constraints (1b) Assignment
        # ------------------------------------------------------
        for a in acts:
            a_id = a["id"]
            valid = self.act_valid_resources[a_id]

            m.addConstr(
                gp.quicksum(x[(a_id, r)] for r in valid) == 1,
                name=f"assign[{a_id}]",
            )

        # ------------------------------------------------------
        # Constraints (1c) Definition of q
        # q[j,a,a'] ≥ s1[a,a'] + s2[a,a'] + x[a,j] + x[a',j] − 3
        # ------------------------------------------------------
        for (r, a, a2) in q_index:
            m.addConstr(
                q[(r, a, a2)]
                >= s1[(a, a2)] + s2[(a, a2)] + x[(a, r)] + x[(a2, r)] - 3,
                name=f"q_def[{r},{a},{a2}]",
            )

        # ------------------------------------------------------
        # Constraints (1d) Capacity
        # Σ q[j,a,a'] ≤ capacity[j] − 1
        # ------------------------------------------------------
        for r in self.resources:  # r is resource ID
            cap = self.resource_capacity.get(r, 1)

            q_terms = [
                q[(r, a, a2)]
                for (rr, a, a2) in q_index
                if rr == r
            ]

            if q_terms:
                m.addConstr(
                    gp.quicksum(q_terms) <= cap - 1,
                    name=f"cap[{r}]"
                )

        # ------------------------------------------------------
        # Stage 2: Precedence & Waiting (2b), (2c)
        # ------------------------------------------------------
        total_wait = 0

        for n in scenarios:

            for a in acts:
                a_id = a["id"]

                # (2b) Initial activity
                if a["is_start"]:
                    scheduled = a["scheduled_start"]
                    m.addConstr(b[(a_id, n)] >= scheduled)
                    total_wait += b[(a_id, n)] - scheduled

                else:
                    # (2c) Precedence: b[a] ≥ b[pre] + dur(pre, scenario)
                    pre = a["predecessor"]
                    pre_act = next(item for item in acts if item["id"] == pre)
                    dur = pre_act["durations"][n]

                    m.addConstr(
                        b[(a_id, n)] >= b[(pre, n)] + dur,
                        name=f"prec[{a_id},{n}]",
                    )

                    total_wait += b[(a_id, n)] - b[(pre, n)] - dur

        # ------------------------------------------------------
        # Sequencing (2d)–(2g)
        # ------------------------------------------------------
        for n in scenarios:
            for (a, a2) in competition_pairs:
                act2 = next(item for item in acts if item["id"] == a2)
                d2 = act2["durations"][n]

                # (2d)
                m.addConstr(
                    self.M * s1[(a, a2)] >= b[(a, n)] - b[(a2, n)],
                    name=f"s1d[{a},{a2},{n}]",
                )

                # (2e)
                m.addConstr(
                    self.M * (1 - s1[(a, a2)]) >= b[(a2, n)] - b[(a, n)],
                    name=f"s1e[{a},{a2},{n}]",
                )

                # (2f)
                m.addConstr(
                    self.M * s2[(a, a2)] >= b[(a2, n)] - b[(a, n)] + d2,
                    name=f"s2f[{a},{a2},{n}]",
                )

                # (2g)
                m.addConstr(
                    self.M * (1 - s2[(a, a2)]) >= b[(a, n)] - b[(a2, n)] - d2,
                    name=f"s2g[{a},{a2},{n}]",
                )

        # ------------------------------------------------------
        # Objective
        # ------------------------------------------------------
        m.setObjective(total_wait / self.num_scenarios, GRB.MINIMIZE)

        # ------------------------------------------------------
        # Solve
        # ------------------------------------------------------
        m.optimize()

        # ------------------------------------------------------
        # Extract solution
        # ------------------------------------------------------
        if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            sol_x = {k: v.X for k, v in x.items()}
            sol_s1 = {k: v.X for k, v in s1.items()}
            sol_s2 = {k: v.X for k, v in s2.items()}
            sol_q = {k: v.X for k, v in q.items()}

            return {
                "status": m.status,
                "objective": m.objVal,
                "x": sol_x,
                "s1": sol_s1,
                "s2": sol_s2,
                "q": sol_q,
            }

        return None
