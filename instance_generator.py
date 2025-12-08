import numpy as np
import random


class InstanceGenerator:
    def __init__(self, num_patients=20, arrival_interval=10, random_seed=42):
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.num_patients = num_patients
        self.arrival_interval = arrival_interval

        # ---------------------------------------------------------
        # 1. Resource definitions (可多資源)
        # ---------------------------------------------------------
        self.resources = {
            "Intake Nurse":     {"capacity": 3, "id": 0},
            "Radiology Tech":   {"capacity": 1,   "id": 1},
            "XRay Room":        {"capacity": 1,   "id": 2},
            "Provider":         {"capacity": 1,   "id": 3},
            "Ortho Tech":       {"capacity": 1,   "id": 4},
            "Casting Room":     {"capacity": 1,   "id": 5},
            "Discharge Nurse":  {"capacity": 3, "id": 6},
        }

        # ---------------------------------------------------------
        # 2. Activity → multi-resource mapping
        # ---------------------------------------------------------
        self.activity_resource_map = {
            "Intake":               ["Intake Nurse"],
            "Radiology":            ["Radiology Tech", "XRay Room"],
            "Provider Visit":       ["Provider"],
            "Casting Procedure":    ["Ortho Tech", "Casting Room"],
            "Discharge":            ["Discharge Nurse"],
        }

        # ---------------------------------------------------------
        # 3. RTLS-based dominant pathways
        # ---------------------------------------------------------
        self.pathway_data = [
            {
                "prob": 0.3803,
                "sequence": ["Intake", "Radiology", "Provider Visit", "Casting Procedure", "Discharge"],
                "means":     [4.7, 3.48, 4.52, 11.62, 3.43],
                "variances": [3.25, 4.85, 13.9, 185.52, 2.63],
            },
            {
                "prob": 0.2455,
                "sequence": ["Intake", "Provider Visit", "Casting Procedure", "Discharge"],
                "means":     [4.93, 4.99, 12.47, 3.69],
                "variances": [4.0, 20.11, 206.96, 2.92],
            },
            {
                "prob": 0.1393,
                "sequence": ["Intake", "Casting Procedure", "Discharge"],
                "means":     [4.75, 11.82, 3.44],
                "variances": [4.35, 232.7, 3.36],
            },
            {
                "prob": 0.1378,
                "sequence": ["Intake", "Radiology", "Casting Procedure", "Discharge"],
                "means":     [4.91, 3.58, 11.25, 3.49],
                "variances": [3.75, 5.79, 224.31, 3.31],
            },
            {
                "prob": 0.0971,
                "sequence": ["Intake", "Radiology", "Provider Visit", "Discharge"],
                "means":     [5.06, 3.52, 6.15, 3.62],
                "variances": [3.98, 4.95, 30.61, 4.06],
            },
        ]

    def _convert_params_to_lognormal(self, mean, var):
        sigma2 = np.log(1 + var / (mean ** 2))
        sigma = np.sqrt(sigma2)
        mu = np.log(mean) - 0.5 * sigma2
        return mu, sigma

    def generate_data(self, num_scenarios=30):
        patients = []
        activities = []

        current_arrival = 0
        activity_global_id = 0

        path_probs = np.array([p["prob"] for p in self.pathway_data])
        path_probs /= path_probs.sum()

        for pid in range(self.num_patients):

            # ------------- Pathway selection -------------
            path_idx = np.random.choice(len(self.pathway_data), p=path_probs)
            pinfo = self.pathway_data[path_idx]

            seq = pinfo["sequence"]
            means = pinfo["means"]
            vars = pinfo["variances"]

            patient_act_ids = []

            # ----- arrival -----
            interval = max(1, np.random.normal(self.arrival_interval, 1.0))
            current_arrival += int(round(interval))

            predecessor = None

            # ---------------- Generate activities ----------------
            for i, act_name in enumerate(seq):

                required_res = [
                    self.resources[r]["id"] for r in self.activity_resource_map[act_name]
                ]

                mean_d = means[i]
                var_d = vars[i]

                mu, sigma = self._convert_params_to_lognormal(mean_d, var_d)

                samples = np.maximum(
                    1, np.round(np.random.lognormal(mu, sigma, num_scenarios))
                )

                act_id = activity_global_id
                activity_global_id += 1
                patient_act_ids.append(act_id)

                activities.append({
                    "id": act_id,
                    "patient_id": pid,
                    "name": act_name,
                    "required_resources": required_res,
                    "durations": samples.tolist(),
                    "mean_duration": mean_d,
                    "var_duration": var_d,
                    "variance": var_d,
                    "predecessor": predecessor,
                    "is_start": i == 0,
                    "scheduled_start": current_arrival if i == 0 else None
                })

                predecessor = act_id

            patients.append({
                "id": pid,
                "pathway_type": path_idx,
                "activity_ids": patient_act_ids,
                "arrival_time": current_arrival
            })

        return {
            "patients": patients,
            "activities": activities,
            "resources": self.resources,
            "num_scenarios": num_scenarios,
        }
