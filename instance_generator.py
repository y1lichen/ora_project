# instance_generator.py
import numpy as np
import random
from copy import deepcopy

class InstanceGenerator:
    """
    Produce instances compatible with Two-Stage SMILP:
    - resources_by_type: mapping type -> list of unit resource dicts {'uid', 'type', 'orig_id'}
    - activities: list of activities with required_types (list of types), durations per scenario, predecessor, scheduled_start, is_start
    """

    def __init__(self, num_patients=15, arrival_interval=10, random_seed=42):
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.num_patients = num_patients
        self.arrival_interval = arrival_interval

        # Original resource types with capacity
        self.resources = {
            "Nurse":     {"capacity": 3, "id": 0},
            "Radiology Tech":   {"capacity": 1,   "id": 1},
            # "XRay Room":        {"capacity": 1,   "id": 2},
            "Provider":         {"capacity": 1,   "id": 3},
            "Ortho Tech":       {"capacity": 1,   "id": 4},
            # "Casting Room":     {"capacity": 1,   "id": 5},
        }

        # Activity mapping to resource types (names must match keys above)
        self.activity_resource_map = {
            "Intake":               ["Nurse"],
            # "Radiology":            ["Radiology Tech", "XRay Room"],
            "Radiology":            ["Radiology Tech"],
            "Provider Visit":       ["Provider"],
            # "Casting Procedure":    ["Ortho Tech", "Casting Room"],
            "Casting Procedure":    ["Ortho Tech"],
            "Discharge":            ["Nurse"],
        }

        # dominant pathways and distribution params (means, variances) - using your table
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

    def _expand_resource_units(self):
        """
        Convert self.resources like Nurse(cap=3) into explicit resource units,
        each unit has capacity 1 and unique uid integer.
        Returns:
            resources_by_type: mapping type_name -> list of unit dicts {'uid', 'type', 'orig_id'}
            resource_uid_to_type: mapping uid -> type_name
            next_uid: int next free uid
        """
        resources_by_type = {}
        resource_uid_to_type = {}
        uid = 0
        for rtype, rinfo in self.resources.items():
            cap = int(rinfo.get("capacity", 1))
            units = []
            for k in range(cap):
                units.append({"uid": uid, "type": rtype, "orig_id": rinfo.get("id", None)})
                resource_uid_to_type[uid] = rtype
                uid += 1
            resources_by_type[rtype] = units
        return resources_by_type, resource_uid_to_type

    def generate_data(self, num_scenarios=30):
        """
        Returns data dict with:
          - patients
          - activities: list of dicts with 'id','patient_id','name','required_types' (list), 'durations' (list),
                        'predecessor', 'is_start','scheduled_start','mean_duration','var_duration'
          - resources_by_type
          - resource_uid_to_type
          - num_scenarios
        """
        resources_by_type, resource_uid_to_type = self._expand_resource_units()

        patients = []
        activities = []
        current_arrival = 0
        activity_global_id = 0

        path_probs = np.array([p["prob"] for p in self.pathway_data])
        path_probs = path_probs / path_probs.sum()

        for pid in range(self.num_patients):
            # sample pathway type
            path_idx = np.random.choice(len(self.pathway_data), p=path_probs)
            pinfo = self.pathway_data[path_idx]
            seq = pinfo["sequence"]
            means = pinfo["means"]
            vars_ = pinfo["variances"]

            # arrival spacing
            interval = max(1, np.random.normal(self.arrival_interval, 1.0))
            current_arrival += int(round(interval))

            predecessor = None
            patient_act_ids = []

            for i, act_name in enumerate(seq):
                # required types
                req_types = self.activity_resource_map[act_name]
                mean_d = means[i]
                var_d = vars_[i]
                mu, sigma = self._convert_params_to_lognormal(mean_d, var_d)
                samples = np.maximum(1, np.round(np.random.lognormal(mu, sigma, num_scenarios)))
                # samples = np.clip(samples, 1, mean_d + 3*np.sqrt(var_d))
                act_id = activity_global_id
                activity_global_id += 1
                patient_act_ids.append(act_id)

                activities.append({
                    "id": act_id,
                    "patient_id": pid,
                    "name": act_name,
                    "required_types": deepcopy(req_types),  # list of type names
                    "durations": samples.tolist(),
                    "mean_duration": float(mean_d),
                    "var_duration": float(var_d),
                    "predecessor": predecessor,
                    "is_start": (i == 0),
                    "scheduled_start": int(current_arrival) if i == 0 else None
                })
                predecessor = act_id

            patients.append({
                "id": pid,
                "pathway_type": path_idx,
                "activity_ids": patient_act_ids,
                "arrival_time": current_arrival
            })

        data = {
            "patients": patients,
            "activities": activities,
            "resources_by_type": resources_by_type,
            "resource_uid_to_type": resource_uid_to_type,
            "num_scenarios": num_scenarios
        }
        return data

if __name__ == "__main__":
    generator = InstanceGenerator(num_patients=2, arrival_interval=10, random_seed=42)
    data = generator.generate_data(num_scenarios=5)
    import pprint
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(data)