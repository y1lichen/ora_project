import numpy as np
import random

class InstanceGenerator:
    def __init__(self, num_patients=20, arrival_interval=10, random_seed=42):
 
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.num_patients = num_patients
        self.arrival_interval = arrival_interval
        
       
        self.resources = {
            'Intake':        {'capacity': 100, 'id': 0}, 
            'Radiology Tech':{'capacity': 1,   'id': 1}, # 瓶頸
            'Provider':      {'capacity': 2,   'id': 2}, # 瓶頸
            'Ortho Tech':    {'capacity': 1,   'id': 3}, # 瓶頸
            'Discharge':     {'capacity': 100, 'id': 4}
        }
        

        self.pathway_data = [
            {
                'prob': 0.3803,
                'sequence': ['Intake', 'Radiology Tech', 'Provider', 'Ortho Tech', 'Discharge'],
                'means':    [4.7, 3.48, 4.52, 11.62, 3.43],
                'variances':[3.25, 4.85, 13.9, 185.52, 2.63]
            },
            {
                'prob': 0.2455,
                'sequence': ['Intake', 'Provider', 'Ortho Tech', 'Discharge'],
                'means':    [4.93, 4.99, 12.47, 3.69],
                'variances':[4.0, 20.11, 206.96, 2.92]
            },
            {
                'prob': 0.1393,
                'sequence': ['Intake', 'Ortho Tech', 'Discharge'],
                'means':    [4.75, 11.82, 3.44],
                'variances':[4.35, 232.7, 3.36]
            },
            {
                'prob': 0.1378,
                'sequence': ['Intake', 'Radiology Tech', 'Ortho Tech', 'Discharge'],
                'means':    [4.91, 3.58, 11.25, 3.49],
                'variances':[3.75, 5.79, 224.31, 3.31]
            },
            {
                'prob': 0.0971,
                'sequence': ['Intake', 'Radiology Tech', 'Provider', 'Discharge'],
                'means':    [5.06, 3.52, 6.15, 3.62],
                'variances':[3.98, 4.95, 30.61, 4.06]
            }
        ]

    def _convert_params_to_lognormal(self, mean, var):

        sigma2 = np.log(1 + var / (mean**2))
        sigma = np.sqrt(sigma2)
        mu = np.log(mean) - 0.5 * sigma2
        return mu, sigma

    def generate_data(self, num_scenarios=1):
        patients = []
        activities = []
        
        current_arrival_time = 0
        
        activity_global_id = 0
        
        # 正規化路徑機率
        path_probs = [p['prob'] for p in self.pathway_data]
        path_probs = np.array(path_probs) / np.sum(path_probs)
        
        for p_id in range(self.num_patients):
            path_idx = np.random.choice(len(self.pathway_data), p=path_probs)
            path_info = self.pathway_data[path_idx]
            
            sequence = path_info['sequence']
            means = path_info['means']
            variances = path_info['variances']
            
            patient_activities = []
            
            interval = max(1, np.random.normal(self.arrival_interval, 1.0)) 
            current_arrival_time += int(round(interval))
            
            previous_act_id = None
            
            for step_idx, res_name in enumerate(sequence):
                act_id = activity_global_id
                activity_global_id += 1
                
                arithmetic_mean = means[step_idx]
                arithmetic_var = variances[step_idx]
                
                mu, sigma = self._convert_params_to_lognormal(arithmetic_mean, arithmetic_var)
                
                durations = np.random.lognormal(mu, sigma, num_scenarios)
                durations = np.maximum(1, np.round(durations)) # 至少1分鐘
                
                act_data = {
                    'id': act_id,
                    'patient_id': p_id,
                    'resource_type': self.resources[res_name]['id'],
                    'resource_name': res_name,
                    'durations': durations.tolist(),
                    'mean_duration': arithmetic_mean,
                    'is_start': (step_idx == 0),
                    'scheduled_start': current_arrival_time if step_idx == 0 else 0,
                    'predecessor': previous_act_id
                }
                
                activities.append(act_data)
                patient_activities.append(act_id)
                previous_act_id = act_id
            
            patients.append({
                'id': p_id,
                'pathway_type': path_idx,
                'activity_ids': patient_activities,
                'arrival_time': current_arrival_time
            })
            
        return {
            'patients': patients,
            'activities': activities,
            'resources': self.resources,
            'num_scenarios': num_scenarios
        }