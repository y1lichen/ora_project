# Multi-Resource Allocation and Care Sequence Optimization (Two-Stage SMILP)

This project implements the model proposed in  
**Yao et al. (2024), _Multi-resource allocation and care sequence assignment in patient management: A stochastic programming approach_, Health Care Management Science**.  
(https://link.springer.com/article/10.1007/s10729-024-09675-6)

The core objective is to **minimize total patient waiting time** in healthcare systems with **limited medical resources** (personnel and equipment) and **uncertain activity durations**, using a **two-stage stochastic mixed-integer linear programming (SMILP)** formulation.

---

## Background and Motivation

In high-granularity outpatient clinic settings, patient care pathways consist of multiple sequential activities that compete for shared medical resources. Traditional **deterministic scheduling models** fail to capture the variability in service durations, often leading to inefficient schedules and excessive patient waiting.

This work addresses these challenges by proposing:

- **First-stage (planning) decisions**:  
  Resource assignment (`x`) and activity sequencing (`s1`, `s2`) are determined **before** actual service durations are observed.

- **Second-stage (recourse) decisions**:  
  Activity start times (`b`) are determined **after** service durations are realized.

- **Solution methodology**:  
  A combination of **Sample Average Approximation (SAA)** and **Monte Carlo Optimization (MCO)** is used to automatically balance solution accuracy and computational efficiency by selecting an appropriate number of duration scenarios.

---

## Dataset and Data Generation

The original paper uses patient trajectory data collected via a **Real-Time Location System (RTLS)** in an orthopedic clinic to estimate activity durations along care pathways.

Since the original dataset is not publicly available, this project **synthetically generates data** based on the statistical distributions reported in the paper.  
The data generation code is provided in:

```

instance_generator.py

```

### Care Pathway Distributions

| Path ID | Activity Sequence                                   | Probability | Mean Duration (min)                      | Variance                                   |
|--------:|-----------------------------------------------------|-------------|-------------------------------------------|--------------------------------------------|
| 0 | Intake → Radiology Tech → Provider → Ortho Tech → Discharge | 38.03% | [4.7, 3.48, 4.52, 11.62, 3.43] | [3.25, 4.85, 13.9, 185.52, 2.63] |
| 1 | Intake → Provider → Ortho Tech → Discharge | 24.55% | [4.93, 4.99, 12.47, 3.69] | [4.0, 20.11, 206.96, 2.92] |
| 2 | Intake → Ortho Tech → Discharge | 13.93% | [4.75, 11.82, 3.44] | [4.35, 232.7, 3.36] |
| 3 | Intake → Radiology Tech → Ortho Tech → Discharge | 13.78% | [4.91, 3.58, 11.25, 3.49] | [3.75, 5.79, 224.31, 3.31] |
| 4 | Intake → Radiology Tech → Provider → Discharge | 9.71% | [5.06, 3.52, 6.15, 3.62] | [3.98, 4.95, 30.61, 4.06] |

---

## Model Overview: Two-Stage SMILP

The model is designed for outpatient or clinic environments where patients follow **predefined care pathways** consisting of sequential activities that compete for **shared, capacity-limited resources**.

---

## 1. Problem Description

We consider a healthcare system with:

- **Patients** requiring a sequence of medical activities  
  (e.g., consultation → exam → treatment)
- **Multiple resource types**  
  (e.g., physicians, nurses, rooms, equipment)
- **Uncertain activity durations**, modeled as random variables
- **Pre-scheduled appointment times**

The goal is to:

> **Assign resources and determine activity sequences in advance such that the expected total patient waiting time is minimized**, while respecting resource capacities and activity precedence constraints.

The main challenge arises because:
- Resource assignment and sequencing decisions must be made **before** service durations are known.
- Actual activity start times depend on **realized durations** and must be determined **after** uncertainty is revealed.

---

## 2. Model Structure

The problem is formulated as a **two-stage stochastic program with recourse**.

### Stage 1: Planning Decisions (Before Uncertainty)
- Assign activities to specific resources
- Decide relative sequencing between activities sharing the same resource type
- Enforce feasibility with respect to resource capacities

### Stage 2: Operational Decisions (After Uncertainty)
- Determine actual activity start times
- Compute realized patient waiting times for each duration scenario

The objective is to **minimize the expected total waiting time** across all patients.

---

## 3. Sets and Indices

| Symbol | Description |
|------|-------------|
| `I` | Set of patients |
| `A` | Set of all patient activities |
| `A0` | Set of initial activities (first activity of each patient) |
| `A1` | Set of subsequent activities |
| `J` | Set of individual resources |
| `G` | Set of resource types |
| `Jg` | Resources belonging to type `g` |
| `A(g)` | Activities requiring resource type `g` |

---

## 4. Parameters

| Parameter | Description |
|----------|-------------|
| `t_a` | Scheduled appointment time for initial activity `a ∈ A0` |
| `d_a` | Random duration of activity `a` |
| `V_{a,g}` | Number of type-`g` resources required by activity `a` |
| `k_j` | Capacity of resource `j` |
| `u` | Planning horizon |

Activity durations are random variables with known distributions estimated from historical data.

---

## 5. Decision Variables

### First-Stage Variables (Scenario-Independent)

| Variable | Type | Description |
|--------|------|-------------|
| `x_{a,j}` | Binary | 1 if resource `j` is assigned to activity `a` |
| `s1_{a,a'}` | Binary | 1 if activity `a` is not scheduled before `a'` |
| `s2_{a,a'}` | Binary | 1 if activity `a` starts before activity `a'` ends |
| `q_{j,a,a'}` | Binary | 1 if activity `a'` is ongoing when `a` starts and both use resource `j` |

### Second-Stage Variables (Scenario-Dependent)

| Variable | Type | Description |
|--------|------|-------------|
| `b_a` | Continuous | Start time of activity `a` |

---

## 6. Objective Function

The objective minimizes:

- Waiting time after scheduled appointment times (initial activities)
- Waiting time between consecutive activities along each patient pathway

Conceptually:

- **Stage 1** minimizes the expected second-stage waiting cost.
- **Stage 2** computes realized waiting times for each duration scenario.

---

## 7. Constraints

### 7.1 Resource Assignment Constraints

Each activity must receive the required number of resources of each type:

```

sum_{j in J_g} x_{a,j} = V_{a,g}

```

---

### 7.2 Capacity Constraints (Key Innovation)

Unlike traditional time-indexed formulations, **capacity is enforced only at activity start times**.

- `q_{j,a,a'}` indicates whether activity `a'` overlaps with activity `a` on resource `j`
- The total number of overlapping activities cannot exceed the capacity of the resource:

```

sum_{a' != a} q_{j,a,a'} <= k_j - 1

```

This significantly reduces model size and computational complexity.

---

### 7.3 Logical Linking Constraints

Big-M constraints link:

- Resource assignments (`x`)
- Sequencing decisions (`s1`, `s2`)
- Overlap indicators (`q`)

They ensure that `q_{j,a,a'} = 1` **if and only if**:

- Both activities use resource `j`
- Activity `a'` has started
- Activity `a'` has not finished when activity `a` starts

---

### 7.4 Precedence Constraints (Second Stage)

Precedence is enforced as follows:

- Initial activities cannot start before their scheduled appointment times:
```

b_a >= t_a, for a in A0

```

- Subsequent activities cannot start before their predecessor finishes:
```

b_a >= b_pre(a) + d_pre(a), for a in A1

```

These constraints are scenario-dependent due to random durations.

---

### 7.5 Sequencing Constraints

Big-M constraints enforce consistency between activity start times and the first-stage sequencing decisions (`s1`, `s2`) for activities sharing resource types.

---

## 8. Monte Carlo Optimization (MCO) with SAA

Solving the full SMILP directly is computationally intractable due to the expectation over continuous random variables.

### Solution Approach

1. **Sample Average Approximation (SAA)**  
   - Replace the expectation with an average over `N` sampled duration scenarios  
   - Results in a deterministic MILP

2. **Monte Carlo Optimization (MCO)**  
   - Iteratively increase sample size `N`
   - Estimate statistical lower and upper bounds via optimization and simulation
   - Terminate when the **Approximate Optimality Index (AOI)** falls below tolerance `ε`

```

AOI_N = |v̄_N' - v̄_N| / v̄_N'

```

This procedure balances **solution accuracy** and **computational efficiency**.

---

## Applicability

This framework applies to healthcare and service systems with:

- Sequential tasks
- Shared, capacity-limited resources
- Uncertain service durations

## Experiment Results

<img width="900" height="600" alt="smilp_paper_result" src="https://github.com/user-attachments/assets/927ff3bc-3287-4bbe-9f96-e97294cd5f10" />
  
In our experiment, we set 100 scenarios for MCO for only 3 iteration, due to the computation resource limit, the waiting time can improve 10% comparing with the baseline.  
The baseline model uses a generated scenario as the expected value for duration.


<img width="1346" height="673" alt="圖片" src="https://github.com/user-attachments/assets/afc32e62-31a6-4d30-bfe1-6b6fcf46b250" />

This figure shows the Gantt chart of the resources in one scenario of SMILP result.  
It is clear that the same activity can require multiple resources at the a time.

---
## Reference
<https://github.com/y1lichen/ora_project>
