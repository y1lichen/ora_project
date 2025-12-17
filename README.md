# Multi-Resource Allocation and Care Sequence Optimization (SMILP)

本專案實作了 [Multi-resource allocation and care sequence assignment in patient management: a stochastic programming approach. Health Care Management Science](https://link.springer.com/article/10.1007/s10729-024-09675-6)。核心目標是在醫療資源（人力、設備）受限且病患活動工期具有不確定性的情況下，透過兩階段隨機混合整數線性規劃 (Two-stage SMILP) 模型，極小化病患的總體等待時間。

## 論文背景
在高粒度的門診環境中，病患的照護流程（Care Pathways）涉及多種資源分配。傳統的「確定性模型」無法應對工期波動，導致排程效果不佳。本研究提出：

* 第一階段決策 (Planning)：在觀察到實際工期前，決定資源分配 (x) 與活動先後順序 (s1​,s2​)。

* 第二階段決策 (Recourse)：根據實際發生的工期，決定各活動的開始時間 (b)。

* 方法論：結合樣本平均近似法 (SAA) 與 蒙地卡羅優化 (MCO)，自動尋找兼顧精確度與運算效率的最佳樣本數量。

## 資料集
論文使用的原始資料集是使用 Real-Time Location System(RTLS) 取得病患在骨科診所的位置資料，推估病患在診療流程中各個活動的時長。由於論文並未提供資料集，故本專案依論文提供的統計分佈生成數據，生成數據的程式碼為 [instance_generator.py](./instance_generator.py)，

| Path ID | Activity List                                   | Resource Interaction Probability | Mean (minutes)                                   | Variance                                         |
|---------|-------------------------------------------------|---------------------------------|-------------------------------------------------|-------------------------------------------------|
| 0       | Intake, Radiology Tech, Provider, Ortho Tech, Discharge | 38.03%                         | [4.7, 3.48, 4.52, 11.62, 3.43]                  | [3.25, 4.85, 13.9, 185.52, 2.63]               |
| 1       | Intake, Provider, Ortho Tech, Discharge         | 24.55%                         | [4.93, 4.99, 12.47, 3.69]                       | [4.0, 20.11, 206.96, 2.92]                     |
| 2       | Intake, Ortho Tech, Discharge                   | 13.93%                         | [4.75, 11.82, 3.44]                             | [4.35, 232.7, 3.36]                            |
| 3       | Intake, Radiology Tech, Ortho Tech, Discharge   | 13.78%                         | [4.91, 3.58, 11.25, 3.49]                       | [3.75, 5.79, 224.31, 3.31]                     |
| 4       | Intake, Radiology Tech, Provider, Discharge     | 9.71%                          | [5.06, 3.52, 6.15, 3.62]                        | [3.98, 4.95, 30.61, 4.06]                      |

## Model Overview: Two-Stage SMILP for Healthcare Resource Allocation
 
The model is designed for clinic or outpatient settings where patients follow predefined care pathways consisting of multiple sequential activities that compete for shared, capacity-limited resources.

---

## 1. Problem Overview

We consider a healthcare system with:

- **Patients** who require a sequence of medical activities (e.g., consultation → exam → treatment)
- **Multiple resource types** (e.g., physicians, nurses, rooms, equipment)
- **Uncertain activity durations**, modeled as random variables
- **Pre-scheduled appointment times** for each patient

The goal is to:

> **Assign resources and determine activity sequences in advance, such that the expected total patient waiting time is minimized**, while respecting resource capacities and precedence constraints.

This problem is challenging because:
- Resource assignment decisions must be made **before** service durations are realized.
- Activity start times depend on **realized durations** and therefore must be decided **after** uncertainty is revealed.

---

## 2. Model Structure

The model is formulated as a **two-stage stochastic program with recourse**:

### Stage 1 (Planning Decisions – Before Uncertainty)
- Assign activities to specific resources
- Decide relative sequencing between activities that share resource types
- Enforce resource capacity feasibility

### Stage 2 (Operational Decisions – After Uncertainty)
- Determine actual activity start times
- Compute realized patient waiting times for each scenario

The objective is to **minimize expected total waiting time** across all patients.

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
| `Jg` | Resources of type `g` |
| `A(g)` | Activities requiring resource type `g` |

---

## 4. Parameters

| Parameter | Description |
|----------|-------------|
| `ta` | Scheduled appointment time for initial activity `a ∈ A0` |
| `da` | Random duration of activity `a` |
| `Va,g` | Number of type-`g` resources required by activity `a` |
| `kj` | Capacity of resource `j` |
| `u` | Planning horizon |

Activity durations `da` are **random variables** with known distributions estimated from historical data.

---

## 5. Decision Variables

### First-Stage Variables (Scenario-Independent)

| Variable | Type | Description |
|--------|------|-------------|
| `xᵃⱼ` | Binary | 1 if resource `j` is assigned to activity `a` |
| `s¹ₐ,ₐ′` | Binary | 1 if activity `a` is not scheduled before `a′` |
| `s²ₐ,ₐ′` | Binary | 1 if activity `a` starts before `a′` ends |
| `qⱼₐ,ₐ′` | Binary | 1 if activity `a′` is ongoing when `a` starts and both use resource `j` |

### Second-Stage Variables (Scenario-Dependent)

| Variable | Type | Description |
|--------|------|-------------|
| `bᵃ` | Continuous | Start time of activity `a` |

---

## 6. Objective Function

### Expected Waiting Time Minimization

The objective minimizes:
- Waiting after scheduled appointment time (initial activities)
- Waiting between consecutive activities along a patient's pathway

Formally:

- **Stage 1**:  
  \[
  \min \mathbb{E}_\xi[Q(x, s_1, s_2, q, \xi)]
  \]

- **Stage 2 (per scenario)**:  
  Sum of waiting times induced by realized durations

---

## 7. Constraints

### 7.1 Resource Assignment Constraints
Ensure that each activity receives the required number of resources of each type:

\[
\sum_{j \in J_g} x^a_j = V_{a,g}
\]

---

### 7.2 Capacity Constraints (Key Innovation)

Instead of time-indexed formulations, **capacity is enforced only at activity start times**.

- `qⱼₐ,ₐ′` identifies whether activity `a′` overlaps with `a` on resource `j`
- The number of overlapping activities must not exceed resource capacity:

\[
\sum_{a' \neq a} q^j_{a,a'} \le k_j - 1
\]

This dramatically reduces model size and complexity.

---

### 7.3 Logical Linking Constraints

Big-M constraints link:
- Resource assignments (`x`)
- Sequencing decisions (`s₁`, `s₂`)
- Overlap indicators (`q`)

These ensure `qⱼₐ,ₐ′ = 1` **if and only if**:
- Both activities use resource `j`
- `a′` has started
- `a′` has not yet finished when `a` starts

---

### 7.4 Precedence Constraints (Second Stage)

- Initial activities cannot start before appointment time
- Subsequent activities cannot start before their predecessors finish

\[
b_a \ge t_a \quad (a \in A_0)
\]
\[
b_a \ge b_{\text{pre}(a)} + d_{\text{pre}(a)} \quad (a \in A_1)
\]

---

### 7.5 Sequencing Constraints

Big-M constraints enforce relative ordering between activities that share resource types, ensuring consistency with `s₁` and `s₂`.

---

## 8. Monte Carlo Optimization (MCO) with SAA

Directly solving the SMILP is intractable due to the expectation over continuous distributions.

### Solution Approach

1. **Sample Average Approximation (SAA)**
   - Replace expectation with an average over `N` sampled duration scenarios
   - Results in a deterministic MILP

2. **Monte Carlo Optimization (MCO)**
   - Iteratively increase sample size `N`
   - Estimate lower and upper bounds via optimization + simulation
   - Terminate when the **Approximate Optimality Index (AOI)** is below tolerance `ε`

\[
AOI_N = \frac{|\bar{v}_{N'} - \bar{v}_N|}{\bar{v}_{N'}}
\]

This balances **solution quality** and **computational tractability**.

---

## 9. Key Innovations

### 1. Start-Time-Based Capacity Enforcement
- Avoids traditional time-indexed formulations
- Capacity is checked **only at activity start times**
- Significantly reduces binary variables and constraints

### 2. Explicit Modeling of Activity Overlaps
- Novel use of `s₁`, `s₂`, and `q` variables
- Precisely captures partial overlaps between activities sharing resources

### 3. True Two-Stage Healthcare Planning Model
- Reflects real-world operations:
  - Assign staff/resources **before** service day
  - Adjust start times **after** uncertainty is realized

### 4. Integrated Optimization–Simulation Framework
- MCO + SAA provides statistical guarantees
- Produces near-optimal solutions with controllable accuracy

---

## 10. Applicability

This framework is suitable for any service system with shared, capacity-limited resources and uncertain service times

---
