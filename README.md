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

## 