# MORL  
Multi-Objective Reinforcement Learning  
  
## Env: Fruit Tree 
Parameter Tuning:  
  
* Naive Version (Gamma=0.99, epsilon=0.5+linear decay, 5000 Episodes, Adam Optimizer)  

| No. | Memory | Size | Batch Size | Weight (Samples) Number | Update Frequency |  Learning Rate | Practical F1 | Prediction F1 | Practical Discrepancy | Prediction Discrepancy |  
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |  
| 0 | 4000 | 256 | 32 | 100 | 1e-3 | 0.942 | -  | 0.125 | **0.191** |  
| 1 | 8000 | 256 |  32 | 100 | 1e- 3| **0.951** | - | **0.119** | 0.209 |  
| 2 | 4000 | 512 | 16| 100 |  1e-3 |  0.906 |  - | 0.180 | 0.284 |  
| 3 | 4000 | 128 |  64 | 100 | 1e-3 | 0.915 | - | 0.142 | 0.215 |  
| 4 | 4000 | 256 | 32 | 50  | 1e-3 |  0.906 |  - | 0.150 |  0.200 |  
| 5 | 4000 | 256 | 32 |  100 | 5e- 4 | 0.836 |  - | 0.241 | 0.271 |  
  
  
* Envelope Version (Gamma=0.99, epsilon=0.5+linear decay, 5000 Episodes, Adam Optimizer, beta = 0.01)  

No. | Memory Size | Batch Size | Weight (Samples) Number | Update Frequency  | Learning Rate | Practical F1 | Prediction F1 | Practical Discrepancy | Prediction Discrepancy  
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---  
0 | 4000 | 256 | 32 | 100 | 1e-3 | 0.968 | 0.004 | 0.163 | 0.209  
1 | 8000 | 256 | 32 | 100 | 1e-3 | 0.757 | 0.001 | 0.213 | 0.258  
2 | 4000 | 512 | 16 | 100 | 1e-3 | **0.976** | 0.005 | 0.056 | 0.211  
3 | 4000 | 128 | 64 | 100 | 1e-3 | 0.959 | **0.011** | **0.051** | **0.115**  
4 | 4000 | 256 | 32 | 50 | 1e-3 | 0.804 | 0.000 | 0.285 | 0.294  
5 | 4000 | 256 | 32 | 100 | 5e-4 | 0.897 | 0.000 | 0.218 | 0.239  
  
* Envelope Version + Tuning Beta (Gamma=0.99, epsilon=0.5+linear decay, 5000 Episodes, Adam Optimizer)  
    * Configuration 0:  

No. | Beta | Practical F1 | Prediction F1 | Practical Discrepancy | Prediction Discrepancy  
--- | --- | --- | --- | --- | ---  
0+0 | 0.00 | 0.976 | 0.000 | 0.070 | 0.136  
0+1 | 0.01 | **0.992** | 0.007 | 0.042 | **0.128**  
0+2 | 0.02 | 0.976 | 0.006 | **0.077** | 0.143  
0+3 | 0.05 | **0.992** | 0.018 | 0.068 | 0.143  
0+4 | 0.10 | **0.992** | 0.018 | 0.065 | 0.161  
0+5 | 0.20 | 0.976 | 0.001 | 0.111 | 0.290  
0+6 | 0.50 | **0.992** | 0.011 | 0.105 | 0.234  
0+7 | 1.00 | 0.959 | **0.029** | 0.107 | 0.247  

* 
    * Configuration 3: 

No. | Beta | Practical F1 | Prediction F1 | Practical Discrepancy | Prediction Discrepancy  
--- | --- | --- | --- | --- | ---  
3+0 | 0.00 | 0.984 | 0.000 | 0.061 | 0.123  
3+1 | 0.01 | 0.959 | 0.010 | **0.042** | **0.106**  
3+2 | 0.02 | 0.976 | 0.012 | 0.047 | 0.144  
3+3 | 0.05 | 0.942 | 0.009 | 0.095 | 0.228  
3+4 | 0.10 | 0.976 | **0.035** | 0.067 | 0.144  
3+5 | 0.20 | 0.968 | 0.023 | 0.076 | 0.240  
3+6 | 0.50 | **0.992** | 0.022 | 0.107 | 0.286  
3+7 | 1.00 | 0.976 | 0.024 | 0.105 | 0.387  
  
* Envelope Version + Homotopy  (Gamma=0.99, epsilon=0.5+linear decay, 5000 Episodes, Adam Optimizer)  
    * Linear Transform + End beta = 0.50 + Configuration 0:  

No. | Practical F1 | Prediction F1 | Practical Discrepancy | Prediction Discrepancy  
--- | --- | --- | --- | ---   
0+h | **1.000** | 0.010 | 0.084 | 0.236  
1+h | 0.897 | 0.022 | 0.134 | 0.258  
2+h | 0.992 | 0.011 | 0.101 | 0.316  
3+h | 0.959 | **0.060** | **0.083** | 0.247  
4+h | 0.968 | 0.008 | 0.106 | 0.246  
5+h | 0.968 | 0.014 | 0.140 | **0.227**  

* 
    * Exponential Transformation (tau = 1000., 1 -> 0)  

No. | Practical F1 | Prediction F1 | Practical Discrepancy | Prediction Discrepancy  
--- | --- | --- | --- | ---   
0+he | 0.984 | 0.035 | 0.053 | 0.256  
1+he | **1.000** | 0.058 | 0055 | 0.148  
2+he | 0.984 | 0.039 | 0.071 | 0.200  
3+he | 0.992 | 0.071 | 0.053 | 0.170  
4+he | **1.000** | 0.057 | **0.042** | 0.195  
5+he | 0.992 | 0.017 | 0.090 | 0.189
m+he | 0.984 | **0.088** | 0.060 | **0.138**


No. | Practical F1 | Prediction F1 | Practical Discrepancy | Prediction Discrepancy  
--- | --- | --- | --- | ---   
0+_e | 0.992 | 0.042 | 0.053 | 0.256  
1+_e | 0.992 | 0.048 | 0055 | **0.148**  
2+_e | 0.992 | 0.064 | 0.071 | 0.200  
3+_e | 0.951 | 0.075 | 0.324 | 0.163  
4+_e | 0.959 | 0.050 | 0.062 | 0.192  
5+_e | 0.992 | 0.016 | 0.080 | 0.196  
m+_e | **1.000** | **0.075** | **0.045** | 0.163  

where m = 1 + 3 + 4  
