# Network Traffic Anomaly Detection (Kaggle InClass)

This repository contains my solution to a Kaggle InClass competition on **anomaly detection in LTE mobile networks**.  
The goal is to train ML models capable of classifying cell activity as **normal (0)** or **unusual (1)**, based on traffic traces collected from a real LTE deployment.  

## Problem Context
Mobile networks are typically overprovisioned to handle peak-hour traffic. Detecting **unusual patterns** in user demand (e.g., sports events, strikes, demonstrations) is essential to dynamically reconfigure base stations and optimize energy/resource usage.

## Dataset
- Collected from **10 LTE base stations**, over **2 weeks**, with samples every 15 minutes.  
- Features include PRB usage, throughput, number of active users, and cell identifiers.  
- Target variable:  
  - `0 = Normal behavior`  
  - `1 = Unusual behavior`  

## Evaluation Metric
The competition is evaluated using the **Mean F1-Score**, balancing **precision** and **recall**.  

## Project Structure
- `notebooks/`: step-by-step EDA, feature engineering, and modeling.  
- `src/`: modular Python code for configuration, training and evaluation.

````
Kaggle_Network_Anomaly_Detection/
├── data/                   
├── notebooks/               
│   ├── notebooks.ipynb #EDA, feature engineering and training trials
│   
├── src/                     
│   ├── __init__.py
│   ├── config.py      
│   ├── dataset.py
│   ├── model.py            
│   └── predict.py
│              
├── requirements.txt          
└── README.md                 
````

## Dependencies
- Python 3.10  
- scikit-learn  
- pandas, numpy  
- matplotlib, seaborn  
- Jupyter 

 
