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
- `src/`: modular Python code for preprocessing, training, and evaluation.  
- `results/`: final metrics and plots.

````
Kaggle_Network_Anomaly_Detection/
├── data/                   
├── notebooks/               
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Modeling.ipynb
├── src/                     
│   ├── __init__.py
│   ├── preprocessing.py      
│   ├── training.py           
│   └── utils.py
├── results/                 
├── requirements.txt          
└── README.md                 
````

## Tech Stack
- Python 3.10  
- scikit-learn  
- pandas, numpy  
- matplotlib, seaborn  
- Jupyter  

## Next Steps
- Try different classifiers: Logistic Regression, Random Forests, XGBoost.  
- Tune hyperparameters with cross-validation.  
- Compare centralized and feature-engineered approaches.  

 
