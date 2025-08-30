# src/dataset.py
import pandas as pd
#from src.config import DATA_FILE  #comment to work in notebook
from config import DATA_LABELED,DATA_UNLABELED #comment to work in model.py

# def load_data():
#     return pd.read_csv(DATA_FILE)

def load_data_model():
    return pd.read_csv(DATA_LABELED,encoding="latin1")

def load_data_test():
    return pd.read_csv(DATA_UNLABELED,encoding="latin1")
