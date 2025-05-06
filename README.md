# Marathon-Performance-Prediction-Figshare-Data-

# Long-Distance Running Performance Prediction

This project demonstrates a full machine-learning pipeline for predicting running pace using a massive public dataset of long-distance training logs compiled by Afonseca *et al.* (2022). The dataset contains **10,703,690** runs from **36,412** athletes worldwide (2019–2020), originally distributed as Parquet files (via Figshare/Kaggle) and converted to CSV for processing.

---

## Table of Contents

1. [Project Background](#project-background)  
2. [Data Loading & Setup](#data-loading--setup)  
3. [Exploratory Data Analysis](#exploratory-data-analysis)  
4. [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)  
5. [Feature Engineering](#feature-engineering)  
6. [Modeling & Cross-Validation](#modeling--cross-validation)  
7. [Hyperparameter Tuning & Evaluation](#hyperparameter-tuning--evaluation)  
8. [Model Deployment & Usage](#model-deployment--usage)  
9. [References](#references)  

---

## Project Background

- **Dataset**: Afonseca *et al.* (2022) “Long-distance running training logs (2019–2020)”  
- **Source**: Figshare / Kaggle (Parquet format)  
- **Size**: 10,703,690 sessions · 36,412 athletes  
- **Focus**: 2020 data (58,326 runs · 8 columns)  
- **Goal**: Predict per-run pace (min/km) from training and athlete features  

---

## Data Loading & Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Convert Parquet → CSV once, then:
df = pd.read_csv('data/run_2020.csv', parse_dates=['datetime'])
print(df.shape)        # (58_326, 8)
print(df.dtypes)
df.head()

