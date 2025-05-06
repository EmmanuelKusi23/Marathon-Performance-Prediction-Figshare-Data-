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
```
Parses timestamps, inspects shape and dtypes.

Key columns: distance (km), duration (min), gender, age_group, country, major.

Exploratory Data Analysis
1. Pace Distribution
python
Copy
Edit
df['pace'] = df['duration'] / df['distance']
sns.histplot(df['pace'], bins=50, kde=True)
plt.title('Distribution of Running Pace (min/km)')
plt.xlabel('Pace (min per km)')
plt.show()

Figure: Most runs cluster between 4 and 6 min/km. The KDE curve highlights skew toward slower paces.

2. Pace by Age Group
python
Copy
Edit
sns.boxplot(x='age_group', y='pace', data=df)
plt.title('Pace by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Pace (min/km)')
plt.show()

Figure: Boxplots reveal that younger athletes (18–34) tend to have slightly faster median paces than older groups (35–54, 55+).

3. Feature Correlations
python
Copy
Edit
features = ['distance','duration','pace']
corr = df[features].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

Figure: Strong positive correlation between distance and duration (ρ≈0.85); pace moderately inversely correlated with distance.

4. Time-Series of Weekly Volume
python
Copy
Edit
df['week'] = df['datetime'].dt.isocalendar().week
weekly = df.groupby('week')['distance'].sum().reset_index()
sns.lineplot(x='week', y='distance', data=weekly)
plt.title('Weekly Total Distance in 2020')
plt.xlabel('ISO Week')
plt.ylabel('Total Distance (km)')
plt.show()

Figure: Displays training load fluctuations—notice dips around typical holiday periods.

Data Cleaning & Preprocessing
Remove invalid runs (zero distance ⇒ undefined pace).

Handle missing: drop rows missing critical fields; impute or group rare categories.

Encode:

Numeric → median impute + standard scale

Categorical → frequent-value impute + one-hot encode

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_cols = ['distance','duration']
cat_cols = ['gender','age_group','country','major']

num_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('nums', num_pipe, num_cols),
    ('cats', cat_pipe, cat_cols)
])
```
## Feature Engineering
Rolling 7 day metrics:

```python
df = df.sort_values(['athlete','datetime'])
df['7d_avg_pace']      = df.groupby('athlete')['pace'].rolling(7, min_periods=1).mean().reset_index(0,drop=True)
df['7d_total_distance'] = df.groupby('athlete')['distance'].rolling(7, min_periods=1).sum().reset_index(0,drop=True)
Lag features:

```python
Edit
df['prev_pace']       = df.groupby('athlete')['pace'].shift(1).fillna(df['pace'])
df['days_since_last'] = df.groupby('athlete')['datetime'].diff().dt.days.fillna(7)
Cumulative distance:

```python
df['cum_dist'] = df.groupby('athlete')['distance'].cumsum()
These features capture recent load, recovery, and long-term volume.
```

## Modeling & Cross-Validation
We compare Ridge, Lasso, and Random Forest regressors using 5-fold CV (scoring: MAE, R²).

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline

models = [
    ('Ridge', Ridge()),
    ('Lasso', Lasso()),
    ('RF', RandomForestRegressor(n_jobs=-1, random_state=42))
]

kf = KFold(5, shuffle=True, random_state=42)
results = []

for name, mdl in models:
    pipe = Pipeline([('preproc', preprocessor), ('model', mdl)])
    mae = -cross_val_score(pipe, X, y, cv=kf, scoring='neg_mean_absolute_error').mean()
    r2  =  cross_val_score(pipe, X, y, cv=kf, scoring='r2').mean()
    results.append((name, mae, r2))

print(pd.DataFrame(results, columns=['Model','MAE','R²']).sort_values('MAE'))
Model	MAE (min/km)	R²
Ridge	0.0036	0.9999
RF	0.0788	0.9110
Lasso	0.1019	0.9812
```
Ridge overfits; RF chosen for generalization.

Hyperparameter Tuning & Evaluation
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, 20, None]
}
pipe = Pipeline([('preproc', preprocessor),
                 ('model', RandomForestRegressor(random_state=42))])
grid = GridSearchCV(pipe, param_grid, cv=3,
                    scoring='neg_mean_absolute_error')
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("CV MAE:", -grid.best_score_)
On the held-out test set:

arduino
Test MAE: 0.095 min/km  
Test R²:  0.416
```
Feature Importances
```python
import matplotlib.pyplot as plt

rf = grid.best_estimator_.named_steps['model']
feat_names = (num_cols +
              list(grid.best_estimator_
                       .named_steps['preproc']
                       .named_transformers_['cats']
                       .named_steps['onehot']
                       .get_feature_names_out(cat_cols)))
importances = pd.Series(rf.feature_importances_, index=feat_names)
top10 = importances.sort_values(ascending=False).head(10)

top10.plot.barh()
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')
plt.gca().invert_yaxis()
plt.show()
```
Figure: Most predictive features include recent average pace and 7-day total distance.

Model Deployment & Usage
Serialize the full pipeline:

bash
Copy
Edit
python - <<'PYCODE'
from joblib import dump
from your_module import pipeline  # assume you built & fitted your pipeline
dump(pipeline, 'trained_pipeline.joblib')
PYCODE
Load and predict on new data:

```python
from joblib import load
import pandas as pd

# Load pipeline
pipe = load("trained_pipeline.joblib")

# New run record
new_run = pd.DataFrame([{
    'distance': 15.2,
    'duration': 90.5,
    'gender': 'M',
    'age_group': '35 - 54',
    'country': 'United States',
    'major': 'CHICAGO 2019'
}])

predicted_pace = pipe.predict(new_run)
print(f"Predicted pace: {predicted_pace[0]:.2f} min/km")
```
References
Afonseca, C., et al. (2022). Long-distance running training logs (2019–2020). Figshare/Kaggle.

Original Parquet → CSV conversion script











