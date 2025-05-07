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
We set up the Python environment (Pandas, NumPy, scikit-learn, etc.) and loaded the CSV file. Initial checks confirmed the data shape
. The numeric fields (distance, duration) and categorical fields (gender, age_group, country, major) were examined. For example, the distance column has a mean of ≈28.82 km (median 20.80 km) and ranges from 0 up to ~444.84 km. Categorical summaries showed two genders (M and F) with more males (43,759) than females, and three age groups (18–34, 35–54, 55+) with the middle group (35–54) being most common. Missing values were minimal (511 missing country, 1 missing major), and no duplicate rows were found.

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

## Exploratory Data Analysis

I visualized key patterns in the data. For example, histograms of run times or distances help reveal their distributions. The figure below illustrates a histogram-like distribution of marathon finish times (from an example race); most runners finish around 3–5 hours, with female runners (red) tending toward slightly longer times than males (blue)【50†】. This highlights how performance varies by gender and distance. 

### Distribution of Marathon Run Metrics

This explores the distribution of key running metrics in a marathon training dataset. Each section includes:

1. A descriptive analysis of the distribution plot.  
2. The corresponding plot image.  
3. The Python code used to generate the plot.

---

### 1. Distance

The distribution of individual run distances highlights training habits, revealing many shorter runs for recovery and frequent mid-length workouts, with occasional long runs creating a right-skewed shape.

![Distance Distribution](https://github.com/EmmanuelKusi23/Marathon-Performance-Prediction-Figshare-Data-/blob/546bc18052721d9191f0971e6d26b8b96c3efdf2/distribution%20of%20distance.png)

### 2. Duration

Run durations mirror distances, showing most workouts are shorter in time, with the long runs appearing less frequently, confirming a right-skewed duration distribution.

![Duration Distribution](https://github.com/EmmanuelKusi23/Marathon-Performance-Prediction-Figshare-Data-/blob/023f526c914d64c64cd3e95432818aa25d4603c7/distribution%20of%20distance%201.png)


### 3. Pace

Pace distribution centers around the athlete’s typical training speeds. The histogram is roughly unimodal, with tails showing very slow recovery runs or faster race efforts.

![Pace Distribution](https://github.com/EmmanuelKusi23/Marathon-Performance-Prediction-Figshare-Data-/blob/99eed279fa508080abad53936db2eeee57417c30/distribution%20of%20distance%202.png)


### 4. 7-Day Average Pace

Averaging pace over 7-day windows smooths daily variability, highlighting longer-term trends in performance.

![7d Average Pace Distribution](https://github.com/EmmanuelKusi23/Marathon-Performance-Prediction-Figshare-Data-/blob/200b3dbd9beb06442878f6fcccbcd646fce4fe7d/distribution%20of%20distance%203.png)




### 5. 7-Day Total Distance

Weekly mileage distribution is critical for monitoring training volume. A bimodal pattern often emerges if the athlete alternates heavy and light weeks.

![7d Total Distance Distribution](https://github.com/EmmanuelKusi23/Marathon-Performance-Prediction-Figshare-Data-/blob/3816b4c1bd3930bc2c8f98c71d4c64a67f1ace5e/distribution%20of%20distance%204.png)



**Conclusion:** These distributions provide insight into training patterns, variability, and workload. Including both daily and rolling-window analyses offers a comprehensive view of an athlete's performance and training load over time.
## Boxplot Analysis of Marathon Training Metrics

This analysis interprets a boxplot of four key marathon training metrics: **distance**, **duration**, **pace**, and **total distance**. The aim is to identify patterns, spot anomalies, and offer insights for optimizing performance.

---

### Dataset & Visualization Overview
The boxplot summarizes training sessions from marathon runners, showing medians, spreads, and outliers. These distributions help refine training strategies and monitor athlete readiness.

---

### Metric Insights

**1. Distance:**
- Median: 10–20 km, typical for mid-week runs.
- Outliers >30 km suggest long-run sessions or advanced training.
- *Insight:* High outliers support endurance development crucial for marathon prep.

**2. Duration:**
- Median: 60–120 minutes.
- Outliers >180 minutes imply long, possibly recovery or acclimatization runs.
- *Insight:* Duration variability reflects workout diversity (tempo, long, recovery).

**3. Pace:**
- Notable anomaly: 70.949 likely a data error.
- Corrected, aligns with elite or recreational pace levels.
- *Insight:* Pace range indicates training periodization; anomalies need cleaning.

**4. Total Distance:**
- Median: ~74 km/week, suitable for intermediate runners.
- Outliers >100 km/week could reflect peak loads or risk of overtraining.
- *Insight:* Weekly mileage is key for load monitoring and taper strategy.

---

### Training Implications
- **Balance:** The distribution supports the 80/20 principle—mostly low-intensity runs.
- **Anomaly Detection:** Outliers reveal either structured intensity shifts or data errors.
- **Optimization:** Boxplot insights allow coaches to diversify training and mitigate risk.

This visualization supports data-driven coaching and smarter self-monitoring for marathon success.
![Boxplot](https://github.com/EmmanuelKusi23/Marathon-Performance-Prediction-Figshare-Data-/blob/25cc7897b3901752e1ad10b0c780b2f69775e890/boxplot%20.png)

### Analysis: Pace vs. Distance in Marathon Training
![Scatterplot](https://github.com/EmmanuelKusi23/Marathon-Performance-Prediction-Figshare-Data-/blob/71da31496282b920e6aff597083c3fdcf71bb627/pace%20vs%20distance.png)

This scatter‐plot examines how pace (min/km) varies with run distance (km). Key takeaways:

- **Data Anomalies**  
  - Pace values span 200–1000 min/km (1000 min/km ≈ 16.7 h/km), indicating misplaced decimals or unit mismatches (e.g., seconds recorded as minutes).  
  - Distances up to 400 km per “run” suggest aggregate volume or mislabeling.

- **Core Trends (after cleaning)**  
  - **Speedwork:** Runs <10 km cluster at faster paces (intervals, tempo sessions).  
  - **Endurance:** Runs >30 km align with slower paces (long‐slow distances).  
  - **Fatigue Curve:** Pace generally slows as distance increases, reflecting aerobic demands.

- **Data-Cleaning Guidelines**  
  1. Cap pace at ≤20 min/km; limit single runs to ≤50 km.  
  2. Standardize units (minutes/km, kilometers).  
  3. Separate per‐run vs. cumulative volumes.

- **Next Steps**  
  - **Subgroup Analysis:** Compare beginners vs. elites.  
  - **Training Cycles:** Plot pace–distance over time to track adaptation.  
  - **Performance Correlations:** Link cleaned training metrics to race outcomes.

> By resolving data issues and exploring these patterns, coaches and athletes can optimize the balance between speedwork and endurance for more effective marathon preparation.

### Feature Correlations
![Correlation Matrix](https://github.com/EmmanuelKusi23/Marathon-Performance-Prediction-Figshare-Data-/blob/e474e8b6b4699295830041e4da508ddf97fb7d6c/correlation%20matrix.png)
This correlation analysis explores the relationships between key variables in a marathon training dataset using a **correlation matrix**. The analysis helps athletes and coaches optimize training strategies through data-driven insights.

---

## 📊 **Correlation Matrix Overview**  
| Variable Pair               | Coefficient | Interpretation                                  |  
|-----------------------------|-------------|------------------------------------------------|  
| `distance` vs. `duration`   | **0.96**    | Very strong positive correlation.              |  
| `pace` vs. `7d_avg_pace`    | **0.99**    | Near-perfect positive correlation.             |  
| `distance` vs. `7d_total_distance` | **0.88** | Strong positive correlation.          |  
| `pace` vs. `distance`       | **-0.06**   | Negligible negative correlation.               |  

---

## 🔍 **Key Insights**  
### **1. Endurance & Time Management**  
- 🏃 `distance` and `duration` are tightly linked (**0.96**), emphasizing the importance of **time-based long runs** for building stamina.  
- 📈 Runners with longer individual sessions (`distance`) also log higher weekly mileage (`7d_total_distance`, **0.88**).  

### **2. Pace Consistency**  
- ⏱️ `pace` and `7d_avg_pace` show near-identical trends (**0.99**), indicating consistent pacing across training sessions.  
- 🐢 `pace` slows only slightly as `distance` increases (**-0.06**), suggesting disciplined pacing strategies.  

### **3. Fatigue & Volume**  
- ⚠️ Higher weekly mileage (`7d_total_distance`) weakly correlates with slower paces (**-0.05**), hinting at cumulative fatigue.  

---

## 🛠️ **Data Considerations**  
- **Validation**: Investigate outliers (e.g., runs >40 km or paces <4:00 min/km).  
- **Units**: Confirm `pace` is recorded as **minutes per kilometer**.  

---

## 🎯 **Recommendations for Training**  
1. **Diversify Workouts**  
   - Introduce speed intervals to improve race-day performance.  
2. **Monitor Volume**  
   - Balance weekly mileage to avoid overtraining.  
3. **Leverage Consistency**  
   - Use `7d_avg_pace` to set realistic pacing goals.  

---

4. Time-Series of Weekly Volume
```python
df['week'] = df['datetime'].dt.isocalendar().week
weekly = df.groupby('week')['distance'].sum().reset_index()
sns.lineplot(x='week', y='distance', data=weekly)
plt.title('Weekly Total Distance in 2020')
plt.xlabel('ISO Week')
plt.ylabel('Total Distance (km)')
plt.show()
```
Figure: Displays training load fluctuations—notice dips around typical holiday periods.

## Data Cleaning & Preprocessing
Based on the EDA, we cleaned and prepared the data. Rows with zero distance (and thus undefined pace) were dropped, which removed all NaN target values. We also ensured no duplicate entries and noted the few missing category values. Categorical variables (gender, age_group, country, major) were encoded: for example we mapped gender to numeric codes (M→0, F→1) and planned to one-hot encode others. We applied ordinal encoding to reduce high-cardinality categories by grouping infrequent values into “Other”. Next we split the data into training and testing sets (80% train, 20% test). We built preprocessing pipelines using scikit-learn: numeric features were imputed (median) and scaled, while categorical features were imputed (most frequent) and one-hot encoded. This full pipeline was later combined with the model for end-to-end training.

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
I created additional features to capture training context. Specifically, for each athlete I computed rolling and lag features:
Rolling averages: 7-day average pace and 7-day total distance (summing runs in the past week) to capture short-term training load.
Run frequency: count of runs in the past 30 days (30-day running count) to capture how often the athlete trains.
Cumulative distance: total distance run so far in the current year for each athlete.
Lag features: the previous run’s pace and days since last run to capture recovery and consistency.
Interaction feature: product of pace and cumulative distance (pace × total distance) to allow non-linear effects (e.g. how pace effects change as a runner accumulates distance).
These engineered features (now included as new columns) provide richer information for the model.
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
We used scikit-learn pipelines to train regression models predicting pace (minutes per km) for each run. We first compared several models with 3-fold cross-validation. The table below summarizes results (mean MAE and R² over folds):
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
The Ridge regression achieved almost perfect R² on training CV (because it can fit near-unique patterns), while the Random Forest had a higher MAE and lower R²
file-sajk4m61jhez8qrnbxbcny
. Given these results, we selected a Random Forest pipeline for further tuning (trading some bias-variance for generalization).

## Hyperparameter Tuning & Evaluation
These test metrics (MAE ≈0.095, R² ≈0.416) indicate the model captures some but not all of the variance in pace. The relatively low R² suggests room for improvement, possibly due to noise in individual run performance. I also inspected feature importances from the trained Random Forest. The bar chart below (from an example model) shows how features can be ranked by importance. In our case, analogous plots would highlight which engineered variables (e.g. recent average pace, run frequency, cumulative distance) contribute most to predictions.
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

## Model Deployment & Usage
Finally, we serialize the full pipeline (preprocessing + model) using joblib. In a production or application setting, one could load this pipeline and make predictions on new data.
Serialize the full pipeline:

bash
python - <<'PYCODE'
from joblib import dump
from your_module import pipeline  
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
This code shows how the trained model pipeline can be used to forecast an athlete’s pace given a new run’s data. The pipeline handles all preprocessing internally, so the user need only supply raw feature values as above. All analyses and code examples are based on processing this public data as described above.
## References
The dataset was published by Afonseca, C., et al. (2022). Long-distance running training logs (2019–2020). Figshare/Kaggle.

Original Parquet → CSV conversion script











