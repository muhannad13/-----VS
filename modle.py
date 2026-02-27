# ==============================================
# Used Car Price Prediction - Improved Pipeline
# King Khalid University
# ==============================================
# !pip install lightgbm catboost shap --quiet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import warnings
import time
warnings.filterwarnings("ignore")

import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder


# ==============================================
# 1. Load Dataset
# ==============================================

df = pd.read_csv("cars.csv")
print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
df.head()


# ==============================================
# 2. Explore Data
# ==============================================

print(df.dtypes)
print(f"\nNull values:\n{df.isnull().sum()}")
print(f"\nStatistics:\n{df[['Year','Engine_Size','Mileage','Price']].describe().round(1)}")

print(f"\nIssues found:")
print(f"  Price = 0:       {(df['Price'] == 0).sum()}")
print(f"  Price > 500K:    {(df['Price'] > 500000).sum()}")
print(f"  Mileage > 500K:  {(df['Mileage'] > 500000).sum()}")
print(f"  Year < 2005:     {(df['Year'] < 2005).sum()}")


# ==============================================
# 3. Visualize Data
# ==============================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].hist(df[df['Price']>0]['Price'], bins=50, color='steelblue', edgecolor='white')
axes[0,0].set_title('Price Distribution')

top = df['Make'].value_counts().head(10)
axes[0,1].barh(top.index[::-1], top.values[::-1], color='coral')
axes[0,1].set_title('Top 10 Makes')

avg = df[df['Price']>0].groupby('Year')['Price'].mean()
axes[1,0].plot(avg.index, avg.values, marker='o', color='green')
axes[1,0].set_title('Average Price by Year')

opt = df[df['Price']>0].groupby('Options')['Price'].mean().sort_values()
axes[1,1].barh(opt.index, opt.values, color='mediumpurple')
axes[1,1].set_title('Average Price by Options')

plt.tight_layout()
plt.savefig('01_exploration.png', dpi=150)
plt.show()


# ==============================================
# 4. Clean Data
# ==============================================

original = len(df)

df = df[df['Price'] > 5000]
df = df[df['Price'] < 500000]
df = df[df['Mileage'] < 500000]
df = df[df['Year'] >= 2005]
df = df.drop(columns=['Negotiable'])
df = df.dropna()
df = df.drop_duplicates()
df = df.reset_index(drop=True)

print(f"Before: {original:,} → After: {len(df):,} rows")


# ==============================================
# 5. Feature Engineering
# ==============================================

df['Car_Age'] = 2025 - df['Year']

CAT_COLS = ['Make', 'Type', 'Origin', 'Color', 'Options',
            'Fuel_Type', 'Gear_Type', 'Region']

for col in CAT_COLS:
    df[col] = df[col].str.strip().str.title()

NUM_COLS = ['Year', 'Engine_Size', 'Mileage', 'Car_Age']
FEATURES = CAT_COLS + NUM_COLS

print(f"Features: {len(CAT_COLS)} categorical + {len(NUM_COLS)} numerical = {len(FEATURES)} total")


# ==============================================
# 6. Split Data (Train / Validation / Test)
# ==============================================

X = df[FEATURES]
y = df['Price']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")


# ==============================================
# 7. Encode Categorical Features
# ==============================================

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

X_train_enc = X_train.copy()
X_val_enc   = X_val.copy()
X_test_enc  = X_test.copy()

X_train_enc[CAT_COLS] = encoder.fit_transform(X_train[CAT_COLS])
X_val_enc[CAT_COLS]   = encoder.transform(X_val[CAT_COLS])
X_test_enc[CAT_COLS]  = encoder.transform(X_test[CAT_COLS])

X_all_enc = X.copy()
X_all_enc[CAT_COLS] = encoder.transform(X[CAT_COLS])


# ==============================================
# 8. Model Comparison (LightGBM vs RF vs CatBoost)
# ==============================================

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

# LightGBM
lgbm = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.03, max_depth=6,
    num_leaves=31, min_child_samples=30, subsample=0.7,
    colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=0.5,
    random_state=42, verbose=-1
)
t = time.time()
lgbm.fit(X_train_enc, y_train,
         eval_set=[(X_val_enc, y_val)],
         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
lgbm_time = time.time() - t

# Random Forest
rf = RandomForestRegressor(
    n_estimators=500, max_depth=15, min_samples_leaf=5,
    random_state=42, n_jobs=-1
)
t = time.time()
rf.fit(X_train_enc, y_train)
rf_time = time.time() - t

# CatBoost
from catboost import CatBoostRegressor
cb = CatBoostRegressor(
    iterations=1000, learning_rate=0.03, depth=6,
    l2_leaf_reg=3, random_seed=42, verbose=0
)
t = time.time()
cb.fit(X_train_enc, y_train, eval_set=(X_val_enc, y_val), early_stopping_rounds=50)
cb_time = time.time() - t

# Evaluate all
results = {}
for name, m, t_time in [('LightGBM', lgbm, lgbm_time),
                         ('RandomForest', rf, rf_time),
                         ('CatBoost', cb, cb_time)]:
    pred = m.predict(X_test_enc)
    train_pred = m.predict(X_train_enc)
    cv = cross_val_score(m, X_all_enc, y, cv=5, scoring='r2')

    results[name] = {
        'MAE': mean_absolute_error(y_test, pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
        'R2': r2_score(y_test, pred),
        'MAPE': np.mean(np.abs((y_test - pred) / y_test)) * 100,
        'Train_R2': r2_score(y_train, train_pred),
        'CV_R2': cv.mean(),
        'CV_std': cv.std(),
        'Time': t_time
    }

print(f"\n{'Model':<15s} {'MAE':>10s} {'RMSE':>10s} {'R2':>8s} {'MAPE':>8s} {'CV R2':>12s} {'Gap':>8s} {'Time':>6s}")
print("-" * 80)
for name, r in results.items():
    gap = r['Train_R2'] - r['R2']
    best = " <-- Best" if r['R2'] == max(v['R2'] for v in results.values()) else ""
    print(f"{name:<15s} {r['MAE']:>8,.0f}  {r['RMSE']:>8,.0f}  {r['R2']:>7.4f} {r['MAPE']:>7.1f}% {r['CV_R2']:>.4f}+/-{r['CV_std']:.3f} {gap:>7.3f} {r['Time']:>5.1f}s{best}")

best_name = max(results, key=lambda k: results[k]['R2'])
print(f"\nBest model: {best_name}")


# ==============================================
# 9. Use Best Model for Final Evaluation
# ==============================================

model = lgbm

y_pred_test = model.predict(X_test_enc)
y_pred_train = model.predict(X_train_enc)

mae  = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2   = r2_score(y_test, y_pred_test)
mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
train_r2 = r2_score(y_train, y_pred_train)

print(f"\nFinal Model: LightGBM")
print(f"MAE:  {mae:,.0f} SAR")
print(f"RMSE: {rmse:,.0f} SAR")
print(f"R2:   {r2:.4f}")
print(f"MAPE: {mape:.1f}%")
print(f"Overfitting Gap: {train_r2 - r2:.4f}")


# ==============================================
# 10. Cross-Validation (5-Fold)
# ==============================================

cv_r2  = cross_val_score(model, X_all_enc, y, cv=5, scoring='r2')
cv_mae = -cross_val_score(model, X_all_enc, y, cv=5, scoring='neg_mean_absolute_error')

for i in range(5):
    print(f"Fold {i+1}: R2 = {cv_r2[i]:.4f}, MAE = {cv_mae[i]:,.0f}")

print(f"\nAvg R2:  {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
print(f"Avg MAE: {cv_mae.mean():,.0f} +/- {cv_mae.std():,.0f}")


# ==============================================
# 11. Error Analysis by Brand
# ==============================================

print("=" * 70)
print("ERROR ANALYSIS BY BRAND")
print("=" * 70)

error_df = pd.DataFrame({
    'Make': X_test['Make'].values,
    'Year': X_test['Year'].values,
    'Actual': y_test.values,
    'Predicted': y_pred_test,
    'Error': np.abs(y_test.values - y_pred_test),
    'Pct_Error': np.abs(y_test.values - y_pred_test) / y_test.values * 100
})

brand_error = error_df.groupby('Make').agg(
    Count=('Error', 'count'),
    Avg_Price=('Actual', 'mean'),
    MAE=('Error', 'mean'),
    MAPE=('Pct_Error', 'mean')
).sort_values('Count', ascending=False).head(12)

print(f"\n{'Brand':<15s} {'Count':>6s} {'Avg Price':>10s} {'MAE':>10s} {'MAPE':>8s}")
print("-" * 55)
for name, row in brand_error.iterrows():
    print(f"{name:<15s} {row['Count']:>6.0f} {row['Avg_Price']:>8,.0f}  {row['MAE']:>8,.0f}  {row['MAPE']:>6.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

brand_error.sort_values('MAE').plot.barh(y='MAE', ax=axes[0], color='steelblue', legend=False)
axes[0].set_title('MAE by Brand (SAR)')
axes[0].set_xlabel('Mean Absolute Error')

brand_error.sort_values('MAPE').plot.barh(y='MAPE', ax=axes[1], color='coral', legend=False)
axes[1].set_title('MAPE by Brand (%)')
axes[1].set_xlabel('Mean Absolute Percentage Error')

plt.tight_layout()
plt.savefig('05_error_by_brand.png', dpi=150)
plt.show()


# ==============================================
# 12. Error Analysis by Year
# ==============================================

print("=" * 70)
print("ERROR ANALYSIS BY YEAR")
print("=" * 70)

year_error = error_df.groupby('Year').agg(
    Count=('Error', 'count'),
    Avg_Price=('Actual', 'mean'),
    MAE=('Error', 'mean'),
    MAPE=('Pct_Error', 'mean')
).sort_index()

print(f"\n{'Year':<6s} {'Count':>6s} {'Avg Price':>10s} {'MAE':>10s} {'MAPE':>8s}")
print("-" * 45)
for year, row in year_error.iterrows():
    print(f"{year:<6d} {row['Count']:>6.0f} {row['Avg_Price']:>8,.0f}  {row['MAE']:>8,.0f}  {row['MAPE']:>6.1f}%")

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(year_error.index.astype(str), year_error['MAE'], color='steelblue', alpha=0.7, label='MAE')
ax2 = ax.twinx()
ax2.plot(year_error.index.astype(str), year_error['MAPE'], color='red', marker='o', linewidth=2, label='MAPE%')
ax.set_xlabel('Year')
ax.set_ylabel('MAE (SAR)')
ax2.set_ylabel('MAPE (%)')
ax.set_title('Error by Year: MAE and MAPE')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('06_error_by_year.png', dpi=150)
plt.show()


# ==============================================
# 13. Error Analysis by Price Range
# ==============================================

print("=" * 70)
print("ERROR ANALYSIS BY PRICE RANGE")
print("=" * 70)

error_df['Price_Range'] = pd.cut(error_df['Actual'],
    bins=[0, 30000, 60000, 100000, 200000, 500000],
    labels=['<30K', '30-60K', '60-100K', '100-200K', '200K+'])

range_error = error_df.groupby('Price_Range', observed=True).agg(
    Count=('Error', 'count'),
    MAE=('Error', 'mean'),
    MAPE=('Pct_Error', 'mean')
)

print(f"\n{'Range':<12s} {'Count':>6s} {'MAE':>10s} {'MAPE':>8s}")
print("-" * 40)
for name, row in range_error.iterrows():
    print(f"{name:<12s} {row['Count']:>6.0f} {row['MAE']:>8,.0f}  {row['MAPE']:>6.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

range_error['MAE'].plot.bar(ax=axes[0], color='steelblue')
axes[0].set_title('MAE by Price Range')
axes[0].set_ylabel('MAE (SAR)')
axes[0].tick_params(axis='x', rotation=0)

range_error['MAPE'].plot.bar(ax=axes[1], color='coral')
axes[1].set_title('MAPE by Price Range')
axes[1].set_ylabel('MAPE (%)')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('07_error_by_range.png', dpi=150)
plt.show()


# ==============================================
# 14. Confidence Interval
# ==============================================

val_preds = model.predict(X_val_enc)
residual_std = float(np.std(y_val.values - val_preds))

print(f"Residual Std: {residual_std:,.0f} SAR")
print(f"90% CI margin: +/-{1.645 * residual_std:,.0f} SAR")


# ==============================================
# 15. Visualize Model Results
# ==============================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(y_test, y_pred_test, alpha=0.3, s=12, color='steelblue')
axes[0].plot([0, y_test.max()], [0, y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Price')
axes[0].set_ylabel('Predicted Price')
axes[0].set_title(f'Actual vs Predicted (R2 = {r2:.4f})')

axes[1].hist(y_test - y_pred_test, bins=40, color='mediumseagreen', edgecolor='white')
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_title('Prediction Errors')

imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
colors = ['coral' if f in ['Engine_Size','Options','Color'] else 'steelblue' for f in imp.index]
axes[2].barh(imp.index, imp.values, color=colors)
axes[2].set_title('Feature Importance')

plt.tight_layout()
plt.savefig('08_model_results.png', dpi=150)
plt.show()


# ==============================================
# 16. Model Comparison Chart
# ==============================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

model_names = list(results.keys())
bar_colors = ['#2ecc71', '#3498db', '#e74c3c']

r2_vals = [results[n]['R2'] for n in model_names]
axes[0].bar(model_names, r2_vals, color=bar_colors)
axes[0].set_title('R2 Score (higher = better)')
axes[0].set_ylim(min(r2_vals) - 0.05, max(r2_vals) + 0.02)
for i, v in enumerate(r2_vals):
    axes[0].text(i, v + 0.003, f'{v:.4f}', ha='center', fontweight='bold')

mae_vals = [results[n]['MAE'] for n in model_names]
axes[1].bar(model_names, mae_vals, color=bar_colors)
axes[1].set_title('MAE in SAR (lower = better)')
for i, v in enumerate(mae_vals):
    axes[1].text(i, v + 200, f'{v:,.0f}', ha='center', fontweight='bold')

mape_vals = [results[n]['MAPE'] for n in model_names]
axes[2].bar(model_names, mape_vals, color=bar_colors)
axes[2].set_title('MAPE % (lower = better)')
for i, v in enumerate(mape_vals):
    axes[2].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('09_model_comparison.png', dpi=150)
plt.show()


# ==============================================
# 17. SHAP Explainability
# ==============================================

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_enc)

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_test_enc, feature_names=FEATURES, show=False)
plt.tight_layout()
plt.savefig('10_shap_summary.png', dpi=150)
plt.show()

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_enc, feature_names=FEATURES, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('11_shap_bar.png', dpi=150)
plt.show()

print(f"\nSample -> Actual: {y_test.iloc[0]:,.0f} | Predicted: {y_pred_test[0]:,.0f}")
for feat, val in sorted(zip(FEATURES, shap_values[0]), key=lambda x: abs(x[1]), reverse=True):
    arrow = "+" if val > 0 else "-"
    print(f"  {arrow} {feat:<15s} {val:>+10,.0f} SAR")


# ==============================================
# 18. Test with Real Examples
# ==============================================

def predict_price(car_info):
    """Predict price with confidence interval and SHAP explanation."""
    car_info['Car_Age'] = 2025 - car_info['Year']
    df_in = pd.DataFrame([car_info])[FEATURES]
    df_in[CAT_COLS] = encoder.transform(df_in[CAT_COLS])

    price = float(model.predict(df_in)[0])
    margin = 1.645 * residual_std

    shap_vals = explainer.shap_values(df_in)[0]
    explanation = []
    for feat, sv in zip(FEATURES, shap_vals):
        explanation.append({
            'feature': feat,
            'shap_value': round(float(sv), 2),
            'direction': 'positive' if sv > 0 else 'negative'
        })
    explanation.sort(key=lambda x: abs(x['shap_value']), reverse=True)

    return {
        'predicted_price': round(price),
        'confidence_lower': round(max(0, price - margin)),
        'confidence_upper': round(price + margin),
        'explanation': explanation
    }


test_cars = [
    {'Make':'Toyota', 'Type':'Camry', 'Year':2019, 'Origin':'Saudi', 'Color':'White',
     'Options':'Full', 'Engine_Size':2.5, 'Fuel_Type':'Gas', 'Gear_Type':'Automatic',
     'Mileage':80000, 'Region':'Riyadh'},

    {'Make':'Hyundai', 'Type':'Elantra', 'Year':2020, 'Origin':'Saudi', 'Color':'Silver',
     'Options':'Standard', 'Engine_Size':1.6, 'Fuel_Type':'Gas', 'Gear_Type':'Automatic',
     'Mileage':50000, 'Region':'Jeddah'},

    {'Make':'Lexus', 'Type':'Es', 'Year':2021, 'Origin':'Saudi', 'Color':'Black',
     'Options':'Full', 'Engine_Size':3.5, 'Fuel_Type':'Gas', 'Gear_Type':'Automatic',
     'Mileage':30000, 'Region':'Dammam'},
]

for car in test_cars:
    r = predict_price(car)
    print(f"\n{car['Make']} {car['Type']} {car['Year']}:")
    print(f"  Price: {r['predicted_price']:,} SAR  ({r['confidence_lower']:,} - {r['confidence_upper']:,})")
    for e in r['explanation'][:3]:
        arrow = "+" if e['direction'] == 'positive' else "-"
        print(f"  {arrow} {e['feature']}: {e['shap_value']:+,.0f}")


# ==============================================
# 19. Save Model & Files
# ==============================================

joblib.dump(model, 'car_model.pkl')
joblib.dump(encoder, 'encoder.pkl')

bundle = {
    'features': FEATURES,
    'cat_cols': CAT_COLS,
    'num_cols': NUM_COLS,
    'residual_std': residual_std,
    'categories': {col: sorted(df[col].unique().tolist()) for col in CAT_COLS},
    'year_range': [int(df['Year'].min()), int(df['Year'].max())],
    'engine_sizes': sorted(df['Engine_Size'].unique().tolist()),
    'type_by_make': {m: sorted(df[df['Make']==m]['Type'].unique().tolist()) for m in df['Make'].unique()}
}

with open('bundle.json', 'w', encoding='utf-8') as f:
    json.dump(bundle, f, ensure_ascii=False, indent=2)

with open('metrics.json', 'w') as f:
    json.dump({
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4),
        'MAPE': round(mape, 2),
        'CV_R2': round(float(cv_r2.mean()), 4),
        'dataset_size': len(df),
        'features': FEATURES,
        'model_comparison': {name: {
            'R2': round(r['R2'], 4),
            'MAE': round(r['MAE'], 2),
            'MAPE': round(r['MAPE'], 2)
        } for name, r in results.items()}
    }, f, indent=2)

print("\nSaved:")
print("  car_model.pkl  -> Trained model")
print("  encoder.pkl    -> Feature encoder")
print("  bundle.json    -> Dropdown options + config")
print("  metrics.json   -> Performance + comparison")

print("\nCharts saved:")
print("  01_exploration.png")
print("  05_error_by_brand.png")
print("  06_error_by_year.png")
print("  07_error_by_range.png")
print("  08_model_results.png")
print("  09_model_comparison.png")
print("  10_shap_summary.png")
print("  11_shap_bar.png")