import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split

# --- Load and Prep Data ---
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

target_column = "accident_risk"
id_column = "id"
feature_columns = train_df.drop(columns=[id_column, target_column]).columns

# --- Create your X, y, and X_test ---
# (Using .copy() to avoid the warnings)
X = train_df[feature_columns].copy()
y = train_df[target_column]
X_test = test_df[feature_columns].copy()

# Find feature types
categorical_features = X.select_dtypes(include="object").columns
boolean_features = X.select_dtypes(include="boolean").columns

# --- Preprocessing for BOTH models ---
for col in boolean_features:
    X[col] = X[col].astype(int)
    X_test[col] = X_test[col].astype(int)

# -----------------------------------------------------------------
# MODEL 1: LIGHTGBM (The way you already did it)
# -----------------------------------------------------------------
# For LGBM, we need copies of the data where categories are 'category' dtype
X_lgb = X.copy()
X_test_lgb = X_test.copy()
for col in categorical_features:
    X_lgb[col] = X_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

print("Training LightGBM...")
lgb_model = lgb.LGBMRegressor(
    objective='binary',
    metric='rmse',
    n_estimators=2000,
    learning_rate=0.01,
    random_state=42
)
# Note: You would normally use a validation set and early stopping
lgb_model.fit(X_lgb, y, categorical_feature=list(categorical_features))
lgb_preds = lgb_model.predict(X_test_lgb)


# -----------------------------------------------------------------
# MODEL 2: XGBOOST (The new model)
# -----------------------------------------------------------------
# For XGBoost, we need copies of the data that are One-Hot Encoded
print("Preparing data for XGBoost...")
X_xgb = pd.get_dummies(X, columns=categorical_features, drop_first=True)
X_test_xgb = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

# Align columns - pd.get_dummies() might create different columns if test/train differ
X_xgb, X_test_xgb = X_xgb.align(X_test_xgb, join='left', axis=1, fill_value=0)

print("Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    objective='binary:logistic', # Predicts probability (like LGBM's 'binary')
    eval_metric='rmse',          # Judge on RMSE
    n_estimators=1000,
    learning_rate=0.01,
    enable_categorical=False,    # We set this to False, as we did OHE
    random_state=42
)
# Note: You would also use a validation set and early stopping here
xgb_model.fit(X_xgb, y)
xgb_preds = xgb_model.predict(X_test_xgb)


# -----------------------------------------------------------------
# FINAL STEP: Blend the Predictions
# -----------------------------------------------------------------
print("Blending predictions...")
# Start with a simple 50/50 average
final_predictions = 0.5 * lgb_preds + 0.5 * xgb_preds

# --- Create Submission File ---
submission = pd.DataFrame({
    "id": test_df[id_column],
    "accident_risk": final_predictions
})
submission['accident_risk'] = submission['accident_risk'].clip(0, 1)
submission.to_csv("submission_blend.csv", index=False)
print("✅ submission_blend.csv created successfully!")
