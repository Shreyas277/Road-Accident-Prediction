import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np

# --- Load and Prep Data ---
# Note: Replace with your actual file paths
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    print("Data loaded successfully from Kaggle path.")
except FileNotFoundError:
    print("Could not find Kaggle files, trying local paths 'train.csv' and 'test.csv'...")
    try:
        train_df = pd.read_csv("train.csv")
        test_df = pd.read_csv("test.csv")
        print("Data loaded successfully from local paths.")
    except FileNotFoundError:
        print("Error: Could not find train.csv or test.csv. Please check file paths.")
        exit()


target_column = "accident_risk"
id_column = "id"

# --- MODIFICATION: Remove 'public_road' from the feature list ---
base_features = train_df.drop(columns=[id_column, target_column]).columns
feature_columns = [col for col in base_features if col != 'public_road']
print(f"Dropping 'public_road'. Using {len(feature_columns)} base features for training.")

# --- Create your X, y, and X_test ---
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
# START: NEW FEATURE ENGINEERING STEP
# -----------------------------------------------------------------
print("Creating custom features based on research findings...")

def create_all_new_features(df):
    """
    Applies all feature engineering steps to a dataframe.
    This function encapsulates the logic from the research report.
    """
    # --- Feature 1: Light & Weather Danger Score (Original Feature) ---
    # This feature combines lighting and weather conditions into a ranked danger score.
    light_col = 'lighting'
    weather_col = 'weather'
    val_night = 'night'
    val_dim = 'dim'
    val_day = 'daylight'
    val_foggy = 'foggy'
    val_rainy = 'rainy'
    val_clear = 'clear'

    # --- CHANGED: Values are now floats
    danger_map = {
        f"{val_night}_{val_foggy}": 8.0,
        f"{val_night}_{val_rainy}": 7.0,
        f"{val_dim}_{val_foggy}": 6.0,
        f"{val_dim}_{val_rainy}": 5.0,
        f"{val_day}_{val_foggy}": 4.0,
        f"{val_day}_{val_rainy}": 3.0,
        f"{val_night}_{val_clear}": 2.0,
        f"{val_dim}_{val_clear}": 1.0,
        f"{val_day}_{val_clear}": 0.0
    }
    combo = df[light_col].astype(str) + "_" + df[weather_col].astype(str)
    # --- CHANGED: fillna(0.0) to ensure float type
    df['light_weather_danger_score'] = combo.map(danger_map).fillna(0.0)

    # --- Feature 3: Temporal Risk Score ---
    temporal_score = pd.Series(0.0, index=df.index) # Already float
    # --- CHANGED: Values in map are now floats
    temporal_score += df['time_of_day'].map({'morning': 1.0, 'afternoon': 0.0, 'evening': 2.0, 'night': 3.0}).fillna(0.0)
    # --- CHANGED: np.where values are now floats
    temporal_score += np.where(df['holiday'] == 1, 3.0, 0.0)
    is_school_commute = (df['school_season'] == 1) & (df['time_of_day'].isin(['morning', 'afternoon']))
    # --- CHANGED: np.where values are now floats
    temporal_score += np.where(is_school_commute, 1.0, 0.0)
    temporal_score += df['num_reported_accidents']*0.8 # Already float
    temporal_score += 0.7*((df['road_type'] == 'rural') & (df['lighting'] == 'night')).astype(int) # Becomes float

    # --- CHANGED: Initialized as float
    design_score = pd.Series(0.0, index=df.index)
    # --- CHANGED: np.where values are now floats
    design_score += np.where((df['road_type'] == 'urban') & (df['num_lanes'] > 2), 1.0, 0.0)
    design_score -= np.where((df['road_type'] == 'highway') & (df['num_lanes'] > 3), 1.0, 0.0)
    design_score += np.where((df['road_type'] == 'rural') & (df['num_lanes'] == 1), 1.0, 0.0)
    
    df['temporal_risk_score'] = temporal_score + design_score
    
    # --- Feature 5: How Bad Road ---
    bad_road_score = pd.Series(0.0, index=df.index) 
    bad_road_score += df['curvature']*2
    bad_road_score -= np.where(df['road_signs_present'] == 0, 1.0, 0.5) 
    road_type_risk = {'rural': 1.5, 'urban': 0.91, 'highway': 0.5}
    bad_road_score += df['road_type'].map(road_type_risk).fillna(0.0)
    
    # --- FIX: Use np.exp() for Series, not math.exp() ---
    df['how_bad_road'] = np.exp(bad_road_score)
    
    # --- Feature 6: NEW - Speed + How Bad Road Interaction ---
    # Models the idea that speed on a "bad" road (high curve, no signs, rural) is very risky.
    #df['speed_road_interaction'] = df['speed_limit'] * df['how_bad_road']

    # --- Feature 7: NEW - Speed + Environmental Danger Interaction ---
    # Models the idea that high speed in bad weather/light is very risky.
    #df['speed_env_interaction'] = df['speed_limit'] * df['light_weather_danger_score']
    
    return df

# Apply the feature engineering function to both training and test sets
X = create_all_new_features(X)
X_test = create_all_new_features(X_test)

print("New features created. Sample:")
# --- MODIFIED: Updated list of new features ---
new_features_list = [
    'light_weather_danger_score', 
    'temporal_risk_score', 
    'how_bad_road'
]
print(X[new_features_list].head())

# --- NEW: Save the processed test data with all features ---
print("\nSaving processed test data with all features to d.csv...")
X_test.to_csv("d.csv", index=False)
print("✅ d.csv saved successfully.")
# --- END NEW ---

# -----------------------------------------------------------------
# END: NEW FEATURE ENGINEERING STEP
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# MODEL 1: LIGHTGBM
# -----------------------------------------------------------------
X_lgb = X.copy()
X_test_lgb = X_test.copy()
# Convert original categorical features to 'category' dtype for LightGBM
for col in categorical_features:
    X_lgb[col] = X_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

print("\nTraining LightGBM...")
lgb_model = lgb.LGBMRegressor(
    objective='regression_l1', # Mean Absolute Error
    metric='rmse',
    n_estimators=2000,
    learning_rate=0.01,
    random_state=42,
    n_jobs=-1,
    colsample_bytree=0.7,
    subsample=0.7,
    reg_alpha=0.1,
    reg_lambda=0.1
)

# We still pass the *original* categorical features list
lgb_model.fit(X_lgb, y, categorical_feature=list(categorical_features))
lgb_preds = lgb_model.predict(X_test_lgb)


# -----------------------------------------------------------------
# MODEL 2: XGBOOST
# -----------------------------------------------------------------
print("\nPreparing data for XGBoost...")
# pd.get_dummies will correctly ignore new numeric features
# and only encode the original categorical features.
X_xgb = pd.get_dummies(X, columns=categorical_features, drop_first=True)
X_test_xgb = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

# Align columns - this is critical if one-hot encoding creates
# different sets of columns (e.g., a category is missing in test)
X_xgb, X_test_xgb = X_xgb.align(X_test_xgb, join='left', axis=1, fill_value=0)

print("Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', # Root Mean Squared Error
    eval_metric='rmse',
    n_estimators=1000,
    learning_row=0.01,
    enable_categorical=False, # We are using one-hot encoding
    random_state=42,
    n_jobs=-1,
    colsample_bytree=0.8,
    subsample=0.8
)
xgb_model.fit(X_xgb, y)
xgb_preds = xgb_model.predict(X_test_xgb)


# -----------------------------------------------------------------
# FINAL STEP: Blend the Predictions
# -----------------------------------------------------------------
print("\nBlending predictions...")
# Simple 70/30 blend
final_predictions = 0.7 * lgb_preds + 0.3 * xgb_preds

# --- Create Submission File ---
submission = pd.DataFrame({
    "id": test_df[id_column],
    "accident_risk": final_predictions
})
# Ensure predictions are within the valid range [0, 1]
submission['accident_risk'] = submission['accident_risk'].clip(0, 1)
submission.to_csv("submission_blend.csv", index=False)
print("\n✅ submission_blend.csv created successfully!")
