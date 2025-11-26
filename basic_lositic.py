from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, mean_squared_error , r2_score 
import matplotlib.pyplot as plt
from lightgbm import early_stopping, log_evaluation

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# === Train LightGBM model ===
model = lgb.LGBMRegressor(
    objective='binary',  # Optional but recommended
    n_estimators=1000,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],

    # 3. Change your metric to 'rmse' as we discussed
    eval_metric='rmse',

    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(100)
    ]
)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n📉 Final Test MSE: {mse:.4f}")
print(f"📈 R² Score: {r2:.4f}")

# === Plot feature importance ===
lgb.plot_importance(model, max_num_features=15)
plt.title("Top 15 Feature Importances")
plt.show()
