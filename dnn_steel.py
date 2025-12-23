# -*- coding: utf-8 -*-
"""
Deep learning model to predict mechanical properties from composition
"""

# ====== 1. IMPORT LIBRARIES ======
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor  

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib.patches import FancyBboxPatch

import matplotlib.pyplot as plt


# ====== 2. LOAD DATASET ======
csv_path = r"D:\Papers & Others\Code File\data_steel.csv"

data = pd.read_csv(csv_path)
print("Columns in dataset:\n", data.columns)
print("\nFirst 5 rows:\n", data.head())


# ====== 3. SELECT FEATURES (X) AND TARGET (Y) ======

feature_cols = ['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'V', 'Nb', 'Ti',
                'B', 'P', 'S', 'Cu', 'N', 'Al']

# Start with single output: Yield Strength
target_col = 'YS'

X = data[feature_cols].copy()
Y = data[[target_col]].copy()   # DataFrame with one column


# ====== 4. TRAIN / TEST SPLIT ======

X_train, X_temp, Y_train, Y_temp = train_test_split(
    X, Y, test_size=0.30, random_state=42
)

X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.50, random_state=42
)

print("\nTrain size:", X_train.shape)
print("Val size:  ", X_val.shape)
print("Test size: ", X_test.shape)


# ====== 5. SCALE FEATURES (MIN-MAX) ======

scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled   = scaler_X.transform(X_val)
X_test_scaled  = scaler_X.transform(X_test)


# ====== 6. HELPER FUNCTION TO EVALUATE MODELS ======

def evaluate_model(name, model, X_tr, Y_tr, X_te, Y_te):
    model.fit(X_tr, Y_tr.values.ravel())
    Y_pred = model.predict(X_te)

    mae = mean_absolute_error(Y_te, Y_pred)
    mse = mean_squared_error(Y_te, Y_pred)
    rmse = np.sqrt(mse)

    r2 = r2_score(Y_te, Y_pred)

    print(f"\n=== {name} ===")
    print("MAE:  ", mae)
    print("RMSE: ", rmse)
    print("R²:   ", r2)

    return mae, rmse, r2


# ====== 7. BASELINE ML MODELS ======

# Linear Regression
lr = LinearRegression()
evaluate_model("Linear Regression", lr,
               X_train_scaled, Y_train, X_test_scaled, Y_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
evaluate_model("Random Forest", rf,
               X_train_scaled, Y_train, X_test_scaled, Y_test)

# XGBoost
xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
evaluate_model("XGBoost", xgb,
               X_train_scaled, Y_train, X_test_scaled, Y_test)

# SVR
svr = SVR(kernel='rbf', C=10, gamma='scale')
evaluate_model("SVR", svr,
               X_train_scaled, Y_train, X_test_scaled, Y_test)


# ====== 8. DEEP NEURAL NETWORK (DNN) ======

input_dim = X_train_scaled.shape[1]

model = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)   # single output: YS
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled, Y_train,
    validation_data=(X_val_scaled, Y_val),
    epochs=500,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)


# ====== 9. PLOT TRAINING VS VALIDATION LOSS ======

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training vs Validation Loss (DNN)')
plt.legend()
plt.show()


# ====== 10. EVALUATE DNN ON TEST SET ======

Y_pred_nn = model.predict(X_test_scaled)

mae_nn = mean_absolute_error(Y_test, Y_pred_nn)

# Old sklearn: no 'squared' argument, so we do RMSE manually
mse_nn = mean_squared_error(Y_test, Y_pred_nn)
rmse_nn = np.sqrt(mse_nn)

r2_nn = r2_score(Y_test, Y_pred_nn)

print("\n=== Deep Neural Network (DNN) – YS ===")
print("MAE:  ", mae_nn)
print("RMSE: ", rmse_nn)
print("R²:   ", r2_nn)



# ====== 11. PARITY PLOT: ACTUAL VS PREDICTED ======

plt.figure()
plt.scatter(Y_test, Y_pred_nn, alpha=0.7)
plt.xlabel('Actual YS (MPa)')
plt.ylabel('Predicted YS (MPa)')
plt.title('Parity Plot: Actual vs Predicted YS (DNN)')

min_val = min(Y_test.min()[0], Y_pred_nn.min())
max_val = max(Y_test.max()[0], Y_pred_nn.max())
plt.plot([min_val, max_val], [min_val, max_val])  # 45° line

plt.show()


# ====== 12. Histogram of YS ======


plt.figure(figsize=(6, 4))
plt.hist(data['YS'], bins=20, edgecolor='black')
plt.xlabel('Yield Strength, YS (MPa)')
plt.ylabel('Frequency')
plt.title('Histogram of Yield Strength (YS)')
plt.tight_layout()
plt.savefig('Fig1_YS_histogram.png', dpi=300)
plt.show()

# ====== 13. Boxplots of Elements ======

plt.figure(figsize=(10, 5))
data[feature_cols].boxplot()
plt.xticks(rotation=45, ha='right')
plt.ylabel('wt.%')
plt.title('Boxplot of Alloying Element Compositions')
plt.tight_layout()
plt.savefig('Fig2_composition_boxplots.png', dpi=300)
plt.show()


# ====== 14. Correlation Heatmap (Elements vs YS) ======

corr_cols = feature_cols + ['YS']
corr_matrix = data[corr_cols].corr()

plt.figure(figsize=(8, 6))
im = plt.imshow(corr_matrix, interpolation='nearest', aspect='auto')
plt.colorbar(im, fraction=0.046, pad=0.04)

plt.xticks(ticks=np.arange(len(corr_cols)), labels=corr_cols, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(corr_cols)), labels=corr_cols)

plt.title('Correlation Matrix of Alloying Elements and Yield Strength')
plt.tight_layout()
plt.savefig('Fig3_correlation_heatmap.png', dpi=300)
plt.show()


# ====== 15. Re-run/Collect R^2 Scores for All Models + DNN ======

lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=200, random_state=42)
xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
svr = SVR(kernel='rbf', C=10, gamma='scale')

r2_scores = {}
mae_scores = {}
rmse_scores = {}

for name, model_obj in [
    ("Linear Regression", lr),
    ("Random Forest", rf),
    ("XGBoost", xgb),
    ("SVR", svr)
]:
    mae, rmse, r2 = evaluate_model(
        name,
        model_obj,
        X_train_scaled, Y_train,
        X_test_scaled, Y_test
    )
    r2_scores[name] = r2
    mae_scores[name] = mae
    rmse_scores[name] = rmse

Y_pred_nn = model.predict(X_test_scaled)


# Computing metrics for DNN
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae_nn = mean_absolute_error(Y_test, Y_pred_nn)
mse_nn = mean_squared_error(Y_test, Y_pred_nn)
rmse_nn = np.sqrt(mse_nn)
r2_nn = r2_score(Y_test, Y_pred_nn)

r2_scores["DNN"] = r2_nn
mae_scores["DNN"] = mae_nn
rmse_scores["DNN"] = rmse_nn

print("\n=== Summary of R² Scores ===")
for k, v in r2_scores.items():
    print(f"{k}: R² = {v:.3f}")


# ====== 16. Bar Chart of R^2 across Models ======

model_names = list(r2_scores.keys())
r2_values = [r2_scores[m] for m in model_names]

plt.figure(figsize=(6, 4))
plt.bar(model_names, r2_values)
plt.ylabel('R² Score')
plt.ylim(0, 1.0)
plt.title('Comparison of R² Across Models')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('Fig4_model_R2_comparison.png', dpi=300)
plt.show()


# ====== 17. Simple DNN Architecture Diagram ======

#Horizontal block diagram (Figure 2): 
#Input (16) -> Dense(128) -> Dense(64) -> Dense(32) -> Dropout(0.2) -> Output(1)

plt.figure(figsize=(8, 3))
ax = plt.gca()
ax.set_xlim(0, 6)
ax.set_ylim(0, 2)
ax.axis('off')

def draw_block(x, y, text, width=0.9, height=0.8):
    box = FancyBboxPatch(
        (x, y),
        width, height,
        boxstyle="round,pad=0.2",
        linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha='center',
        va='center',
        fontsize=9
    )

# x positions for layers
x_positions = [0.2, 1.4, 2.6, 3.8, 5.0, 6.2]
layer_texts = [
    "Input\n(16 features)",
    "Dense\n128 neurons",
    "Dense\n64 neurons",
    "Dense\n32 neurons",
    "Dropout\nrate=0.2",
    "Output\n1 neuron"
]

for x, txt in zip(x_positions, layer_texts):
    draw_block(x, 0.6, txt)

# Draw arrows between layers
for i in range(len(x_positions) - 1):
    x_start = x_positions[i] + 0.9
    x_end = x_positions[i+1]
    y_mid = 0.6 + 0.4
    ax.annotate(
        "",
        xy=(x_end, y_mid),
        xytext=(x_start, y_mid),
        arrowprops=dict(arrowstyle="->", linewidth=1.2)
    )

plt.title("Deep Neural Network Architecture (Yield Strength Model)", fontsize=10)
plt.tight_layout()
plt.savefig('Fig5_DNN_architecture.png', dpi=300)
plt.show()


# ====== 18. Error Distribution Plot (Residuals) ======

# Residuals = Predicted - Actual
y_true = Y_test.values.ravel()
y_pred = Y_pred_nn.ravel()
residuals = y_pred - y_true

plt.figure(figsize=(6, 4))
plt.scatter(y_true, residuals, alpha=0.7)
plt.axhline(0, color='black', linewidth=1)
plt.xlabel('Actual YS (MPa)')
plt.ylabel('Residual (Predicted - Actual) (MPa)')
plt.title('Residual Plot: DNN Predictions vs Actual Yield Strength')
plt.tight_layout()
plt.savefig('Fig6_residuals_plot.png', dpi=300)
plt.show()



# ====== 19. Option A - SHAP Summary Plot (XGBoost) ======
# Requirement: pip install shap

try:
    import shap

    print("SHAP version:", shap.__version__)

    # Training XGBoost model (If not already trained)
    xgb_shap = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_shap.fit(X_train_scaled, Y_train.values.ravel())

    # 2. Choosing subset of test data for SHAP
    X_shap = X_test_scaled  # or X_test_scaled[:100]

    # 3. Creating SHAP explainer
    explainer = shap.TreeExplainer(xgb_shap)
    shap_values = explainer.shap_values(X_shap)

    # 4. SHAP summary plot (Beeswarm) – Feature Importance + Effect Direction
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=feature_cols,
        show=False 
    )
    plt.tight_layout()
    plt.savefig("Fig7_SHAP_summary_XGB.png", dpi=300, bbox_inches='tight')
    plt.show()

except ImportError:
    print("SHAP is not installed. Run: pip install shap")
    print("Skipping SHAP plot. Using permutation importance instead.")


# ====== 20. Option B - Permutation Importance (Random Forest) ======


from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

    # 1. Training Random Forest model for interpretability 
rf_perm = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rf_perm.fit(X_train_scaled, Y_train.values.ravel())

    # 2. Computing Permutation Importance on Test Data
r = permutation_importance(
    rf_perm,
    X_test_scaled,
    Y_test.values.ravel(),
    n_repeats=30,
    random_state=42,
    n_jobs=-1
)

importances = r.importances_mean
std = r.importances_std

    # 3. Sorting Features by Importance
indices = np.argsort(importances)[::-1]
sorted_features = [feature_cols[i] for i in indices]
sorted_importances = importances[indices]
sorted_std = std[indices]

    # 4. Plot Bar Chart of Permutation Importance
plt.figure(figsize=(6, 4))
plt.bar(range(len(sorted_features)), sorted_importances, yerr=sorted_std, capsize=3)
plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha="right")
plt.ylabel("Permutation Importance (mean ΔR²)")
plt.title("Feature Importance from Permutation (Random Forest)")
plt.tight_layout()
plt.savefig("Fig8_permutation_importance_RF.png", dpi=300)
plt.show()