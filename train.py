# %%
import pickle

import kagglehub
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%
# Download dataset
path = kagglehub.dataset_download("fedesoriano/company-bankruptcy-prediction")
print("Path to dataset files:", path)

# %%
# Load data
train_data = pd.read_csv(
    "/home/yesavage/.cache/kagglehub/datasets/fedesoriano/company-bankruptcy-prediction/versions/2/data.csv"
)


def clean_col(col):
    return (
        col.strip()  # removes hidden leading spaces
        .replace(" ", "_")
        .replace("%", "percent")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("¥", "Yuan")
    )


train_data.columns = [clean_col(c) for c in train_data.columns]

print(train_data.columns)
print(train_data.shape)

# %%
# Split features and target
X = train_data.drop("Bankrupt?", axis=1)
y = train_data["Bankrupt?"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
# Scale full dataset for RFE
scaler_rfe = StandardScaler()
X_train_scaled_full = scaler_rfe.fit_transform(X_train)

# %%
# Feature selection using RFE
model = LogisticRegression(max_iter=500, class_weight="balanced")
rfe = RFE(model, n_features_to_select=10)
rfe.fit(X_train_scaled_full, y_train)

selected_mask = rfe.support_
selected_features = X.columns[selected_mask]

print("Selected features:")
print(selected_features)

# %%
# Select only chosen features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# %%
# Final scaling
final_scaler = StandardScaler()
X_train_scaled = final_scaler.fit_transform(X_train_selected)
X_test_scaled = final_scaler.transform(X_test_selected)

# %%
# Train final model
final_model = LogisticRegression(max_iter=2000, class_weight="balanced")
final_model.fit(X_train_scaled, y_train)

# %%
# Predictions
y_pred = final_model.predict(X_test_scaled)
y_prob = final_model.predict_proba(X_test_scaled)[:, 1]

# %%
# Evaluation
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Log Loss:", log_loss(y_test, y_prob))

# %%
pickle.dump(final_model, open("model.pkl", "wb"))
pickle.dump(final_scaler, open("scaler.pkl", "wb"))
pickle.dump(selected_features.tolist(), open("selected_features.pkl", "wb"))

print("Model, scaler, and selected features saved successfully!")
