# ------------------------------------------
# Moayad Batwa
# ------------------------------------------

# -----------------------------------------------------------------
# Step 1: Import All Necessary Libraries
# -----------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# -----------------------------------------------------------------
# Step 2: Load and Preprocess the Dataset
# -----------------------------------------------------------------

# --- Load Data ---
# !!! IMPORTANT: Replace this path with the actual path to your dataset !!!
dataset_path = 'UNSW_NB15_dataset.csv' 
data = pd.read_csv(dataset_path)

# --- Inspect and Sample Data ---
print("Original Data Head:")
print(data.head())

# Sample 10% of the data as shown in the file
data = data.sample(frac=0.1, random_state=42)
print(f"\nSampled data shape: {data.shape}")

# --- Data Cleaning ---
# Drop unnecessary columns
data = data.drop(['id', 'attack_cat'], axis=1)

# Fill missing values with 0
data.fillna(0, inplace=True)

# --- Label Encoding (Target Variable) ---
# Encode the 'label' column (Normal=0, Attack=1)
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# --- Feature Encoding (Categorical Features) ---
# One-hot encode categorical features
categorical_columns = ['proto', 'service', 'state']
# We use data.drop('label', axis=1) as the base for X
X = pd.get_dummies(data.drop('label', axis=1), columns=categorical_columns, drop_first=True)

# --- Normalization ---
# Separate the target variable 'y'
y = data['label']

# Normalize all features using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# --- Train-Test Split ---
# Split data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# -----------------------------------------------------------------
# Step 3: Train and Evaluate Machine Learning Models
# -----------------------------------------------------------------
print("\nTraining Machine Learning Models...")

# --- Random Forest ---
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
print("Random Forest:")
print(classification_report(y_test, y_pred_rf))

# --- Support Vector Machine (SVM) ---
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
y_prob_svm = svm.predict_proba(X_test)[:, 1]
print("SVM:")
print(classification_report(y_test, y_pred_svm))

# --- K-Nearest Neighbors (KNN) ---
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
y_prob_knn = knn.predict_proba(X_test)[:, 1]
print("K-Nearest Neighbors:")
print(classification_report(y_test, y_pred_knn))

# -----------------------------------------------------------------
# Step 4: Train and Evaluate Deep Learning Models
# -----------------------------------------------------------------
print("\nTraining Deep Learning Models...")

# --- Simple Neural Network (NN) ---
def build_simple_nn(input_dim):
    model = Sequential()
    model.add(Dense(32, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Setup Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

model_1 = build_simple_nn(X_train.shape[1])
history_1 = model_1.fit(X_train, y_train, epochs=5, batch_size=128, verbose=1, 
                        validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
accuracy_1_list = model_1.evaluate(X_test, y_test, verbose=0)
y_prob_dl_1 = model_1.predict(X_test).ravel()

# Note: evaluate() returns [loss, accuracy], so we use index [1]
accuracy_1 = accuracy_1_list[1] 
print(f"Simple Neural Network Accuracy: {accuracy_1:.4f}")

# --- LSTM Model ---
# Reshape data for LSTM (samples, timesteps, features)
X_train_seq = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_seq = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(32, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model_2 = build_lstm((X_train_seq.shape[1], X_train_seq.shape[2]))
history_2 = model_2.fit(X_train_seq, y_train, epochs=5, batch_size=128, verbose=1, 
                        validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
accuracy_2_list = model_2.evaluate(X_test_seq, y_test, verbose=0)
y_prob_dl_2 = model_2.predict(X_test_seq).ravel()

# Note: Same as NN, we use index [1] for accuracy
accuracy_2 = accuracy_2_list[1]
print(f"LSTM Accuracy: {accuracy_2:.4f}")

# -----------------------------------------------------------------
# Step 5: Model Comparison and Evaluation
# -----------------------------------------------------------------

# --- Model Accuracy Comparison Bar Chart ---
models = ['Random Forest', 'SVM', 'KNN', 'Simple NN', 'LSTM']
# We use the accuracy scores we saved for each model
accuracies = [
    accuracy_score(y_test, y_pred_rf), 
    accuracy_score(y_test, y_pred_svm), 
    accuracy_score(y_test, y_pred_knn), 
    accuracy_1, 
    accuracy_2
]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

# --- Confusion Matrix (for Random Forest) ---
conf_matrix = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --- ROC Curves and AUC ---
# Calculate ROC curve data
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
fpr_dl_1, tpr_dl_1, _ = roc_curve(y_test, y_prob_dl_1)
fpr_dl_2, tpr_dl_2, _ = roc_curve(y_test, y_prob_dl_2)

# Plot all ROC curves
plt.figure(figsize=(10, 6))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc(fpr_rf, tpr_rf):.2f})")
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {auc(fpr_svm, tpr_svm):.2f})")
plt.plot(fpr_knn, tpr_knn, label=f"KNN (AUC = {auc(fpr_knn, tpr_knn):.2f})")
plt.plot(fpr_dl_1, tpr_dl_1, label=f"Simple NN (AUC = {auc(fpr_dl_1, tpr_dl_1):.2f})")
plt.plot(fpr_dl_2, tpr_dl_2, label=f"LSTM (AUC = {auc(fpr_dl_2, tpr_dl_2):.2f})")
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# -----------------------------------------------------------------
# Step 6: Conclusion
# -----------------------------------------------------------------
print("\nAll models trained and evaluated!")