import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import tsfel
import os

# Load the dataset (Assuming dataset is available in 'UCI_HAR_Dataset/')
X_train_raw = np.loadtxt('UCI_HAR_Dataset/train/Inertial Signals/total_acc_x_train.txt')
X_test_raw = np.loadtxt('UCI_HAR_Dataset/test/Inertial Signals/total_acc_x_test.txt')
y_train = np.loadtxt('UCI_HAR_Dataset/train/y_train.txt')
y_test = np.loadtxt('UCI_HAR_Dataset/test/y_test.txt')

# Reshape for Deep Learning models
X_train_dl = X_train_raw.reshape(X_train_raw.shape[0], X_train_raw.shape[1], 1)
X_test_dl = X_test_raw.reshape(X_test_raw.shape[0], X_test_raw.shape[1], 1)

# LSTM Model
def create_lstm_model():
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train_dl.shape[1], 1)),
        LSTM(32),
        Dense(32, activation='relu'),
        Dense(len(np.unique(y_train)), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

lstm_model = create_lstm_model()
lstm_model.fit(X_train_dl, y_train, epochs=10, batch_size=32, validation_data=(X_test_dl, y_test))

# 1D CNN Model
def create_cnn_model():
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_dl.shape[1], 1)),
        MaxPooling1D(2),
        Conv1D(32, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(len(np.unique(y_train)), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

cnn_model = create_cnn_model()
cnn_model.fit(X_train_dl, y_train, epochs=10, batch_size=32, validation_data=(X_test_dl, y_test))

# Feature Extraction using TSFEL
cfg = tsfel.get_features_by_domain()
X_train_feat = tsfel.time_series_features_extractor(cfg, X_train_raw, fs=50)
X_test_feat = tsfel.time_series_features_extractor(cfg, X_test_raw, fs=50)

# Train ML models
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_feat, y_train)
y_pred_rf = rf.predict(X_test_feat)

svm = SVC()
svm.fit(X_train_feat, y_train)
y_pred_svm = svm.predict(X_test_feat)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_feat, y_train)
y_pred_lr = log_reg.predict(X_test_feat)

# Evaluate ML models
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

# Compare with original dataset features
X_train_orig = pd.read_csv('UCI_HAR_Dataset/train/X_train.txt', delim_whitespace=True, header=None)
X_test_orig = pd.read_csv('UCI_HAR_Dataset/test/X_test.txt', delim_whitespace=True, header=None)

rf.fit(X_train_orig, y_train)
y_pred_rf_orig = rf.predict(X_test_orig)
print("Random Forest Accuracy (Original Features):", accuracy_score(y_test, y_pred_rf_orig))
