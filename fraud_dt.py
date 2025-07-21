# fraud_detection_app.py
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

st.title(" Fraud Detection in Financial Transactions")

uploaded_file = st.file_uploader("Upload your transaction CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Date conversion
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    df['PreviousTransactionDate'] = pd.to_datetime(df['PreviousTransactionDate'])
    df['TimeSinceLastTransaction'] = (
        df['TransactionDate'] - df['PreviousTransactionDate']
    ).dt.total_seconds()

    # Prepare features
    df_model = df.drop(columns=[
        'TransactionID', 'AccountID', 'DeviceID', 'IP Address',
        'MerchantID', 'TransactionDate', 'PreviousTransactionDate'
    ])
    label_encoders = {}
    for col in df_model.select_dtypes(include='object'):
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le
    df_model.fillna(0, inplace=True)

    # Normalize
    scaler = MinMaxScaler()
    score_features = ['TransactionAmount', 'LoginAttempts', 'TransactionDuration', 'TimeSinceLastTransaction']
    df_model[score_features] = scaler.fit_transform(df_model[score_features])

    # Risk Score
    df_model['RiskScore'] = (
        0.4 * df_model['TransactionAmount'] +
        0.2 * df_model['LoginAttempts'] +
        0.2 * df_model['TransactionDuration'] +
        0.2 * df_model['TimeSinceLastTransaction']
    ) * 100

    # Isolation Forest
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    df_model['anomaly'] = model.fit_predict(df_model)
    df_model['is_suspicious'] = df_model['anomaly'].apply(lambda x: 1 if x == -1 else 0)

    # Alert Level
    def generate_alert(row):
        if row['is_suspicious'] == 1 and row['RiskScore'] > 70:
            return "HIGH RISK"
        elif row['RiskScore'] > 50:
            return "REVIEW"
        else:
            return "SAFE"

    df_model['AlertLevel'] = df_model.apply(generate_alert, axis=1)
    df['RiskScore'] = df_model['RiskScore']
    df['AlertLevel'] = df_model['AlertLevel']

    # Plots
    st.subheader(" Alert Level Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='AlertLevel', data=df_model, palette='Set2', ax=ax1)
    ax1.set_title('Count of Transactions by Alert Level')
    st.pyplot(fig1)

    st.subheader(" Correlation Heatmap")
    fig2, ax2 = plt.subplots()
    corr = df_model[score_features + ['RiskScore']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True, ax=ax2)
    st.pyplot(fig2)

    # Download
    st.subheader("⬇ Download Processed Results")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "processed_fraud_results.csv","text/csv")
