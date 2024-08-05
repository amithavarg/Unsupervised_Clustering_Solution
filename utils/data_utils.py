import pandas as pd

def load_data(file_path):
    print("Loading data from:", file_path)
    return pd.read_csv(file_path)

def preprocess_data(df):
    print("Preprocessing data")
    # Define default fill values for columns
    fill_values = {
        'Gender': 'Male',
        'Married': df['Married'].mode()[0] if 'Married' in df.columns else None,
        'Dependents': df['Dependents'].mode()[0] if 'Dependents' in df.columns else None,
        'Self_Employed': df['Self_Employed'].mode()[0] if 'Self_Employed' in df.columns else None,
        'LoanAmount': df['LoanAmount'].median() if 'LoanAmount' in df.columns else None,
        'Loan_Amount_Term': df['Loan_Amount_Term'].mode()[0] if 'Loan_Amount_Term' in df.columns else None,
        'Credit_History': df['Credit_History'].mode()[0] if 'Credit_History' in df.columns else None
    }

    # Fill missing values if the column exists
    for col, value in fill_values.items():
        if col in df.columns and value is not None:
            df[col] = df[col].fillna(value)

    return df
