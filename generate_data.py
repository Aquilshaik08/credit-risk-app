import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 1500

# Generate realistic mock data fields
data = {
    'credit_score': np.random.randint(300, 850, n_samples),
    'monthly_income': np.random.randint(2000, 15000, n_samples),
    'existing_debt_payment': np.random.randint(100, 4000, n_samples),
    'loan_amount': np.random.randint(5000, 100000, n_samples),
    'employment_years': np.random.randint(0, 40, n_samples),
    'prior_defaults': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
}

df = pd.DataFrame(data)

# Calculate advanced ratios (Features)
df['dti_ratio'] = df['existing_debt_payment'] / df['monthly_income']

# Generate a structural target field
# Risk increases significantly if credit score is low, DTI is high, or prior defaults exist
risk_score = (
    (850 - df['credit_score']) * 0.4 +
    (df['dti_ratio'] * 100) * 0.4 +
    (df['prior_defaults'] * 150)
)
# Classify default status based on a risk threshold (e.g., risk score > 200)
df['default_status'] = (risk_score > 200).astype(int)

# Save to system storage
df.to_csv('credit_data.csv', index=False)
print("SUCCESS: 'credit_data.csv' created successfully.")
