import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# 1. Load the generated historical data
df = pd.read_csv('credit_data.csv')

# 2. Separate independent features and target
X = df[['credit_score', 'monthly_income','existing_debt_payment','loan_amount','employment_years','prior_deaults','dti_ratio']]
y = df['default_status']

# 3. Stratified split to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train the advanced XGBoost model
model = XGBClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 5. Export the trained model to file
with open('credit_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("SUCCESS: Machine learning model trained and saved!")
