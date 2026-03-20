import pandas as pd
from sklearn.ensemble import IsolationForest

# fake data (for now)
df = pd.DataFrame({
    "amount": [100, 200, 3000, 50, 60, 10000]
})

model = IsolationForest(contamination=0.2)
df['anomaly'] = model.fit_predict(df[['amount']])

print(df)