from xgboost import XGBRegressor
import pandas as pd

m = XGBRegressor()
m.load_model("./m1.model")

df = pd.DataFrame([[1, 41, 172.0, 74.0, 24.0, 98.0, 40.8]])
df = pd.DataFrame(data=df.values, columns=['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate',
       'Body_Temp'])

test_data_prediction = m.predict(df)
print(test_data_prediction)