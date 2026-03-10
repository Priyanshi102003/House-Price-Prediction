import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv('house_data.csv')

# Seprating Input and Target Feature
X = data[['Size','Bedrooms','Age']]
y = data['Price']

# Creating Linear Regression model
model = LinearRegression()
model.fit(X,y)


joblib.dump(model,'house_model.pkl')
print('Model Trained SuccessFully and Saved as house_model.pkl ')