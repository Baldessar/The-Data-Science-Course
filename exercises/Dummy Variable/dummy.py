import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()


rawData = pd.read_csv('real_estate_price_size_year_view.csv')

data = rawData.copy()

data['view'] = data['view'].map({'No sea view': 0, 'Sea view': 1})

y = data['price']
x1 = data[['size', 'year', 'view']]

x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()

newData = pd.DataFrame(
    {'const': 1, 'size': [650.00], 'year': [2019], 'view': [1]})
newData = newData[['const', 'size', 'year', 'view']]

predictions = results.predict(newData)
print(predictions)
