import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
# We can override the default matplotlib styles with those of Seaborn
sns.set()

rawData = pd.read_csv('1.03. Dummies.csv')

data = rawData.copy()
data['Attendance'] = data['Attendance'].map({'Yes': 1, 'No': 0})

y = data['GPA']
x1 = data[['SAT', 'Attendance']]

x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()

plt.scatter(data['SAT'], data['GPA'], c=data['Attendance'], cmap='RdYlGn_r')

yhat_no = 0.6439 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']

fig = plt.plot(data['SAT'], yhat_no, lw=2, c='#006837')
fig = plt.plot(data['SAT'], yhat_yes, lw=2, c='#a50026')

plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.show()

# print(x)
new_data = pd.DataFrame(
    {'const': 1, 'SAT': [1700, 1670], 'Attendance': [0, 1]})

new_data = new_data[['const', 'SAT', 'Attendance']]


new_data.rename(index={0: 'Bob', 1: 'Alice'})
predictions = results.predict(new_data)


predictionsdf = pd.DataFrame({'Predictions': predictions})
joined = new_data.join(predictionsdf)
joined.rename(index={0: 'Bob', 1: 'Alice'})

# print(result.summary())

print(predictions)
