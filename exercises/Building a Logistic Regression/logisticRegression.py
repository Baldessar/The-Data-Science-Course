import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

raw_data = pd.read_csv("Example-bank-data.csv")

data = raw_data.copy()

data = data.drop(['Unnamed: 0'], axis=1)

data['y'] = data['y'].map({'yes': 1, 'no': 0})

print(data.describe())
y = data['y']
x1 = data['duration']

x = sm.add_constant(x1)

reg_log = sm.Logit(y, x)
results_log = reg_log.fit()


# Create a scatter plot of x1 (Duration, no constant) and y (Subscribed)
plt.scatter(x1, y, color='C0')

# Don't forget to label your axes!
plt.xlabel('Duration', fontsize=20)
plt.ylabel('Subscription', fontsize=20)
plt.show()

print(results_log.summary())
