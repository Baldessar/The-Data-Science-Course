import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn
seaborn.set()

raw_data = pd.read_csv('1.03. Dummies.csv')


data = raw_data.copy()

data['Attendance'] = data['Attendance'].map({'Yes': 1,'No': 0})

y = data['GPA']
x1 = data[["SAT", "Attendance"]]

x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
print(results.summary())
