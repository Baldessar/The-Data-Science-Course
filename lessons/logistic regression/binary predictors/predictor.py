import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn
seaborn.set()

from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

raw_data = pd.read_csv("2.02. Binary predictors.csv")

data = raw_data.copy()

data['Admitted'] = data['Admitted'].map({"Yes":1,"No":0})
data['Gender'] = data['Gender'].map({"Male":0,"Female":1})

y = data['Admitted']
x1 = data['Gender']

x =sm.add_constant(x1)
req_log = sm.Logit(y,x)
result_log = req_log.fit()
summary =result_log.summary()
print(summary)


