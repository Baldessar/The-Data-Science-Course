import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

raw_data = pd.read_csv("2.01. Admittance.csv")

data = raw_data.copy()
data["Admitted"] = data["Admitted"].map({"Yes": 1, "No": 0})

y = data["Admitted"]
x1 = data["SAT"]

x = sm.add_constant(x1)
reg_log = sm.Logit(y, x)
result_log = reg_log.fit()
print(result_log.summary())
