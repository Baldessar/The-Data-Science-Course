import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn
seaborn.set()

data = pd.read_csv('real_estate_price_size_year.csv')

y = data['price']

x1 = data[['size', 'year']]

x = sm.add_constant(x1)
result = sm.OLS(y, x).fit()
print(result.summary())
