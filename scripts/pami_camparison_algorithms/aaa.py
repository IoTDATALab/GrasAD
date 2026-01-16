import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('results/femnist_asyn_acc.csv')
x = df['round']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x, df['nbafl'], ls = '--', lw=3)
plt.plot(x, df['adaclip1'], ls = (0,(3,1,1,1)), lw=3)
plt.plot(x, df['fixdps'], ls = ':', lw=3)
plt.plot(x, df['mapas'], ls = '-', lw=3)
plt.show()
