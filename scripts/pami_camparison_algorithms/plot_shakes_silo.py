import pandas as pd
import matplotlib.pyplot as plt

step = 25

fig=plt.figure(figsize=(8,6),dpi=100)
grid = plt.GridSpec(2, 2, wspace=0.3, hspace=0.3)


plt.subplot(grid[0,0])
df = pd.read_csv('results/shakes_silo_asyn_loss.csv')
x = df['Step']
plt.plot(x[0::step], df['nbafl-avg'][0::step], ls = '--', lw=3, label='NBAFL')
plt.fill_between(x[0::step],df['nbafl-min'][0::step],df['nbafl-max'][0::step],
                 alpha=0.5)
plt.plot(x[0::step], df['adaclip1-avg'][0::step], ls = '-.', lw=3, label='AdaClip1')
plt.fill_between(x[0::step],df['adaclip1-min'][0::step],df['adaclip1-max'][0::step],
                 alpha=0.5)
plt.plot(x[0::step], df['fixdps-avg'][0::step], ls = ':', lw=3, label='FixDP-S')
plt.fill_between(x[0::step],df['fixdps-min'][0::step],df['fixdps-max'][0::step],
                 alpha=0.5)
plt.plot(x[0::step], df['mapas-avg'][0::step], ls = '-', lw=3, label='MAPA-S')
plt.fill_between(x[0::step],df['mapas-min'][0::step],df['mapas-max'][0::step],
                 alpha=0.5)
plt.legend()
plt.ylabel('Test Average Loss', fontsize=12)
plt.title('Shakespeare (K=1)')
# plt.ylim([-0.15,4.25])
plt.grid(axis='y')

plt.subplot(grid[0,1])
df = pd.read_csv('results/shakes_silo_loss.csv')
x = df['Step']
plt.plot(x[0::step], df['nbafl-avg'][0::step], ls = '--', lw=3, label='NBAFL')
plt.fill_between(x[0::step],df['nbafl-min'][0::step],df['nbafl-max'][0::step],
                 alpha=0.5)
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x[0::step], df['adaclip1-avg'][0::step], ls = '-.', lw=3, label='AdaClip1')
plt.fill_between(x[0::step],df['adaclip1-min'][0::step],df['adaclip1-max'][0::step],
                 alpha=0.5)

plt.plot(x[0::step], df['fixdps-avg'][0::step], ls = ':', lw=3,label='FixDP-S')
plt.fill_between(x[0::step],df['fixdps-min'][0::step],df['fixdps-max'][0::step],
                 alpha=0.5)

plt.plot(x[0::step], df['mapas-avg'][0::step], ls = '-', lw=3, label='MAPA')
plt.fill_between(x[0::step],df['mapas-min'][0::step],df['mapas-max'][0::step],
                 alpha=0.5)
plt.title('Shakespeare (K=3)')
# plt.xticks([])
plt.legend()
# plt.ylim([-0.15,4.25])
plt.grid(axis='y')

plt.subplot(grid[1,0])
df = pd.read_csv('results/shakes_silo_asyn.csv')
x = df['round']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x, df['nbafl'], ls = '--', lw=3)
plt.plot(x, df['adaclip1'], ls = '-.', lw=3)
plt.plot(x, df['fixdps'], ls = ':', lw=3)
plt.plot(x, df['mapas'], ls = '-', lw=3)
# plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 # color='lightsteelblue',alpha=0.6)
# plt.xticks([])
plt.legend(['NBAFL','AdaClip1','FixDP-S','MAPA-S'], loc=4)
# plt.ylim([-0.15,4.25])
plt.ylabel('Test Accuracy', fontsize=12)
plt.grid(axis='both')

plt.subplot(grid[1,1])
df = pd.read_csv('results/shakes_silo.csv')
x = df['round']
plt.plot(x, df['nbafl'], ls = '--', lw=3)
plt.plot(x, df['adaclip1'], ls = '-.', lw=3)
plt.plot(x, df['fixdps'], ls = ':', lw=3)
plt.plot(x, df['mapas'], ls = '-', lw=3)
plt.legend(['NBAFL','AdaClip1','FixDP-S','MAPA-S'], loc=4)
# plt.ylim([-0.15,4.25])
plt.grid(axis='both')

plt.show()