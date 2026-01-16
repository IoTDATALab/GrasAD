import pandas as pd
import matplotlib.pyplot as plt

step = 50

fig=plt.figure(figsize=(8,6),dpi=100)
grid = plt.GridSpec(2, 2, wspace=0.3, hspace=0.3)

plt.subplot(grid[0,0])
df = pd.read_csv('results/femnist_asyn_nbafl.csv')
x = df['Step']
# avg = df.iloc[1:,1:-4].mean(axis=1)
plt.plot(x[0::step], df['avg'][0::step], ls = '--', lw=3, label='NBAFL')
plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 alpha=0.5)
# plt.title('NBAFL')
df = pd.read_csv('results/femnist_asyn_adaclip1.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x[0::step], df['avg'][0::step], ls = '-.', lw=3, label='AdaClip1')
plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 alpha=0.5)
plt.title('FEMNIST (K=1)')
df = pd.read_csv('results/femnist_asyn_fixdps.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x[0::step], df['avg'][0::step], ls = ':', lw=3, label='FixDP-S')
plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 alpha=0.5)
# plt.title('FixDP-S')
df = pd.read_csv('results/femnist_asyn_mapas.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x[0::step], df['avg'][0::step], ls = '-', lw=3, label='MAPA-S')
plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 alpha=0.5)
# plt.title('MAPA-S')
plt.ylabel('Test Average Loss', fontsize=12)
plt.legend()
plt.ylim([-0.15,4.25])
plt.grid(axis='y')

plt.subplot(grid[0,1])
df = pd.read_csv('results/femnist_nbafl.csv')
x = df['Step']
# avg = df.iloc[1:,1:-4].mean(axis=1)
plt.plot(x[0::step], df['avg'][0::step], ls = '--', lw=3, label='NBAFL')
plt.fill_between(x[0::100],df['min'][0::100],df['max'][0::100],
                 alpha=0.5)
df = pd.read_csv('results/femnist_adaclip1.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x[0::step], df['avg'][0::step], ls = '-.', lw=3, label='AdaClip1')
plt.fill_between(x[0::100],df['min'][0::100],df['max'][0::100],
                 alpha=0.5)
df = pd.read_csv('results/femnist_mapas.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x[0::step], df['avg'][0::step], ls = ':', lw=3,label='FixDP-S')
plt.fill_between(x[0::100],df['min'][0::100],df['max'][0::100],
                 alpha=0.5)
df = pd.read_csv('results/femnist_fixdps.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x[0::step], df['avg'][0::step], ls = '-', lw=3, label='MAPA')
plt.fill_between(x[0::100],df['min'][0::100],df['max'][0::100],
                 alpha=0.5)
plt.title('FEMNIST (K=3)')
# plt.xticks([])
plt.legend()
plt.ylim([-0.15,4.25])
plt.grid(axis='y')

plt.subplot(grid[1,0])
df = pd.read_csv('results/femnist_asyn_acc.csv')
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
df = pd.read_csv('results/femnist_algorithms_acc.csv')
x = df['round']
# avg = df.iloc[1:,1:-1].mean(axis=1)
print(len(df['adaclip1']))
plt.plot(x, df['nbafl'], ls = '--', lw=3)
plt.plot(x, df['adaclip1'], ls = '-.', lw=3)
plt.plot(x, df['mapas'], ls = ':', lw=3)
plt.plot(x, df['fixdps'], ls = '-', lw=3)
# plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 # color='lightsteelblue',alpha=0.6)
# plt.title('FEMNIST (K=3)')
# plt.xticks([])
plt.legend(['NBAFL','AdaClip1','FixDP-S','MAPA-S'])
# plt.ylim([-0.15,4.25])
plt.grid(axis='both')

plt.show()