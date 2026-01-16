import pandas as pd
import matplotlib.pyplot as plt

step = 50

fig=plt.figure(figsize=(12,6),dpi=100)
grid = plt.GridSpec(2, 4, wspace=0.3, hspace=0.3)

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
plt.plot(x[0::step], df['avg'][0::step], ls = (0,(3,1,1,1)), lw=3, label='AdaClip1')
plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 alpha=0.5)
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
plt.title('A')
plt.xticks([])
# plt.xticks([0,10000,20000,30000],[0, 1000,2000,3000])

plt.subplot(grid[0,1])
df = pd.read_csv('results/femnist_nbafl.csv')
x = df['Step']
# avg = df.iloc[1:,1:-4].mean(axis=1)
plt.plot(x[0::step], df['avg'][0::step], ls = '--', lw=3, label='NBAFL')
plt.fill_between(x[0::100],df['min'][0::100],df['max'][0::100],
                 alpha=0.5)
df = pd.read_csv('results/femnist_mapas.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x[0::step], df['avg'][0::step], ls = ':', lw=3,label='MAPA-S')
plt.fill_between(x[0::100],df['min'][0::100],df['max'][0::100],
                 alpha=0.5)
df = pd.read_csv('results/femnist_fixdps.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x[0::step], df['avg'][0::step], ls = '-', lw=3, label='FixDP-S')
plt.fill_between(x[0::100],df['min'][0::100],df['max'][0::100],
                 alpha=0.5)
df = pd.read_csv('results/femnist_adaclip1.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x[0::step], df['avg'][0::step], ls = (0,(3,1,1,1)), lw=3, label='AdaClip1')
plt.fill_between(x[0::100],df['min'][0::100],df['max'][0::100],
                 alpha=0.5)

plt.title('B')
# plt.xticks([])
# plt.legend()
plt.ylim([-0.15,4.25])
plt.grid(axis='y')
plt.xticks([])
# plt.xticks([0,25000,50000,750000],[0, 1000,2000,3000])

plt.subplot(grid[1,0])
df = pd.read_csv('results/femnist_asyn_acc.csv')
x = df['round']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x, df['nbafl'], ls = '--', lw=3)
plt.plot(x, df['adaclip1'], ls = (0,(3,1,1,1)), lw=3)
plt.plot(x, df['fixdps'], ls = ':', lw=3)
plt.plot(x, df['mapas'], ls = '-', lw=3)
# plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 # color='lightsteelblue',alpha=0.6)
# plt.xticks([])
plt.legend(['NBAFL','AdaClip1','FixDP-S','MAPA-S'], loc=4)
# plt.ylim([-0.15,4.25])
plt.title('A\'')
plt.xlabel('FEMNIST with K=1')
plt.ylabel('Test Accuracy', fontsize=12)
plt.grid(axis='both')

plt.subplot(grid[1,1])
df = pd.read_csv('results/femnist_algorithms_acc.csv')
x = df['round']
# avg = df.iloc[1:,1:-1].mean(axis=1)
print(len(df['adaclip1']))
plt.plot(x, df['nbafl'], ls = '--', lw=3)
plt.plot(x, df['mapas'], ls = ':', lw=3)
plt.plot(x, df['fixdps'], ls = '-', lw=3)
plt.plot(x, df['adaclip1'], ls = (0,(3,1,1,1)), lw=3)

# plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 # color='lightsteelblue',alpha=0.6)
# plt.title('FEMNIST (K=3)')
# plt.xticks([])
# plt.legend(['NBAFL','AdaClip1','FixDP-S','MAPA-S'])
# plt.ylim([-0.15,4.25])
plt.title('B\'')
plt.xlabel('FEMNIST with K=3')
plt.grid(axis='both')

plt.subplot(grid[0,2])
df = pd.read_csv('results/shakes_silo_asyn_loss.csv')
x = df['Step']
plt.plot(x[0::step], df['nbafl-avg'][0::step], ls = '--', lw=3, label='NBAFL')
plt.fill_between(x[0::step],df['nbafl-min'][0::step],df['nbafl-max'][0::step],
                 alpha=0.5)
plt.plot(x[0::step], df['adaclip1-avg'][0::step], ls = (0,(3,1,1,1)), lw=3, label='AdaClip1')
plt.fill_between(x[0::step],df['adaclip1-min'][0::step],df['adaclip1-max'][0::step],
                 alpha=0.5)
plt.plot(x[0::step], df['fixdps-avg'][0::step], ls = ':', lw=3, label='FixDP-S')
plt.fill_between(x[0::step],df['fixdps-min'][0::step],df['fixdps-max'][0::step],
                 alpha=0.5)
plt.plot(x[0::step], df['mapas-avg'][0::step], ls = '-', lw=3, label='MAPA-S')
plt.fill_between(x[0::step],df['mapas-min'][0::step],df['mapas-max'][0::step],
                 alpha=0.5)
# plt.legend()
# plt.ylabel('Test Average Loss', fontsize=12)
plt.title('C')
# plt.ylim([-0.15,4.25])
plt.grid(axis='y')
plt.xticks([])

plt.subplot(grid[0,3])
step = 30
df = pd.read_csv('results/shakes_silo_loss.csv')
x = df['Step']
plt.plot(x[0::step], df['nbafl-avg'][0::step], ls = '--', lw=3, label='NBAFL')
plt.fill_between(x[0::step],df['nbafl-min'][0::step],df['nbafl-max'][0::step],
                 alpha=0.5)
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x[0::step], df['adaclip1-avg'][0::step], ls = (0,(3,1,1,1)), lw=3, label='AdaClip1')
plt.fill_between(x[0::step],df['adaclip1-min'][0::step],df['adaclip1-max'][0::step],
                 alpha=0.5)

plt.plot(x[0::step], df['fixdps-avg'][0::step], ls = ':', lw=3,label='FixDP-S')
plt.fill_between(x[0::step],df['fixdps-min'][0::step],df['fixdps-max'][0::step],
                 alpha=0.5)

plt.plot(x[0::step], df['mapas-avg'][0::step], ls = '-', lw=3, label='MAPA')
plt.fill_between(x[0::step],df['mapas-min'][0::step],df['mapas-max'][0::step],
                 alpha=0.5)
plt.title('D')
# plt.xticks([])
# plt.legend()
# plt.ylim([-0.15,4.25])
plt.grid(axis='y')
plt.xticks([])

step = 50

plt.subplot(grid[1,2])
df = pd.read_csv('results/shakes_silo_asyn.csv')
x = df['round']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x, df['nbafl'], ls = '--', lw=3)
plt.plot(x, df['adaclip1'], ls = (0,(3,1,1,1)), lw=3)
plt.plot(x, df['fixdps'], ls = ':', lw=3)
plt.plot(x, df['mapas'], ls = '-', lw=3)
# plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 # color='lightsteelblue',alpha=0.6)
# plt.xticks([])
# plt.legend(['NBAFL','AdaClip1','FixDP-S','MAPA-S'], loc=4)
# plt.ylim([-0.15,4.25])
# plt.ylabel('Test Accuracy', fontsize=12)
plt.grid(axis='both')
plt.title('C\'')
plt.xlabel('Shakespeare with K=1')

plt.subplot(grid[1,3])
df = pd.read_csv('results/shakes_silo.csv')
x = df['round']
plt.plot(x, df['nbafl'], ls = '--', lw=3)
plt.plot(x, df['adaclip1'], ls = (0,(3,1,1,1)), lw=3)
plt.plot(x, df['fixdps'], ls = ':', lw=3)
plt.plot(x, df['mapas'], ls = '-', lw=3)
# plt.legend(['NBAFL','AdaClip1','FixDP-S','MAPA-S'], loc=4)
# plt.ylim([-0.15,4.25])
plt.grid(axis='both')
plt.title('D\'')
plt.xlabel('Shakespeare with K=3')

plt.savefig('cross_silo.pdf', dpi=1020)
plt.show()