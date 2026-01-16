import pandas as pd
import matplotlib.pyplot as plt

step = 25

fig=plt.figure(figsize=(14,10),dpi=100)
grid = plt.GridSpec(4, 5, wspace=0.3, hspace=0.3)

plt.subplot(grid[0,0])
df = pd.read_csv('results/femnist_nbafl.csv')
x = df['Step']
# avg = df.iloc[1:,1:-4].mean(axis=1)
plt.plot(x[0::step], df['avg'][0::step], ls = '-', lw=2)
plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 alpha=0.6)
df = pd.read_csv('results/femnist_adaclip1.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x[0::step], df['avg'][0::step], ls = '-', lw=2)
plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 color='lightsteelblue',alpha=0.6)

plt.ylabel('FEMNIST (K=3)', fontsize=11)
plt.xticks([])
plt.legend(['Avg Loss'])
plt.ylim([-0.15,4.25])
plt.grid(axis='y')

plt.subplot(grid[0,1])
df = pd.read_csv('results/femnist_adaclip1.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x[0::step], df['avg'][0::step], ls = '-', lw=2)
plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 color='lightsteelblue',alpha=0.6)
plt.title('AdaClip1')
plt.xticks([])
plt.legend(['Avg Loss'])
plt.ylim([-0.15,4.25])
plt.grid(axis='y')

plt.subplot(grid[0,3])
df = pd.read_csv('results/femnist_fixdps.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x[0::step], df['avg'][0::step], ls = '-', lw=2)
plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 color='lightsteelblue',alpha=0.6)
plt.title('MAPA-S')
plt.xticks([])
plt.legend(['Avg Loss'])
plt.ylim([-0.15,4.25])
plt.grid(axis='y')

plt.subplot(grid[0,2])
df = pd.read_csv('results/femnist_mapas.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x[0::step], df['avg'][0::step], ls = '-', lw=2)
plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 color='lightsteelblue',alpha=0.6)
plt.title('FixDP-S')
plt.xticks([])
plt.legend(['Avg Loss'])
plt.ylim([-0.15,4.25])
plt.grid(axis='y')

plt.subplot(grid[0,4])
df = pd.read_csv('results/femnist_algorithms.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
print(len(df['adaclip1']))
plt.plot(x, df['nbafl'], ls = '-', lw=2)
plt.plot(x, df['adaclip1'], ls = '-', lw=2)
plt.plot(x, df['mapas'], ls = '-', lw=2)
plt.plot(x, df['fixdps'], ls = '-', lw=2)

# plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 # color='lightsteelblue',alpha=0.6)
plt.title('Accuracy')
plt.xticks([])
plt.legend(['NBAFL','AdaClip1','FixDP-S','MAPA-S'])
# plt.ylim([-0.15,4.25])
# plt.grid(axis='y')

plt.subplot(grid[1,0])
df = pd.read_csv('results/femnist_asyn_nbafl.csv')
x = df['Step']
# avg = df.iloc[1:,1:-4].mean(axis=1)
plt.plot(x[0::step], df['avg'][0::step], ls = '-', lw=2)
plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 color='lightsteelblue',alpha=0.6)
# plt.title('NBAFL')
plt.ylabel('FEMNIST (K=1)', fontsize=11)
plt.xticks([])
plt.legend(['Avg Loss'])
plt.ylim([-0.15,4.25])
plt.grid(axis='y')

plt.subplot(grid[1,1])
df = pd.read_csv('results/femnist_asyn_adaclip1.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x[0::step], df['avg'][0::step], ls = '-', lw=2)
plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 color='lightsteelblue',alpha=0.6)
# plt.title('AdaClip1')
plt.xticks([])
plt.legend(['Avg Loss'])
plt.ylim([-0.15,4.25])
plt.grid(axis='y')

plt.subplot(grid[1,2])
df = pd.read_csv('results/femnist_asyn_fixdps.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x[0::step], df['avg'][0::step], ls = '-', lw=2)
plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 color='lightsteelblue',alpha=0.6)
# plt.title('FixDP-S')
plt.xticks([])
plt.legend(['Avg Loss'])
plt.ylim([-0.15,4.25])
plt.grid(axis='y')

plt.subplot(grid[1,3])
df = pd.read_csv('results/femnist_asyn_mapas.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x[0::step], df['avg'][0::step], ls = '-', lw=2)
plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 color='lightsteelblue',alpha=0.6)
# plt.title('MAPA-S')
plt.xticks([])
plt.legend(['Avg Loss'])
plt.ylim([-0.15,4.25])
plt.grid(axis='y')

plt.subplot(grid[1,4])
df = pd.read_csv('results/femnist_asyn_algorithms.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x, df['nbafl'], ls = '-', lw=2)
plt.plot(x, df['adaclip1'], ls = '-', lw=2)
plt.plot(x, df['fixdps'], ls = '-', lw=2)
plt.plot(x, df['mapas'], ls = '-', lw=2)
# plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 # color='lightsteelblue',alpha=0.6)
plt.xticks([])
plt.legend(['NBAFL','AdaClip1','FixDP-S','MAPA-S'], loc=4)
# plt.ylim([-0.15,4.25])
# plt.grid(axis='y')

plt.subplot(grid[2,0])
df = pd.read_csv('results/femnist_asyn_algorithms.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x, df['nbafl'], ls = '-', lw=2)
plt.plot(x, df['adaclip1'], ls = '-', lw=2)
plt.plot(x, df['fixdps'], ls = '-', lw=2)
plt.plot(x, df['mapas'], ls = '-', lw=2)
# plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 # color='lightsteelblue',alpha=0.6)
plt.xticks([])
plt.legend(['Avg Loss'])
# plt.ylim([-0.15,4.25])
# plt.grid(axis='y')

plt.subplot(grid[2,1])
df = pd.read_csv('results/femnist_asyn_algorithms.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x, df['nbafl'], ls = '-', lw=2)
plt.plot(x, df['adaclip1'], ls = '-', lw=2)
plt.plot(x, df['fixdps'], ls = '-', lw=2)
plt.plot(x, df['mapas'], ls = '-', lw=2)
# plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 # color='lightsteelblue',alpha=0.6)
plt.xticks([])
plt.legend(['Avg Loss'])
# plt.ylim([-0.15,4.25])
# plt.grid(axis='y')

plt.subplot(grid[2,2])
df = pd.read_csv('results/femnist_asyn_algorithms.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x, df['nbafl'], ls = '-', lw=2)
plt.plot(x, df['adaclip1'], ls = '-', lw=2)
plt.plot(x, df['fixdps'], ls = '-', lw=2)
plt.plot(x, df['mapas'], ls = '-', lw=2)
# plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 # color='lightsteelblue',alpha=0.6)
plt.xticks([])
plt.legend(['Avg Loss'])
# plt.ylim([-0.15,4.25])
# plt.grid(axis='y')

plt.subplot(grid[2,3])
df = pd.read_csv('results/femnist_asyn_algorithms.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x, df['nbafl'], ls = '-', lw=2)
plt.plot(x, df['adaclip1'], ls = '-', lw=2)
plt.plot(x, df['fixdps'], ls = '-', lw=2)
plt.plot(x, df['mapas'], ls = '-', lw=2)
# plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 # color='lightsteelblue',alpha=0.6)
plt.xticks([])
plt.legend(['Avg Loss'])
# plt.ylim([-0.15,4.25])
# plt.grid(axis='y')

plt.subplot(grid[3,1])
df = pd.read_csv('results/femnist_asyn_algorithms.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x, df['nbafl'], ls = '-', lw=2)
plt.plot(x, df['adaclip1'], ls = '-', lw=2)
plt.plot(x, df['fixdps'], ls = '-', lw=2)
plt.plot(x, df['mapas'], ls = '-', lw=2)
# plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 # color='lightsteelblue',alpha=0.6)
plt.xticks([])
plt.legend(['Avg Loss'])
# plt.ylim([-0.15,4.25])
# plt.grid(axis='y')

plt.subplot(grid[3,2])
df = pd.read_csv('results/femnist_asyn_algorithms.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x, df['nbafl'], ls = '-', lw=2)
plt.plot(x, df['adaclip1'], ls = '-', lw=2)
plt.plot(x, df['fixdps'], ls = '-', lw=2)
plt.plot(x, df['mapas'], ls = '-', lw=2)
# plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 # color='lightsteelblue',alpha=0.6)
plt.xticks([])
plt.legend(['Avg Loss'])
# plt.ylim([-0.15,4.25])
# plt.grid(axis='y')

plt.subplot(grid[3,3])
df = pd.read_csv('results/femnist_asyn_algorithms.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x, df['nbafl'], ls = '-', lw=2)
plt.plot(x, df['adaclip1'], ls = '-', lw=2)
plt.plot(x, df['fixdps'], ls = '-', lw=2)
plt.plot(x, df['mapas'], ls = '-', lw=2)
# plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 # color='lightsteelblue',alpha=0.6)
plt.xticks([])
plt.legend(['Avg Loss'])
# plt.ylim([-0.15,4.25])
# plt.grid(axis='y')

plt.subplot(grid[3,4])
df = pd.read_csv('results/femnist_asyn_algorithms.csv')
x = df['Step']
# avg = df.iloc[1:,1:-1].mean(axis=1)
plt.plot(x, df['nbafl'], ls = '-', lw=2)
plt.plot(x, df['adaclip1'], ls = '-', lw=2)
plt.plot(x, df['fixdps'], ls = '-', lw=2)
plt.plot(x, df['mapas'], ls = '-', lw=2)
# plt.fill_between(x[0::step],df['min'][0::step],df['max'][0::step],
                 # color='lightsteelblue',alpha=0.6)
plt.xticks([])
plt.legend(['Avg Loss'])
# plt.ylim([-0.15,4.25])
# plt.grid(axis='y')

plt.show()