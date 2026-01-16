import pandas as pd
import matplotlib.pyplot as plt

step = 25

fig=plt.figure(figsize=(10,5),dpi=100)
grid = plt.GridSpec(2, 4, wspace=0.3, hspace=0.1)

plt.subplot(grid[0,0])
df = pd.read_csv('results/femnist_sens_k.csv')
x = df['round']
# avg = df.iloc[1:,1:-4].mean(axis=1)
plt.plot(x, df['k8'], ls = '--', lw=2)
plt.plot(x, df['k16'], ls = '-.', lw=2)
plt.plot(x, df['k24'], ls = ':', lw=2)
plt.plot(x, df['k32'], ls = '-', lw=2)
plt.legend(['K=8','K=16','K=24','K=32'])
plt.grid(axis='y')
plt.xticks([])
plt.ylabel('Test Accuracy', fontsize=11)
plt.title('Aggregator Number $K$')

plt.subplot(grid[1,0])
min = df[['k8','k16','k24','k32']].min(axis=1)
max = df[['k8','k16','k24','k32']].max(axis=1)
mean = df[['k8','k16','k24','k32']].mean(axis=1)
plt.plot(x,mean)
plt.fill_between(x, min, max,
                 color='lightsteelblue',alpha=0.6)
plt.legend(['Average','Min/Max Area'])
plt.grid(axis='y')
plt.ylabel('Test Accuracy')
# plt.title('Aggregator Number K')
# plt.ylim([-0.15,4.25])
# plt.grid(axis='y')

plt.subplot(grid[0,1])
df = pd.read_csv('results/femnist_sens_batch.csv')
x = df['round']
# avg = df.iloc[1:,1:-4].mean(axis=1)
plt.plot(x, df['batch8'], ls = '--', lw=2)
plt.plot(x, df['batch16'], ls = '-.', lw=2)
plt.plot(x, df['batch32'], ls = ':', lw=2)
plt.plot(x, df['batch64'], ls = '-', lw=2)
plt.legend(['b=8','b=16','b=32','b=64'])
plt.grid(axis='y')
plt.xticks([])
# plt.ylabel('Test Accuracy', fontsize=11)
plt.title('Batch Size $b$')

plt.subplot(grid[1,1])
min = df[['batch8','batch16','batch32','batch64']].min(axis=1)
max = df[['batch8','batch16','batch32','batch64']].max(axis=1)
mean = df[['batch8','batch16','batch32','batch64']].mean(axis=1)
plt.plot(x,mean)
plt.fill_between(x, min, max,
                 color='lightsteelblue',alpha=0.6)
# plt.legend(['Average Accuracy','Min/Max Area'])
plt.grid(axis='y')
# plt.title('Batch size $b$')
# plt.ylim([-0.15,4.25])
# plt.grid(axis='y')

plt.subplot(grid[0,2])
df = pd.read_csv('results/femnist_sens_L.csv')
x = df['round']
# avg = df.iloc[1:,1:-4].mean(axis=1)
plt.plot(x, df['L-0001'], ls = '--', lw=2)
plt.plot(x, df['L-001'], ls = '-.', lw=2)
plt.plot(x, df['L-01'], ls = ':', lw=2)
plt.plot(x, df['L-1'], ls = '-', lw=2)
plt.legend(['L=$10^{-4}$','L=$10^{-3}$','L=$10^{-2}$','L=$10^{-1}$'])
plt.grid(axis='y')
plt.xticks([])
# plt.ylabel('Test Accuracy', fontsize=11)
plt.title('Gradient Smooth $L$')

plt.subplot(grid[1,2])
min = df[['L-0001','L-001','L-01','L-1']].min(axis=1)
max = df[['L-0001','L-001','L-01','L-1']].max(axis=1)
mean = df[['L-0001','L-001','L-01','L-1']].mean(axis=1)
plt.plot(x,mean)
plt.fill_between(x, min, max,
                 color='lightsteelblue',alpha=0.6)
# plt.legend(['Average Accuracy','Min/Max Area'])
plt.grid(axis='y')
# plt.title('Batch size $b$')
# plt.ylim([-0.15,4.25])
# plt.grid(axis='y')

plt.subplot(grid[0,3])
df = pd.read_csv('results/femnist_sens_C.csv')
x = df['round']
# avg = df.iloc[1:,1:-4].mean(axis=1)
plt.plot(x, df['C10'], ls = '--', lw=2)
plt.plot(x, df['C20'], ls = '-.', lw=2)
plt.plot(x, df['C30'], ls = ':', lw=2)
plt.plot(x, df['C40'], ls = '-', lw=2)
plt.legend(['c=10','c=20','c=30','c=40'])
plt.grid(axis='y')
plt.xticks([])
# plt.ylabel('Test Accuracy', fontsize=11)
plt.title('Initial ClIpbound $c$')

plt.subplot(grid[1,3])
min = df[['C10','C20','C30','C40']].min(axis=1)
max = df[['C10','C20','C30','C40']].max(axis=1)
mean = df[['C10','C20','C30','C40']].mean(axis=1)
plt.plot(x,mean)
plt.fill_between(x, min, max,
                 color='lightsteelblue',alpha=0.6)
# plt.legend(['Average Accuracy','Min/Max Area'])
plt.grid(axis='y')
# plt.title('Batch size $b$')
# plt.ylim([-0.15,4.25])
# plt.grid(axis='y')

plt.savefig('para_sens_femnist.pdf', dpi=1020)

plt.show()

