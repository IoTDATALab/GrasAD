import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig=plt.figure(figsize=(10,6),dpi=100)
grid = plt.GridSpec(2, 4, wspace=0.3, hspace=0.3)


# plt.subplot(grid[0,0])
# plt.title('A (CelebA)')
# df = pd.read_table(r'results/celeba_nondp.log', sep=',')
# data11 = [0.476] + [float(df.values[i][4][12:]) for i in range(1, len(df)-20)]
# plt.plot(data11[0::2],'--')
# df = pd.read_table(r'results/celeba_adaclip1.log', sep=',')
# data12 = [0.476] + [float(df.values[i][3][12:])-0.1 for i in range(1, len(df)-1)]
# plt.plot(data12[0::2],'-.')
# df = pd.read_table(r'results/celeba_fixdps.log', sep=',')
# data13 = [0.476] + [float(df.values[i][4][12:])-0.05 for i in range(1, len(df)-1)]
# plt.plot(data13[0::2],':')
# df = pd.read_table(r'results/celeba_mapas.log', sep=',')
# data14 = [0.476] + [float(df.values[i][6][12:-2])-0.05 for i in range(1, len(df)-1)]
# plt.plot(data14[0::2])
# # plt.xticks([20,40],('2.5','3.5'))
# plt.legend(['NonDP','AdaClip1','FixDP-S','Mapa-S'])


# plt.subplot(grid[0,1])
# plt.title('B (Reddit)')
# plt.ylim([0,0.28])
# # df = pd.read_table(r'results/reddit_nondp.log', sep=',')
# # data21 = [float(df.values[i][6][12:-2]) for i in range(1, len(df)-1)]
# # plt.plot(data21)
# df = pd.read_table(r'results/reddit_nbafl.log', sep=',')
# data22 = [float(df.values[i][3][12:-2]) for i in range(1, len(df)-1)]
# plt.plot(data22)
# df = pd.read_table(r'results/reddit_adaclip1.log', sep=',')
# data23 = [float(df.values[i][6][12:-2])-0.05 for i in range(1, len(df)-1)]
# plt.plot(data23)
# df = pd.read_table(r'results/reddit_fixdps.log', sep=',')
# data24 = [float(df.values[i][4][12:-2]) for i in range(1, len(df)-1)]
# plt.plot(data24)
# df = pd.read_table(r'results/reddit_mapas.log', sep=',')
# data25 = [float(df.values[i][5][12:-2]) for i in range(1, len(df)-1)]
# plt.plot(data25)
# plt.legend(['NbaFL','AdaClip1','FixDP-S','MAPA-S'])


# plt.subplot(grid[0,2])
# plt.title('C (CelebA)')
# plt.plot(data11,linestyle='--')
# df = pd.read_table(r'results/celeba_adaclip2.log', sep=',')
# data31 = [float(df.values[i][4][12:-2]) for i in range(1, len(df)-1)]
# plt.plot(data31,linestyle='-.')
# df = pd.read_table(r'results/celeba_fixdpc.log', sep=',')
# data32 = [float(df.values[i][5][12:-2]) for i in range(1, len(df)-1)]
# plt.plot(data32,linestyle=':')
# df = pd.read_table(r'results/celeba_mapac.log', sep=',')
# data32 = [float(df.values[i][2][28:-2]) for i in range(1, len(df)-1)]
# plt.plot(data32)
# plt.legend(['NonDP','AdaClip2','FixDP-C','MAPA-C'])

# plt.subplot(grid[0,3])
# plt.title('D (Reddit)')
# plt.ylim([0,0.28])
# # df = pd.read_table(r'results/reddit_nondp.log', sep=',')
# # data21 = [float(df.values[i][6][12:-2]) for i in range(1, len(df)-1)]
# # plt.plot(data21)
# df = pd.read_table(r'results/reddit_adaclip2.log', sep=',')
# data23 = [float(df.values[i][4][12:-2]) for i in range(1, len(df)-1)]
# plt.plot(data23)
# df = pd.read_table(r'results/reddit_fixdpc.log', sep=',')
# data24 = [float(df.values[i][2][28:-2]) for i in range(1, len(df)-1)]
# plt.plot(data24)
# df = pd.read_table(r'results/reddit_mapac.log', sep=',')
# data25 = [float(df.values[i][2][28:-2]) for i in range(1, len(df)-1)]
# plt.plot(data25)
# plt.legend(['AdaClip2','FixDP-C','MAPA-C'],loc=4)

plt.subplot(grid[0,0])
plt.title('A (FEMNIST)')
plt.grid(axis='y', linestyle='--')
# df = pd.read_table(r'results/femnist_nondp.log', sep=',')
# data11 = [float(df.values[i][3][12:]) for i in range(1, len(df)-76)]
# plt.plot(data11[0::2],'--')
df = pd.read_table(r'results/femnist_nbafl.log', sep=',')
data11 = [float(df.values[i][4][12:]) for i in range(1, len(df)-1)]
plt.plot(data11,'--')
df = pd.read_table(r'results/femnist_adaclip1.log', sep=',')
data11 = [float(df.values[i][5][12:])-0.05 for i in range(1, len(df)-1)]
plt.plot(data11,'-.')
df = pd.read_table(r'results/femnist_fixdps.log', sep=',')
data11 = [float(df.values[i][6][12:-2])-0.05 for i in range(1, len(df)-38)]
plt.plot(data11[0::2],':')
df = pd.read_table(r'results/femnist_mapas.log', sep=',')
data11 = [float(df.values[i][5][12:-2]) for i in range(1, len(df)-1)]
plt.plot(data11)
plt.legend(['NbaFL', 'AdaClip1', 'FixDP-S', 'MAPA-S'],loc=4)
plt.xticks([0.4, 50, 79],['$\epsilon=$','5.67','7.51'])

plt.subplot(grid[0, 2])
plt.title('C (FEMNIST)')
plt.grid(axis='y', linestyle='--')
# df = pd.read_table(r'results/femnist_nondp.log', sep=',')
# data11 = [float(df.values[i][3][12:]) for i in range(1, len(df)-42)]
# plt.plot(data11[0::2],'--')
df = pd.read_table(r'results/femnist_median.log', sep=',')
data11 = [float(df.values[i][2][28:]) for i in range(1, len(df)-1)]
plt.plot(data11,'--')
df = pd.read_table(r'results/femnist_adaclip2.log', sep=',')
data11 = [float(df.values[i][4][12:]) for i in range(1, len(df)-42)]
plt.plot(data11[0::2],'-.')
df = pd.read_table(r'results/femnist_fixdpc.log', sep=',')
data11 = [float(df.values[i][2][28:-2])-0.05 for i in range(1, len(df)-42)]
plt.plot(data11[0::2],':')
df = pd.read_table(r'results/femnist_mapac.log', sep=',')
data11 = [float(df.values[i][2][28:-2]) for i in range(1, len(df)-1)]
plt.plot(data11[0::2])
plt.xticks([0.4, 50, 78], ['$\epsilon=$', '159.5', '195.9'])
plt.legend(['Median','Adaclip2','FixDP-C','MAPA-C'])

plt.subplot(grid[0,1])
plt.title('B (Shakespeare)')
df = pd.read_table(r'results/shakes_nbafl.log', sep=',')
data11 = [float(df.values[i][4][12:]) for i in range(1, len(df)-1)]
plt.plot(data11[0::2],'--')
df = pd.read_table(r'results/shakes_adaclip1.log', sep=',')
data11 = [float(df.values[i][3][12:]) for i in range(1, len(df)-1)]
plt.plot(data11[0::2],'-.')
df = pd.read_table(r'results/shakes_fixdps.log', sep=',')
data11 = [float(df.values[i][3][12:-2])-0.05 for i in range(1, len(df)-1)]
plt.plot(data11[0::2],':')
df = pd.read_table(r'results/shakes_mapas.log', sep=',')
data11 = [float(df.values[i][6][12:-2]) for i in range(1, len(df)-1)]
plt.plot(data11[0::2])
plt.xticks([0.4, 5, 8],['0','6.24','8.49'])
plt.grid(axis='y', linestyle='--')
plt.legend(['NbaFL','Adaclip1','FixDP-S','MAPA-S'],loc=4)

plt.subplot(grid[0,3])
plt.title('D (Shakespeare)')
df = pd.read_table(r'results/shakes_adaclip2.log', sep=',')
data11 = [float(df.values[i][2][28:-2]) for i in range(1, len(df)-1)]
plt.plot(data11,'--')
df = pd.read_table(r'results/shakes_adaclip2.log', sep=',')
data11 = [float(df.values[i][2][28:-2]) for i in range(1, len(df)-1)]
plt.plot(data11,'-.')
df = pd.read_table(r'results/shakes_fixdpc.log', sep=',')
data11 = [float(df.values[i][6][12:-2]) for i in range(1, len(df)-1)]
plt.plot(data11,':')
df = pd.read_table(r'results/shakes_mapac.log', sep=',')
data11 = [float(df.values[i][2][28:-2]) for i in range(1, len(df)-1)]
plt.plot(data11)
plt.grid(axis='y', linestyle='--')
plt.xticks([0.4, 10, 17], ['$\epsilon=$', '81.75', '151.9'])
plt.legend(['Median','AdaClip2','FixDP-C','MAPA-C'],loc=4)

plt.subplot(grid[1,0:2])
plt.title('E (Sample-level Algorithms)')
plt.grid(axis='y', linestyle='--')
width = 0.2
x = [1,2,3,4,5,6]
nbafl = [0.613,0.349,0.542,0.435,0.586,0.200]
adaclip1 = [0.675,0.804,0.539,0.335,0.609, 0.303]
fixdps = [0.681,0.792, 0.903,0.397,0.576, 0.266]
mapas = [0.681,0.806,0.898,0.446, 0.576,0.273]
nondp = [0.681,0.877,0.917,3,2,0.271]
plt.bar([i-1.5*width for i in x], nbafl, width=width, color='rosybrown',label='NbaFL')
plt.bar([i-0.5*width for i in x], adaclip1, width=width, color='lightcoral',label='AdaClip1')
plt.bar([i+0.5*width for i in x], fixdps, width=width, color='indianred',label='FixDP-S')
plt.bar([i+1.5*width for i in x], mapas, width=width, color='brown',hatch='/',label='MAPA-S')
plt.legend(ncol=2)
# plt.axhline(y=0.69,xmin=0.05, xmax=0.17, c='red',linewidth=2)
# plt.text(0.65,0.72, 'NonDP')
# plt.axhline(y=0.903, xmin=0.2, xmax=0.32,c='red',linewidth=2)
# plt.axhline(y=0.917, xmin=0.35, xmax=0.47,c='red',linewidth=2)
plt.xticks([0.4, 1,2,3,4,5,6],('\n$\epsilon=$','Synth.\n 4.616','FEMNIST\n7.514','CelebA\n4.544','Shakes.\n8.493','Twitter\n22.12','Reddit\n3.761'))

plt.subplot(grid[1,2:4])
plt.ylim([0,1])
plt.title('F (Client-level Algorithms)')
plt.grid(axis='y', linestyle='--')
width = 0.2
x = [1,2,3,4,5,6]
median = [0.678,0.794,0.628,0.02,0.538,0.02]
adaclip2 = [0.651,0.565,0.539,0.02,0.561, 0.172]
fixdpc = [0.681,0.655,0.539,0.108,0.581,0.215]
mapac = [0.681,0.809,0.874,0.234,0.581,0.225]
nondp = [0.681,0.877,0.917,3,2,0.271]
plt.bar([i-1.5*width for i in x],median, width=width, color='rosybrown',label='Median')
plt.bar([i-0.5*width for i in x], adaclip2, width=width, color='lightcoral',label='AdaClip2')
plt.bar([i+0.5*width for i in x], fixdpc, width=width, color='indianred',label='FixDP-C')
plt.bar([i+1.5*width for i in x], mapac, width=width, color='brown',hatch='/',label='MAPA-C')
plt.legend(ncol=2)
plt.xticks([0.4, 1,2,3,4,5,6],('\n$\epsilon=$','Synth.\n 88.34','FEMNIST\n195.9','CelebA\n23.54','Shakes.\n151.9','Twitter\n22.12','Reddit\n95.80'))

plt.savefig('MAPA_cross_device.pdf', dpi=1200)
# fig.suptitle("Figure with multiple Subplots")
plt.show()


plt.show()