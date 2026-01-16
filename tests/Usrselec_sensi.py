import matplotlib.pyplot as plt
import ast
import random
import numpy as np
import os

# 定义一个函数，提取每个文件的 round 和 test_acc 数据
def extract_round_and_acc(file_path):
    rounds, test_accs = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            entry = ast.literal_eval(line.strip())
            round_num = entry.get('Round')
            test_acc = entry.get('Results_raw', {}).get('test_acc')
            if round_num is not None and test_acc is not None:
                rounds.append(round_num)
                test_accs.append(test_acc)
    return rounds, test_accs

# 文件路径列表
file_paths_shakes_factor = [
    '../results/sensitivity/shakes/factor0.3.log', '../results/sensitivity/shakes/factor0.4.log',
    '../results/sensitivity/shakes/factor0.5.log','../results/sensitivity/shakes/factor0.6.log',
    '../results/sensitivity/shakes/factor0.7.log']

file_paths_shakes_seed = [
    '../results/sensitivity/shakes/seed1.log', '../results/sensitivity/shakes/seed12.log',
    '../results/sensitivity/shakes/seed1234.log','../results/sensitivity/shakes/seed12345.log',
    '../results/sensitivity/shakes/seed12345.log']

file_paths_shakes_lr = [
    '../results/sensitivity/shakes/lr0.5.log', '../results/sensitivity/shakes/lr1.0.log',
    '../results/sensitivity/shakes/lr1.5.log','../results/sensitivity/shakes/lr2.0.log',
    '../results/sensitivity/shakes/lr2.5.log']

file_paths_shakes_epoch = [
    '../results/sensitivity/shakes/epoch1.log', '../results/sensitivity/shakes/epoch2.log',
    '../results/sensitivity/shakes/epoch3.log','../results/sensitivity/shakes/epoch4.log',
    '../results/sensitivity/shakes/epoch5.log']

file_paths_celeba_factor = [
    '../results/sensitivity/celeba/factor0.3.log', '../results/sensitivity/celeba/factor0.4.log',
    '../results/sensitivity/celeba/factor0.5.log','../results/sensitivity/celeba/factor0.6.log',
    '../results/sensitivity/celeba/factor0.7.log']

file_paths_celeba_seed = [
    '../results/sensitivity/celeba/seed1.log', '../results/sensitivity/celeba/seed12.log',
    '../results/sensitivity/celeba/seed1234.log','../results/sensitivity/celeba/seed12345.log',
    '../results/sensitivity/celeba/seed12345.log']

file_paths_celeba_lr = [
    '../results/sensitivity/celeba/lr0.001.log', '../results/sensitivity/celeba/lr0.01.log',
    '../results/sensitivity/celeba/lr0.02.log','../results/sensitivity/celeba/lr0.03.log',
    '../results/sensitivity/celeba/lr0.1.log']

file_paths_celeba_epoch = [
    '../results/sensitivity/celeba/epoch1.log', '../results/sensitivity/celeba/epoch2.log',
    '../results/sensitivity/celeba/epoch3.log','../results/sensitivity/celeba/epoch4.log',
    '../results/sensitivity/celeba/epoch5.log']


# 定义 markers 和 linestyles 列表
markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'X', 'h', '<', '>']
linestyles = ['-', '--', '-.', ':']
# 使用 matplotlib 的内置 'tab10','Set1','Dark2' 颜色循环
colors = plt.cm.Dark2.colors

# 初始化绘图
plt.figure(figsize=(6, 4))
# 为每个文件提取数据并绘图
test_acc, test_acc_best = [], []
for i, file_path in enumerate(file_paths_celeba_epoch):
    rounds, test_accs = extract_round_and_acc(file_path)
    # 获取文件名（去掉路径和扩展名），用作标签
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    test_stable = np.mean(test_accs[-10:])
    print(file_name, round(test_accs[-1]*100,2), round((test_accs[-1]-test_stable)*100,2))
    test_acc.append(test_accs[-1])
    test_acc_best.append(max(test_accs))

plt.plot(np.arange(5), test_acc, label='Final', marker='o', ms=10,
                 linestyle='--', linewidth=3)
plt.plot(np.arange(5), test_acc_best, label='Best', marker='s', ms=10,
                 linestyle='--', linewidth=3)
plt.plot(np.arange(5), test_acc[0]*np.ones(5), label='Default',
                 linestyle=':', linewidth=3)


# 自定义图表
# plt.xlabel('Factor', fontsize=24)
plt.xticks(ticks=np.arange(5),labels=['1','2','3','4','5'],fontsize = 24)
plt.yticks(fontsize = 24)
plt.ylabel('CelebA', fontsize=24)
# plt.ylim((0.45,0.5))
# axins = ax.inset_axes([0.5, 0.5, 0.4, 0.4])
# zoom_x = x[(x >= 4) & (x <= 6)]
# zoom_y = y[(x >= 4) & (x <= 6)]
# axins.plot(zoom_x, zoom_y)
# plt.title('Test Accuracy Comparison across Multiple Files', fontsize=16)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()

plt.draw()
plt.savefig('sensi_celeba_epoch.eps', format='eps')
plt.show()