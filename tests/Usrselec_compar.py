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
file_paths_celeba_full = [
    '../results/celeba/--fast.log', '../results/celeba/--aouprior.log',
    '../results/celeba/--fedavg15.log','../results/celeba/--fedavg30.log',
    '../results/celeba/--freagg0.8.log','../results/celeba/--freagg1.2.log',
    '../results/celeba/--hfl.log',
    '../results/celeba/--kafl.log',
    '../results/celeba/--schagg0.8.log','../results/celeba/--schagg1.2.log'# 添加更多文件路径
]
file_paths_celeba = [
    '../results/celeba/--fast.log', '../results/celeba/--aouprior.log',
    '../results/celeba/--fedavg15.log',
    '../results/celeba/--freagg0.8.log',
    '../results/celeba/--hfl.log',
    '../results/celeba/--kafl.log',
    '../results/celeba/--schagg0.8.log'# 添加更多文件路径
]
file_paths_femnist_full = [
    '../results/femnist/fast-w.log', '../results/femnist/aouprior.log',
    '../results/femnist/fedavg.log','../results/femnist/fedavg30.log',
    '../results/femnist/freagg.log','../results/femnist/freagg1.2.log',
    '../results/femnist/hfl.log',
    '../results/femnist/kafl.log',
    '../results/femnist/schagg.log','../results/femnist/schagg1.2.log'# 添加更多文件路径
]
file_paths_femnist = [
    '../results/femnist/fast-w.log', '../results/femnist/aouprior.log',
    '../results/femnist/fedavg.log',
    '../results/femnist/freagg.log',
    '../results/femnist/hfl.log',
    '../results/femnist/kafl.log',
    '../results/femnist/schagg.log'# 添加更多文件路径
]
file_paths_shakes_full = [
    '../results/shakes/fast-w.log', '../results/shakes/aouprior.log',
    '../results/shakes/fedavg.log','../results/shakes/fedavg30.log',
    '../results/shakes/freagg.log','../results/shakes/freagg1.2.log',
    '../results/shakes/hfl.log',
    '../results/shakes/kafl.log',
    '../results/shakes/schagg.log','../results/shakes/schagg1.2.log'# 添加更多文件路径
]
file_paths_shakes = [
    '../results/shakes/fast-w.log', '../results/shakes/aouprior.log',
    '../results/shakes/fedavg.log',
    '../results/shakes/freagg.log',
    '../results/shakes/hfl.log',
    '../results/shakes/kafl.log',
    '../results/shakes/schagg.log'# 添加更多文件路径
]
file_paths_synthetic_full = [
    '../results/synthetic/fast-w.log', '../results/synthetic/aouprior.log',
    '../results/synthetic/fedavg.log','../results/synthetic/fedavg30.log',
    '../results/synthetic/freagg.log','../results/synthetic/freagg1.2.log',
    '../results/synthetic/hfl.log',
    '../results/synthetic/kafl.log',
    '../results/synthetic/schagg.log','../results/synthetic/schagg1.2.log'# 添加更多文件路径
]
file_paths_synthetic = [
    '../results/synthetic/fast-w.log', '../results/synthetic/aouprior.log',
    '../results/synthetic/fedavg.log',
    '../results/synthetic/freagg.log',
    '../results/synthetic/hfl.log',
    '../results/synthetic/kafl.log',
    '../results/synthetic/schagg.log'# 添加更多文件路径
]

# 定义 markers 和 linestyles 列表
markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'X', 'h', '<', '>']
linestyles = ['-', '--', '-.', ':']
legends = ['GraSAD','AouPrior', 'AFedavg15','FreAgg0.8', 'HFL', 'KAFL', 'SchAgg0.8']
# 使用 matplotlib 的内置 'tab10','Set1','Dark2' 颜色循环
colors = plt.cm.Dark2.colors

# 初始化绘图
plt.figure(figsize=(12, 8))
# 为每个文件提取数据并绘图
for i, file_path in enumerate(file_paths_celeba):
    rounds, test_accs = extract_round_and_acc(file_path)
    # 获取文件名（去掉路径和扩展名），用作标签
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    test_stable = np.mean(test_accs[-10:])
    print(file_name, round(test_accs[-1]*100,2), round((test_accs[-1]-test_stable)*100,2))
    # 自动选择 marker 和 linestyle，使用循环取值
    marker = markers[i % len(markers)]
    linestyle = linestyles[i % len(linestyles)]
    color = colors[i % len(colors)]
    plt.plot(rounds[::2], test_accs[::2], marker=marker, ms=10,
             linestyle=linestyle, linewidth=3, color=color, label=legends[i])

# 自定义图表
plt.xlabel('Round', fontsize=24)
plt.xticks(fontsize = 24)
plt.yticks(fontsize = 24)
plt.ylabel('Test Accuracy', fontsize=24)
# plt.ylim((0.5,0.85))
# axins = ax.inset_axes([0.5, 0.5, 0.4, 0.4])
# zoom_x = x[(x >= 4) & (x <= 6)]
# zoom_y = y[(x >= 4) & (x <= 6)]
# axins.plot(zoom_x, zoom_y)
# plt.title('Test Accuracy Comparison across Multiple Files', fontsize=16)
plt.legend(fontsize=22,loc= 'best')
plt.grid(True)
plt.tight_layout()

plt.draw()
plt.savefig('compar_celeba.eps', format='eps')
plt.show()