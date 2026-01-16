import copy, itertools
import torch
import numpy as np


def discount_func(self, staleness):
    """
    Served as an example, we discount the model update with staleness tau \
    as: ``(1.0/((1.0+\tau)**factor))``, \
    which has been used in previous studies such as FedAsync ( \
    Asynchronous Federated Optimization) and FedBuff \
    (Federated Learning with Buffered Asynchronous Aggregation).
    """
    return (1.0 /
            ((1.0 + staleness) ** self.cfg.asyn.staleness_discount_factor))

    # 定义用户分布选择
def _distributions_filter(self, distr):
        # 统计用户的分布列表distr中用户的数量len_distr和训练集的标签数量len_class
        len_distr, len_class = len(distr), len(distr[0])
        # 也可以使用wasserstein_distance来计算分布间的距离，但其缺点是不能完全反映
        # 标签分布的差异，如[1,2,3]和[2,3,1]的wasserstein_distance=0
        # 使用方法import scipy.stats, scipy.stats.wasserstein_distance(dis1,dis2)

        if self.cfg.is_frozen():
            self.cfg.defrost()

        # 当使用user select 并且在第一回合时，初始化history population 为零向量
        if self.global_num == 0:
            self.cfg.usrsele.his_popu = [0.] * len_class

        # 转换为tensor格式，用于计算分布的范数
        if torch.is_tensor(distr) is False:
            distr_tensor = torch.tensor(distr, dtype=float)
        # 需要筛选的分布数量
        filter_num = round(self.cfg.usrsele.factor * len_distr)
        # 快速筛选方案，按照单个分布与均匀分布的差异从小到大进行选取
        strategy = self.cfg.usrsele.strategy
        his_popu = torch.tensor(self.cfg.usrsele.his_popu)
        if strategy == 'fast':
            # 将当前用户分布与历史累积使用的分布进行求和
            distr_tensor_accu = distr_tensor if self.global_num == 0 else (his_popu + distr_tensor) / 2
            # 计算分布列表中每个用户分布（行）与均匀分布的差异
            distr_tensor_diff = distr_tensor_accu - torch.ones_like(distr_tensor_accu) / len_class
            # 计算每行(norm(2,1)中的1)与均匀分布差异的L2范数并升序ascend排列
            _, index_as = distr_tensor_diff.norm(2, 1).sort()
            # 筛选出filter_num个前差异最小的分布
            filter_distri = distr_tensor[index_as[:filter_num]]
            his_popu = filter_distri.mean(0) if self.global_num == 0 else (his_popu + filter_distri.mean(0)) / 2
            self.cfg.usrsele.his_popu = his_popu.tolist()
            return index_as[:filter_num]

        # 组合筛选方案，逐一计算组合与均匀分布的差异，挑选出L2范数最小的组合
        if strategy == 'combi':
            # 遍历所有可能的组合数，共C_n^k，其中n=len_distr，k=filter_num
            for i in range(np.math.comb(len_distr, filter_num)):
                # 生成一种组合
                combi = np.random.choice(len_distr, filter_num)
                # 选出对应的候选分布candidate_distribution
                distr_tensor_accu = (his_popu + distr_tensor) / 2
                distr_uniform = torch.ones_like(distr_tensor_accu) / len_class
                distr_tensor_diff = distr_tensor_accu - distr_uniform
                can_distri = distr_tensor_diff[combi]
                if can_distri.norm() <= self.cfg.usrsele.combi_thre:
                    filter_distri = distr_tensor[combi]
                    self.cfg.usrsele.combi_thre = can_distri.norm().item()
                    his_popu = (his_popu + filter_distri.mean(0)) / 2
            self.cfg.usrsele.his_popu = his_popu.tolist()
            return combi

        if strategy == 'coor_group':
            print("current strategy is coor_group")
            group_num = round(np.sqrt(len_class))
            group_ind = self.global_num % group_num
            distr_tensor_accu = (distr_tensor + his_popu) / 2
            end = (group_ind + 1) * group_num if (group_ind + 1) * group_num <= len_class else len_class
            distr_slice = distr_tensor_accu[:, group_ind * group_num: end]
            print("切片的区间是", group_ind * group_num, end)
            # 可将len_class 换为 len(distr_slice)
            uniform_slice = torch.ones_like(distr_slice) / len(distr_slice)
            distr_slice_diff = distr_slice - uniform_slice
            _, index_as = distr_slice_diff.norm(2, 1).sort()
            group_elem = []
            for i in range(len(index_as)):
                distr_slice_diff_mean = distr_slice_diff[0] if i == 0 else distr_slice_diff[:i + 1].mean(0)
                if distr_slice_diff_mean.norm() <= 0.7 * uniform_slice[0].norm():
                    group_elem.append(index_as[:i + 1])
                    filter_distri = distr_tensor[index_as[:i + 1]]
                    his_popu = (his_popu + filter_distri.mean(0)) / 2
                    self.cfg.usrsele.his_popu = his_popu.tolist()
                    return index_as[:i + 1]
                    break
            if len(group_elem) == 0:
                filter_distri = distr_tensor[index_as[:filter_num]]
                group_elem.append(index_as[:filter_num])
                his_popu = (his_popu + filter_distri.mean(0)) / 2
                self.cfg.usrsele.his_popu = his_popu.tolist()
                return index_as[:filter_num]
