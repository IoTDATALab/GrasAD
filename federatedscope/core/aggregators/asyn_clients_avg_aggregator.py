import copy, itertools
import sys

import torch, random
import numpy as np
from click import pause

from federatedscope.core.aggregators import ClientsAvgAggregator
from federatedscope.core.trainers.utils import get_random
from collections import OrderedDict


class AsynClientsAvgAggregator(ClientsAvgAggregator):
    """
    The aggregator used in asynchronous training, which discounts the \
    staled model updates
    """

    def __init__(self, model=None, device='cpu', config=None):
        super(AsynClientsAvgAggregator, self).__init__(model, device, config)

    def aggregate(self, agg_info):
        """
        To preform aggregation

        Arguments:
            agg_info (dict): the feedbacks from clients

        Returns:
            dict: the aggregated results
        """
        if self.cfg.is_frozen():
            self.cfg.defrost()

        # 判断是否只使用一个算法
        total_sum = (self.cfg.usrsele.use + self.cfg.usrsele_compa_hfl.use +
                     self.cfg.usrsele_compa_schandagg.use + self.cfg.usrsele_compa_freqandage.use +
                     self.cfg.usrsele_compa_kafl.use + self.cfg.usrsele_compa_aouprior.use +
                              self.cfg.usrsele_compa_wkafl.use)
        config_checks = [
            ('usrsele', self.cfg.usrsele.use),
            ('usrsele_compa_hfl', self.cfg.usrsele_compa_hfl.use),
            ('usrsele_compa_schandagg', self.cfg.usrsele_compa_schandagg.use),
            ('usrsele_compa_freqandage', self.cfg.usrsele_compa_freqandage.use),
            ('usrsele_compa_kafl', self.cfg.usrsele_compa_kafl.use),
            ('usrsele_compa_aouprior', self.cfg.usrsele_compa_aouprior.use),
            ('usrsele_compa_wkafl', self.cfg.usrsele_compa_wkafl.use)
        ]

        if total_sum == 1:
            models_plus_dist = agg_info["client_feedback"]  # 接收到用户发送的消息，包含训练集大小，模型参量，训练集分布
            models = [models_plus_dist[i][0:2] for i in range(len(models_plus_dist))]  # 分离出模型
            dists_ratio = [models_plus_dist[i][-1] for i in range(len(models_plus_dist))]  # 分离出训练集分布
            client_aggnum = np.array([user[0] for user in models_plus_dist])  # 提取用户上传的样本数量，将列表转为数组
            dists_num = np.array(dists_ratio) * client_aggnum[:, np.newaxis]  # dists_ratio中每一行乘以对应的样本数量
            dists = dists_num / (dists_num + 10 ** (-6)).sum(0)  # 按列归一化
            dists = dists / dists.sum(1)[:, np.newaxis]  # 按列归一化后再按行归一化
            if self.cfg.data.type == 'celeba' or self.cfg.data.type == 'synthetic':
                dists = dists[:,0:2]
        elif total_sum == 0:
            models = agg_info["client_feedback"]
        else:
            enabled_configs = [name for name, value in config_checks if value]
            print(f"警告：检测到 {total_sum} 个用户选择配置同时启用:")
            for config in enabled_configs:
                print(f"  - {config}.use = True")
            sys.exit(1)

        recover_fun = agg_info['recover_fun'] if (
                'recover_fun' in agg_info and self.cfg.federate.use_ss) else None
        staleness = [x[1]
                     for x in agg_info['staleness']]  # (client_id, staleness)
        client_id = [x[0]
                     for x in agg_info['staleness']]
        id_staleness = agg_info['staleness']
        # print(staleness)
        self.global_num = agg_info['global_num']

        # 如果设置了usrsele，则使用filtered_clients从models中选出相应用户的模型
        # 否则models为接收到的全部模型
        if self.cfg.usrsele.use == True:
            if self.cfg.usrsele.ablation == 'fast-s' or self.cfg.usrsele.ablation == 'combi-s':
                if self.cfg.asyn.min_received_num:
                    num = np.int0(self.cfg.asyn.min_received_num * self.cfg.usrsele.factor).item()
                else:
                    num = np.int0(self.cfg.asyn.min_received_rate * self.cfg.federate.client_num * self.cfg.usrsele.factor).item()
                models = models[0:num]
                staleness = staleness[0:num]
                sort_idx, sort_value = self._usrsele_sort(models, staleness)
                weight_list = self._weight_func_ours(models, staleness, sort_idx)
                avg_model = self._para_weighted_avg(models,
                                                    recover_fun=recover_fun,
                                                    staleness=staleness,
                                                    weights=weight_list)
            elif self.cfg.usrsele.ablation == 'fast-w' or self.cfg.usrsele.ablation == 'combi-w':
                filtered_clients, sort_idx, common_idx = self._distributions_filter(models, staleness, client_id, distr=dists)
                # print("第{}轮次筛选出用户为{}".format(self.global_num, filtered_clients))
                if self.global_num <= 1/3 * self.cfg.federate.total_round_num:
                    models = [models[i] for i in filtered_clients]
                    staleness = [staleness[i] for i in filtered_clients]
                else:
                    models = [models[i] for i in sort_idx]
                    staleness = [staleness[i] for i in sort_idx]
                avg_model = self._para_weighted_avg(models,
                                                    recover_fun=recover_fun,
                                                    staleness=staleness)
            else:
                # 筛选出符合条件的filtered clients
                filtered_clients, sort_idx, common_idx = self._distributions_filter(models, staleness, client_id, distr=dists)
                # print("第{}轮次筛选出用户为{}".format(self.global_num, filtered_clients))
                # amplify_idx = [filtered_clients.index(value) for value in common_idx]
                if self.global_num <= 1/3 * self.cfg.federate.total_round_num:
                    models = [models[i] for i in filtered_clients]
                    staleness = [staleness[i] for i in filtered_clients]
                else:
                    models = [models[i] for i in sort_idx]
                    staleness = [staleness[i] for i in sort_idx]

                # _, sort_value = self._usrsele_sort(models, staleness)
                weight_list = self._weight_func_ours(models, staleness, sort_idx)
                avg_model = self._para_weighted_avg(models,
                                                    recover_fun=recover_fun,
                                                    staleness=staleness,
                                                    weights=weight_list)
            self.cfg.usrsele.his_grad[0] = avg_model
        elif self.cfg.usrsele_compa_schandagg.use == True:
            filtered_clients = self._distributions_filter(distr=dists)
            models = [models[i] for i in filtered_clients]
            staleness = [staleness[i] for i in filtered_clients]
            avg_model = self._para_weighted_avg(models,
                                                recover_fun=recover_fun,
                                                staleness=staleness)
        elif self.cfg.usrsele_compa_freqandage.use == True:
            filtered_clients = self._distributions_filter(distr=dists, client_id=client_id)
            models = [models[i] for i in [client_id.index(value) for value in filtered_clients]]
            staleness = [staleness[i] for i in [client_id.index(value) for value in filtered_clients]]
            avg_model = self._para_weighted_avg(models,
                                                recover_fun=recover_fun,
                                                staleness=staleness)
        elif self.cfg.usrsele_compa_hfl.use == True:
            filtered_clients = self._distributions_filter(models, staleness, client_id, distr=dists)
            print("第{}轮次筛选出用户为{}".format(self.global_num, filtered_clients))
            models = [models[i] for i in [client_id.index(value) for value in filtered_clients]]
            staleness = [staleness[i] for i in [client_id.index(value) for value in filtered_clients]]
            avg_model = self._para_weighted_avg(models,
                                                recover_fun=recover_fun,
                                                staleness=staleness)
        elif self.cfg.usrsele_compa_kafl.use == True:
            weight_list = self._weight_func_kafl(models, client_id)
            avg_model = self._para_weighted_avg(models,
                                                recover_fun=recover_fun,
                                                staleness=staleness,
                                                weights=weight_list)
            self.cfg.usrsele_compa_kafl.global_model[0] = avg_model
        elif self.cfg.usrsele_compa_aouprior.use == True:
            filtered_clients = self._distributions_filter(models, staleness, distr=dists)
            models = [models[i] for i in filtered_clients]
            staleness = [staleness[i] for i in filtered_clients]
            avg_model = self._para_weighted_avg(models,
                                                recover_fun=recover_fun,
                                                staleness=staleness)
        elif self.cfg.usrsele_compa_wkafl.use == True:
            print('This is wkafl')
            # 获取模型层数和各层形状
            Layers_shape = self.getModelLayers(models[0][1])
            e = torch.exp(torch.tensor(1.))
            _, index = torch.sort(torch.tensor(staleness))
            # normStandard = self.L_norm(models[index[0]][1])
            weight = (e/2)**(-torch.tensor(staleness))
            if torch.sum(weight) == 0:
                print("延时过大。")
                Collect_Gradients = models[index[0]][1]
            else:
                weight = weight / torch.sum(weight)
                Collect_Gradients = models[0][1]
                for key in Collect_Gradients:
                    for i in range(len(models)):
                        local_sample_size, local_model = models[i]
                        assert staleness is not None
                        if isinstance(local_model[key], torch.Tensor):
                            local_model[key] = local_model[key].float()
                        else:
                            local_model[key] = torch.FloatTensor(local_model[key])
                        if i == 0:
                            Collect_Gradients[key] = local_model[key] * weight[i]
                        else:
                            Collect_Gradients[key] += local_model[key] * weight[i]

            Sel_gradients = self._compa_wkafl_collect_gradient(Collect_Gradients, models, Clip=False)
            avg_model = self._para_weighted_avg(Sel_gradients,
                                                recover_fun=recover_fun,
                                                staleness=staleness)

        else:
            avg_model = self._para_weighted_avg(models,
                                                recover_fun=recover_fun,
                                                staleness=staleness)
        updated_model = copy.deepcopy(avg_model)
        init_model = self.model.state_dict()
        for key in avg_model:
            updated_model[key] = init_model[key] + avg_model[key]

        return updated_model

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

    def _para_weighted_avg(self, models, recover_fun=None, staleness=None, weights=None):
        """
        Calculates the weighted average of models.
        """
        training_set_size = 0
        for i in range(len(models)):
            sample_size, _ = models[i]
            training_set_size += sample_size

        sample_size, avg_model = models[0]

        if self.cfg.usrsele.use == True:
            print('EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE, the used algorithm is UserSelect')
            for key in avg_model:
                for i in range(len(models)):
                    # 根据数据量倒数计算权重
                    local_sample_size, local_model = models[i]
                    assert staleness is not None
                    if self.cfg.usrsele.ablation == 'fast-w' or self.cfg.usrsele.ablation == 'combi-w':
                        weight = local_sample_size / training_set_size
                        weight *= self.discount_func(staleness[i])
                    else:
                        # 使用列表时要用深拷贝，否则weights中的值会随weight改变
                        weight = copy.deepcopy(weights[i])
                        weight *= self.discount_func(staleness[i])

                    if isinstance(local_model[key], torch.Tensor):
                        local_model[key] = local_model[key].float()
                    else:
                        local_model[key] = torch.FloatTensor(local_model[key])

                    if i == 0:
                        avg_model[key] = local_model[key] * weight
                    else:
                        avg_model[key] += local_model[key] * weight
                    # local_sample_size, local_model = models[i]
                    # if self.cfg.federate.ignore_weight:
                    #     weight = 1.0 / len(models)
                    # else:
                    #     weight = local_sample_size / training_set_size
                    #
                    # assert staleness is not None
                    # weight *= self.discount_func(staleness[i])
                    # if isinstance(local_model[key], torch.Tensor):
                    #     local_model[key] = local_model[key].float()
                    # else:
                    #     local_model[key] = torch.FloatTensor(local_model[key])
                    #
                    # if i == 0:
                    #     avg_model[key] = local_model[key] * weight
                    # else:
                    #     avg_model[key] += local_model[key] * weight
            return avg_model

        elif self.cfg.usrsele_compa_schandagg.use or self.cfg.usrsele_compa_freqandage.use == True:
            print('EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE, the used algorithm is SchandAgg or FreandAgg')
            if self.cfg.usrsele_compa_schandagg.use == True:
                gamma = self.cfg.usrsele_compa_schandagg.gamma
            else:
                gamma = self.cfg.usrsele_compa_freqandage.gamma
            weights = torch.zeros(len(models))
            training_size_weights = 0
            for i in range(len(models)):
                sample_size, _ = models[i]
                training_size_weights += sample_size*(gamma**staleness[i])
            for i in range(len(models)):
                weights[i] = sample_size*(gamma**staleness[i])/training_size_weights
            for key in avg_model:
                for i in range(len(models)):
                    local_sample_size, local_model = models[i]
                    weight = weights[i]*0.1 # 这里的权重需要减小，按照往上数4行的归一化权重将导致不收敛。
                    assert staleness is not None
                    if isinstance(local_model[key], torch.Tensor):
                        local_model[key] = local_model[key].float()
                    else:
                        local_model[key] = torch.FloatTensor(local_model[key])
                    if i == 0:
                        avg_model[key] = local_model[key] * weight
                    else:
                        avg_model[key] += local_model[key] * weight
            return avg_model
        elif self.cfg.usrsele_compa_kafl.use == True:
            print('EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE, the used algorithm is KAFL')
            for key in avg_model:
                for i in range(len(models)):
                    local_sample_size, local_model = models[i]
                    weight = copy.deepcopy(weights[i])
                    assert staleness is not None
                    weight *= self.discount_func(staleness[i])
                    if isinstance(local_model[key], torch.Tensor):
                        local_model[key] = local_model[key].float()
                    else:
                        local_model[key] = torch.FloatTensor(local_model[key])

                    if i == 0:
                        avg_model[key] = local_model[key] * weight
                    else:
                        avg_model[key] += local_model[key] * weight
            return avg_model
        else:
            print('EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE, the used algorithm is non-dp')
            parameters_num = self._model_para_size(avg_model)
            for key in avg_model:
                for i in range(len(models)):
                    local_sample_size, local_model = models[i]
                    if self.cfg.federate.ignore_weight:
                        weight = 1.0 / len(models)
                    else:
                        weight = local_sample_size / training_set_size

                    assert staleness is not None
                    weight *= self.discount_func(staleness[i])
                    if isinstance(local_model[key], torch.Tensor):
                        local_model[key] = local_model[key].float()
                    else:
                        local_model[key] = torch.FloatTensor(local_model[key])

                    if i == 0:
                        avg_model[key] = local_model[key] * weight
                    else:
                        avg_model[key] += local_model[key] * weight
            return avg_model

    # 计算模型可训练参量个数 lyn
    def _model_para_size(self, model):
        model_num = 0
        for key in model:
            if 'running' not in key and 'tracked' not in key:
                shape_size = len(model[key].shape)
                layer_num = 1
                for i in range(shape_size):
                    layer_num *= model[key].shape[i]
                model_num += layer_num
        return model_num

    # 定义梯度范数计算
    def _model_norm(self, model):
        norm_temp = 0.
        for key in model:
            if 'running' not in key and 'tracked' not in key:
                norm_temp += model[key].float().norm() ** 2
        return torch.sqrt(norm_temp)

    # 定义梯度裁剪, 异步联邦中cleint上传的是本地模型改变量，包含了学习率，所以cipbound应该是裁剪大小乘以学习率
    def _grad_clip(self, clipbound, model):
        norm = self._model_norm(model)
        for key in model:
            if norm > clipbound and 'running' not in key and 'tracked' not in key:
                model[key] = model[key] * clipbound / norm  # 裁剪
        return norm, model

    # aggregator_num choice
    def _aggregator_num_set(self):
        if self.cfg.data.type == 'femnist':
            if (self.cfg.mapa.use and self.cfg.mapa.style == 'sample') or self.cfg.adaclip1.use or self.cfg.fixdps.use:
                aggregator_num = self.cfg.dataloader.batch_size  # 70
            else:
                aggregator_num = self.cfg.asyn.min_received_num  # 70
        elif self.cfg.data.type == 'celeba':
            if (self.cfg.mapa.use and self.cfg.mapa.style == 'client') or self.cfg.adaclip2.use or self.cfg.fixdpc.use \
                    or self.cfg.median.use:
                aggregator_num = self.cfg.asyn.min_received_num  # 140
            else:
                aggregator_num = self.cfg.dataloader.batch_size  # 30
        else:
            aggregator_num = self.cfg.asyn.min_received_num
        return aggregator_num

    # 定义用户分布选择
    def _distributions_filter(self, models=None, staleness=None, client_id=None, distr=None):
        # 统计用户的分布列表distr中用户的数量len_distr和训练集的标签数量len_class
        len_distr, len_class = len(distr), len(distr[0])
        # 也可以使用wasserstein_distance来计算分布间的距离，但其缺点是不能完全反映
        # 标签分布的差异，如[1,2,3]和[2,3,1]的wasserstein_distance=0
        # 使用方法import scipy.stats, scipy.stats.wasserstein_distance(dis1,dis2)

        # 转换为tensor格式，用于计算分布的范数
        if torch.is_tensor(distr) is False:
            distr_tensor = torch.tensor(distr, dtype=float)
        if self.cfg.usrsele_compa_schandagg.use == True:
            print('usrsele_compa_schandagg')
            filter_idx = self._client_select_comp_schandagg(distr_tensor, len_distr, len_class)
            return filter_idx
        elif self.cfg.usrsele_compa_hfl.use == True:
            print('usrsele_compa_hfl')
            filter_idx = self._client_select_comp_hfl(models, staleness, client_id, distr)
            return filter_idx
        elif self.cfg.usrsele_compa_kafl.use == True:
            print('usrsele_compa_kafl')
        elif self.cfg.usrsele_compa_freqandage.use == True:
            print('usrsele_compa_freqandage')
            filter_idx = self._client_select_comp_freandagg(client_id, len_distr)
            return filter_idx
        elif self.cfg.usrsele.use == True:
            print('usrsele')
            sort_idx, sort_value = self._usrsele_sort(models, staleness)
            filter_idx, common_idx, sort_filteridx = self._client_select_ours(distr_tensor, len_distr, len_class, sort_idx)
            return filter_idx, sort_filteridx, common_idx
        elif self.cfg.usrsele_compa_aouprior.use == True:
            print('aouprior')
            filter_idx = self._client_select_comp_aouprio(staleness, distr_tensor)
            return filter_idx
        else:
            print('random selection')
        # 快速筛选方案，按照单个分布与均匀分布的差异从小到大进行选取


    def group_ele_sum(self, group_ele):
        if len(group_ele) == 0:
            return 0
        if len(group_ele) > 0:
            sum = 0
            for i in range(len(group_ele)):
                sum += len(group_ele[i])
            return sum

    def _weight_func_ours(self, models, staleness, sort_idx):
        training_set_size, training_set_size_inv = 0, 0
        for i in range(len(models)):
            sample_size, _ = models[i]
            training_set_size += sample_size
            training_set_size_inv += 1/sample_size
        weights_initial1 = [model[0]/training_set_size for model in models]
        weights_initial2 = [(1/model[0])/training_set_size_inv for model in models]
        # weight_list = weights_initial1 if self.global_num <= 2/3 * self.cfg.federate.total_round_num else weights_initial2
        if self.global_num <= 1/3 * self.cfg.federate.total_round_num:
            weight_list = weights_initial1
        elif self.global_num <=2/3 * self.cfg.federate.total_round_num:
            weight_list = weights_initial2
        else:
            weight_list = weights_initial1



        # weight_list = weights_initial2 if (self.global_num/20) % 10 == 7|8|9 else weights_initial1
        # weight_list = [weights_initial1[i]+1/(1+self.global_num)*weights_initial2[i] for i in range(len(weights_initial2))]
        # weight_list = [weights_initial1[i]+1/2*np.sin(weights_initial2[i]) for i in range(len(weights_initial2))]
        # weight_list = [weights_initial1[i] + weights_initial2[i] for i in range(len(weights_initial2))]

        # print('AAAAAAAAAAAAAAAAAAaaaaaaaaaaaaaaa',sort_value)
        # weights_adjust1 = [weights_initial1[i] * sort_value[i] for i in range(len(weights_initial1))]
        # weights_adjust2 = [weights_initial2[i] * sort_value[i] for i in range(len(weights_initial2))]
        # weight1 = [value / sum(weights_adjust1) for value in weights_adjust1]
        # weight2 = [value / sum(weights_adjust2) for value in weights_adjust2]
        # weight_list = weight1 if self.global_num % 2 == 0 else weight2


        # if self.global_num == 0:
        #     weight_list = weights_initial1
        #     # weight = []
        #     # for i in range(len(models)):
        #     #     sample_size, _ = models[i]
        #     #     if self.cfg.federate.ignore_weight:
        #     #         weight.append(1.0 / len(models))
        #     #     else:
        #     #         weight.append((1/sample_size) / training_set_size_inv)
        # else:
        #     vector_deep, out_sim, in_sim, weight = [], [], [], []
        #     model_num, stand_grad = len(models), self.cfg.usrsele.his_grad[0]
        #     if self.cfg.model.type == 'lr':
        #         stand_grad = torch.cat([stand_grad['fc.weight'].view(-1), stand_grad['fc.bias'].view(-1)])
        #     else :
        #         stand_grad = torch.cat([stand_grad['fc1.weight'].view(-1), stand_grad['fc1.bias'].view(-1),
        #                                 stand_grad['fc2.weight'].view(-1), stand_grad['fc2.bias'].view(-1)])
        #
        #     for i, model in enumerate(models):
        #         if self.cfg.model.type == 'lr':
        #             vector = torch.cat([model[1]['fc.weight'].view(-1), model[1]['fc.bias'].view(-1)])
        #         else :
        #             vector = torch.cat([model[1]['fc1.weight'].view(-1), model[1]['fc1.bias'].view(-1),
        #                                 model[1]['fc2.weight'].view(-1), model[1]['fc2.bias'].view(-1)])
        #         sim = torch.cosine_similarity(vector.unsqueeze(0), stand_grad.unsqueeze(0))
        #         out_sim.append(sim)
        #         # out_sim.append(0) if sim < 0 else out_sim.append(sim)
        #         # vector_deep.append(vector)
        #
        #     # for i in range(model_num):
        #     #     score = torch.tensor([0.])
        #     #     for j in range(model_num):
        #     #         if j != i:
        #     #             cosine_sim = torch.cosine_similarity(vector_deep[i].unsqueeze(0), vector_deep[j].unsqueeze(0))
        #     #             # exp_cosine_sim = torch.exp(cosine_sim)
        #     #             # score += exp_cosine_sim
        #     #             score += cosine_sim
        #     #             #score += torch.cosine_similarity(vector_deep[i].unsqueeze(0), vector_deep[j].unsqueeze(0))
        #     #     in_sim.append(score / (model_num - 1) if model_num > 1 else score)
        #     # print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',out_sim, in_sim)
        #     for i in range(model_num):
        #         if self.global_num <= 2/3 * self.cfg.federate.total_round_num:
        #             weight.append(torch.exp(out_sim[i]) * weights_initial1[i])
        #         else:
        #             weight.append(torch.exp(out_sim[i]) * weights_initial2[i])
        #         # weight.append(torch.exp((out_sim[i]+in_sim[i]) * weights_initial[i]))
        #         # weight.append((0.3 * torch.exp(out_sim[i]) + 0.7 * torch.exp(in_sim[i])) * weights_initial[i])
        #         # weight.append(0.3 * out_sim[i] + 0.7 * (score / (model_num - 1)))
        #     weight_list = [value / sum(weight) for value in weight]
        return weight_list

    def _weight_func_kafl(self, models, client_id):
        training_set_size = 0
        for i in range(len(models)):
            sample_size, _ = models[i]
            training_set_size += sample_size

        if self.global_num == 0:
            self.cfg.usrsele_compa_kafl.frelist = [0] * self.cfg.federate.client_num
            weights = []
            for i in range(len(models)):
                sample_size, _ = models[i]
                if self.cfg.federate.ignore_weight:
                    weights.append(1.0 / len(models))
                else:
                    weights.append(sample_size / training_set_size)
        else:
            vector_deep, weight, lambda_value, q = [], [], [], []
            frelist = self.cfg.usrsele_compa_kafl.frelist
            frelist_update = [frelist[x-1] + 1 if x in client_id else frelist[x-1]
                              for x in range(1, self.cfg.federate.client_num+1)]
            self.cfg.usrsele_compa_kafl.frelist = frelist_update
            frelist_slice = [frelist_update[x-1] for x in client_id] #原来的id是从1开始计数，改为从0开始
            sorted_zip = sorted(zip(frelist_slice, client_id))
            sorted_fre, sorted_id = map(list, zip(*sorted_zip))
            model_num, stand_grad = len(models), self.cfg.usrsele_compa_kafl.global_model[0]
            if self.cfg.model.type == 'lr':
                global_model = torch.cat([stand_grad['fc.weight'].view(-1), stand_grad['fc.bias'].view(-1)])
            elif self.cfg.model.type == 'lstm':
                global_model = torch.cat([stand_grad['decoder.weight'].view(-1), stand_grad['decoder.bias'].view(-1)])
            else:
                global_model = torch.cat([stand_grad['fc1.weight'].view(-1), stand_grad['fc1.bias'].view(-1),
                                        stand_grad['fc2.weight'].view(-1), stand_grad['fc2.bias'].view(-1)])

            lambda_deno = torch.dot(global_model, global_model)
            for i, model in enumerate(models):
                if self.cfg.model.type == 'lr':
                    vector = torch.cat([stand_grad['fc.weight'].view(-1), stand_grad['fc.bias'].view(-1)])
                elif self.cfg.model.type == 'lstm':
                    vector = torch.cat([stand_grad['decoder.weight'].view(-1), stand_grad['decoder.bias'].view(-1)])
                else:
                    vector = torch.cat([stand_grad['fc1.weight'].view(-1), stand_grad['fc1.bias'].view(-1),
                                        stand_grad['fc2.weight'].view(-1), stand_grad['fc2.bias'].view(-1)])

                lambda_nume= torch.dot(vector, global_model)
                lambda_value.append(lambda_nume/lambda_deno - 1)
                q.append(sorted_fre[len(models)-i-1]/sum(sorted_fre))
            for i in range(len(models)):
                sample_size, _ = models[i]
                pos = sorted_id.index(client_id[i]) # 从排序的索引中定位与model相应的位置
                value = self.cfg.usrsele_compa_kafl.rho * np.abs(lambda_value[i]/q[pos])
                weight.append(sample_size * np.exp(-value))
            weight_sum = sum(weight)
            weights = [x / weight_sum for x in weight]
        return weights

    def _client_select_comp_hfl(self, models, staleness, client_id, distr):
        # 统计设备的分布列表distr中用户的数量len_distr
        len_distr = len(distr)

        training_set_size = 0
        for i in range(len(models)):
            sample_size, _ = models[i]
            training_set_size += sample_size

        if self.global_num == 0:
            weight = []
            for i in range(len(models)):
                sample_size, _ = models[i]
                if self.cfg.federate.ignore_weight:
                    weight.append(1.0 / len(models))
                else:
                    weight.append(sample_size / training_set_size)

        vector_deep,out_sim, in_sim, result ,result_value ,value= [], [], [], [], [] ,[]
        model_num = len(models)
        for i in range(model_num):
            if self.cfg.model.type == 'lr':
                vector = torch.cat([models[i][1]['fc.weight'].view(-1), models[i][1]['fc.bias'].view(-1)])
            elif self.cfg.model.type == 'lstm':
                vector = torch.cat([models[i][1]['decoder.weight'].view(-1), models[i][1]['decoder.bias'].view(-1)])
            else :
                vector = torch.cat([models[i][1]['fc1.weight'].view(-1), models[i][1]['fc1.bias'].view(-1),
                                    models[i][1]['fc2.weight'].view(-1), models[i][1]['fc2.bias'].view(-1)])

            vector_deep.append(vector)

        T = vector_deep[0].clone()
        for v in vector_deep[1:]:
            T += v

        for i in range(model_num):
            score = torch.tensor([0.])
            for j in range(model_num):
                if j != i:
                    #score += torch.cosine_similarity(vector_deep[i].unsqueeze(0), vector_deep[j].unsqueeze(0))
                    score += torch.dot(vector_deep[i],vector_deep[j])
            in_sim.append(-score / (model_num - 1))
        #将列表in_sim里的0维张量以数值的形式放到列表value里
        for s in in_sim:
            value.append(s.item())

        for i in range(model_num):
            dot_product = torch.dot(T/model_num, vector_deep[i])
            #列表result是一个包含0维张量的列表
            result.append(dot_product)
        #把列表result里的0维张量数值取出来放到result_value列表里
        for tensor in result:
            result_value.append(tensor.item())

        #learning utility
        learning_utility = [x + y for x, y in zip(result_value , value)]

        round_latency = [1 / (i + 1) for i in staleness]
        U = [a * b for a, b in zip(learning_utility, round_latency)]
        sorted_U,list1,list1_values,list2 = [],[],[],[]

       # 需要筛选的设备数量/
        filter_num = round(self.cfg.usrsele_compa_hfl.factor * len_distr)
        # 首先对原列表进行排序，并保持索引
        sorted_U = sorted(enumerate(U), key=lambda x: x[1], reverse=True)
        # 提取最大的filter_num个值以及它们的索引
        list1 = sorted_U[:filter_num]
        # 分开值和索引，并将索引加1
        # list1_values = [value for index, value in list1]
        list2 = [index for index, value in list1] #list2指从0开始编号的下标
        list3 = [client_id[index] for index in list2] #list3指用户的id
        return list3

    def _usrsele_sort(self, models, staleness):
        # 统计设备的models中用户的数量
        len_distr = len(models)

        training_set_size = 0
        for i in range(len(models)):
            sample_size, _ = models[i]
            training_set_size += sample_size

        if self.global_num == 0:
            weight = []
            for i in range(len(models)):
                sample_size, _ = models[i]
                if self.cfg.federate.ignore_weight:
                    weight.append(1.0 / len(models))
                else:
                    weight.append(sample_size / training_set_size)

        vector_deep,out_sim, in_sim, result ,result_value ,value= [], [], [], [], [] ,[]
        model_num = len(models)
        for i in range(model_num):
            if self.cfg.model.type == 'lr':
                vector = torch.cat([models[i][1]['fc.weight'].view(-1), models[i][1]['fc.bias'].view(-1)])
            elif self.cfg.model.type == 'lstm':
                vector = torch.cat([models[i][1]['decoder.weight'].view(-1), models[i][1]['decoder.bias'].view(-1)])
            else :
                vector = torch.cat([models[i][1]['fc1.weight'].view(-1), models[i][1]['fc1.bias'].view(-1),
                                    models[i][1]['fc2.weight'].view(-1), models[i][1]['fc2.bias'].view(-1)])

            vector_deep.append(vector)

        T = vector_deep[0].clone()
        for v in vector_deep[1:]:
            T += v

        for i in range(model_num):
            score = torch.tensor([0.])
            for j in range(model_num):
                if j != i:
                    #score += torch.cosine_similarity(vector_deep[i].unsqueeze(0), vector_deep[j].unsqueeze(0))
                    score += torch.dot(vector_deep[i],vector_deep[j])
            in_sim.append(-score / (model_num - 1))
        #将列表in_sim里的0维张量以数值的形式放到列表value里
        for s in in_sim:
            value.append(s.item())

        for i in range(model_num):
            dot_product = torch.dot(T/model_num, vector_deep[i])
            #列表result是一个包含0维张量的列表
            result.append(dot_product)
        #把列表result里的0维张量数值取出来放到result_value列表里
        for tensor in result:
            result_value.append(tensor.item())

        #learning utility
        learning_utility = [x + y for x, y in zip(result_value , value)]

        round_latency = [1 / (i + 1) for i in staleness]
        U = [a * b for a, b in zip(learning_utility, round_latency)]
        sorted_U,list1,list1_values,list2 = [],[],[],[]

       # 需要筛选的设备数量/
        filter_num = round(self.cfg.usrsele.factor * len_distr)
        # 首先对原列表进行排序，并保持索引
        sorted_U = sorted(enumerate(U), key=lambda x: x[1], reverse=True)
        sort_idx = [index for index, value in sorted_U]
        sort_value = [value for index, value in sorted_U]
        return sort_idx, sort_value

    def _client_select_ours(self, distr_tensor, len_distr, len_class, sort_idx):
        # 需要筛选的分布数量
        filter_num = round(self.cfg.usrsele.factor * len_distr)
        # 当使用user select 并且在第一回合时，初始化history population 为零向量
        if self.global_num == 0:
            self.cfg.usrsele.his_popu = [0.] * len_class

        #学习率衰减
        #if self.global_num > 0 and self.global_num % 100 == 0:
           #self.cfg.train.optimizer.lr=0.5 * self.cfg.train.optimizer.lr

        strategy = self.cfg.usrsele.strategy
        his_popu = torch.tensor(self.cfg.usrsele.his_popu)
        if strategy == 'fast':
            # 将当前用户分布与历史累积使用的分布进行求和
            distr_tensor_accu = (his_popu + distr_tensor) / 2
            distr_tensor_accu = distr_tensor_accu / distr_tensor_accu.sum(1)[:,np.newaxis]
            # 计算分布列表中每个用户分布（行）与均匀分布的差异
            distr_tensor_diff = distr_tensor_accu - torch.ones_like(distr_tensor_accu) / len_class
            # 计算每行(norm(2,1)中的1)与均匀分布差异的L2范数并升序ascend排列
            # 首先对原列表进行排序，并保持索引
            sorted_U = sorted(enumerate(distr_tensor_diff.norm(2, 1)), key=lambda x: x[1], reverse=False)
            index_as = [index for index, value in sorted_U]
            common_set = set(sort_idx[:filter_num]) & set(index_as[:filter_num])
            union_set = set(sort_idx[:filter_num]) | set(index_as[:filter_num])
            common_idx = list(common_set)
            if len(common_set) < filter_num:
                remain_idx = random.sample(list(union_set - common_set), filter_num - len(common_set))
                filter_idx = list(common_set | set(remain_idx))
            else:
                filter_idx = common_idx
            print('aaaaaaaaaaAAAAAAAA',sort_idx[:filter_num], filter_idx)
            # 筛选出filter_num个前差异最小的分布
            filter_distri = distr_tensor[filter_idx]
            his_popu = (his_popu + filter_distri.mean(0)) / 2
            self.cfg.usrsele.his_popu = his_popu.tolist()
            return filter_idx, common_idx, sort_idx[:filter_num]

        # 组合筛选方案，逐一计算组合与均匀分布的差异，挑选出L2范数最小的组合
        elif strategy == 'combi':
            # 遍历所有可能的组合数，共C_n^k，其中n=len_distr，k=filter_num
            threshold = self.cfg.usrsele.combi_thre
            distr_tensor_accu = (his_popu + distr_tensor) / 2
            # distr_tensor_accu = distr_tensor_accu / distr_tensor_accu.sum(1)[:, np.newaxis]
            distr_uniform = torch.ones_like(distr_tensor_accu) / len_class
            distr_tensor_diff = distr_tensor_accu - distr_uniform
            combinations = itertools.combinations(range(len_distr),filter_num)
            for num, combi in enumerate(combinations):
                if num>100:
                    break
                can_distri = distr_tensor_diff[list(combi)]
                if can_distri.norm() <= threshold:
                    threshold = can_distri.norm().item()
                    index_as = combi
                    filter_distri = distr_tensor[list(combi)]
            his_popu = (his_popu + filter_distri.mean(0)) / 2
            self.cfg.usrsele.his_popu = his_popu.tolist()
            common_set = set(sort_idx[:filter_num]) & set(index_as)
            union_set = set(sort_idx[:filter_num]) | (set(range(len_distr))-set(index_as))
            common_idx = list(common_set)
            if len(common_set) < filter_num:
                remain_idx = random.sample(list(union_set - common_set), filter_num - len(common_set))
            filter_idx = list(common_set | set(remain_idx))
            return filter_idx, common_idx, sort_idx[:filter_num]

        elif strategy == 'coor_group':
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
                distr_slice_diff_mean = distr_slice_diff[:i + 1].mean(0)
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
        else:
            raise ValueError('check, no such strategy')

    def _client_select_comp_schandagg(self, distr_tensor, len_distr, len_class):
        threshold = len_class
        for filter_num in range(len_distr):
            combinations = itertools.combinations(range(len_distr), filter_num+1)
            for num, combi in enumerate(combinations):
                if num >= 100:
                    break
                # 用选取的标签分布减去平均分布，平均分布的定义为总分布和（combi的长度）除以标签数量
                diff = sum(distr_tensor[list(combi)], 0)/len(combi) - 1 / len_class
                var = sum(diff**2)
                if var <= threshold:
                    threshold = var
                    filter_idx = combi
        return filter_idx

    def _client_select_comp_freandagg(self, client_id, len_distr):
        if self.global_num == 0:
            self.cfg.usrsele_compa_freqandage.frelist = [0] * self.cfg.federate.client_num
        frelist = self.cfg.usrsele_compa_freqandage.frelist
        frelist_slice = [frelist[x-1] for x in client_id] #原来的id是从1开始计数，改为从0开始
        sorted_zip = sorted(zip(frelist_slice, client_id))
        _, sorted_id = map(list, zip(*sorted_zip))
        filter_num = round(len_distr * self.cfg.usrsele_compa_freqandage.factor)
        filter_idx = sorted_id[:filter_num]
        frelist_update = [frelist[x-1] + 1 if x in filter_idx else frelist[x-1] for x in range(1, self.cfg.federate.client_num+1)]
        # print('frelist is ', frelist, '\n' 'filter_idx is ', filter_idx, '\n' 'frelist_update is', frelist_update)
        self.cfg.usrsele_compa_freqandage.frelist = frelist_update
        return filter_idx

    def _client_select_comp_aouprio(self, staleness, distrs):
        shanno_value = [np.sum([-p*np.log(p) for p in value]) for value in distrs]
        filter_num = round(self.cfg.usrsele_compa_aouprior.factor * len(staleness))
        staleness = [value/(max(staleness)+10**(-5)) for value in staleness]
        shanno_value = [value/max(shanno_value) for value in shanno_value]
        prior_list = [0.5*shanno_value[i] + 0.5*staleness[i] for i in range(len(staleness))]
        sorted_U = sorted(enumerate(prior_list), key=lambda x: x[1], reverse=True)
        index_as = [index for index, value in sorted_U]
        filter_idx = index_as[:filter_num]
        return filter_idx

    def _compa_wkafl_collect_gradient(self, Collect_Gradients, K_Gradients, Clip=True):
        """
        WKAFL梯度聚合函数 - 支持state_dict字典格式

        Args:
            Collect_Gradients: dict, state_dict格式的梯度字典 {layer_name: tensor}
            K_Gradients: list, 包含K个元组的列表，每个元组为(user_id, gradient_dict)
            Layers_shape: dict, 层形状字典 {layer_name: shape}
            Clip: bool, 是否进行梯度裁剪

        Returns:
            dict: 聚合后的梯度字典
        """
        K = len(K_Gradients)
        sim = torch.zeros([K])

        # 计算每个客户端梯度的范数
        Standnorm = self.L_norm_dict(Collect_Gradients)

        # 计算相似度
        for i in range(K):
            sim[i] = self.similarity_dict(K_Gradients[i][1], Collect_Gradients)
        index = (sim > self.cfg.usrsele_compa_wkafl.threshold)

        # 如果没有相似度足够高的梯度，直接返回
        if sum(index) == 0:
            print("相似度均较低")
            return Collect_Gradients

        # 选择相似度高的梯度进行聚合
        totalSim = []
        Sel_Gradients = []
        for i in range(K):
            if sim[i] > self.cfg.usrsele_compa_wkafl.threshold:
                totalSim.append((torch.exp(sim[i] * 20)).tolist())
                Sel_Gradients.append(K_Gradients[i])
        # # 加权聚合梯度
        # for i in range(len(totalSim)):
        #     Gradients_Sample = Sel_Gradients[i]
        #     if Clip:
        #         standNorm = self.L_norm_dict(Collect_Gradients)
        #         # Gradients_Sample = self.TensorClip_dict(Gradients_Sample, self.cfg.usrsele_compa_wkafl.cb2 * standNorm)
        #         Gradients_Sample = self.clip_gradients(Gradients_Sample, self.cfg.usrsele_compa_wkafl.cb2 * standNorm)
        #
        #     # 对每一层进行加权累加
        #     for layer_name in Collect_Gradients.keys():
        #         if layer_name in Gradients_Sample:
        #             if i == 0:
        #                 avg_gradients[layer_name] = Gradients_Sample[layer_name] * totalSim[i]
        #             else:
        #                 avg_gradients[layer_name] += Gradients_Sample[layer_name] * totalSim[i]
        return Sel_Gradients

    # def _compa_wkafl_collect_gradient(self, Collect_Gradients, K_Gradients, Layers_shape, Clip=False):
    #     K = len(K_Gradients)
    #     sim = torch.zeros([K])
    #     Gradients_Total = torch.zeros([K + 1])
    #     for i in range(K):
    #         Gradients_Total[i] = self.L_norm(K_Gradients[i][1])
    #     Gradients_Total[K] = self.L_norm(Collect_Gradients)
    #     # print('Gradients_norm', Gradients_Total)
    #     for i in range(K):
    #         sim[i] = self.similarity(K_Gradients[i][1], Collect_Gradients)
    #     index = (sim > self.cfg.usrsele_compa_wkafl.threshold)
    #     # print('sim:', sim)
    #     if sum(index) == 0:
    #         print("相似度均较低")
    #         return Collect_Gradients
    #     Collect_Gradients = self.ZerosGradients(Layers_shape)
    #
    #     totalSim = []
    #     Sel_Gradients = []
    #     for i in range(K):
    #         if sim[i] > self.cfg.usrsele_compa_wkafl.threshold:
    #             totalSim.append((torch.exp(sim[i] * 20)).tolist())
    #             Sel_Gradients.append(K_Gradients[i][1])
    #     totalSim = torch.tensor(totalSim)
    #     totalSim = totalSim / torch.sum(totalSim)
    #     for i in range(len(totalSim)):
    #         Gradients_Sample = Sel_Gradients[i]
    #         if Clip:
    #             standNorm = self.L_norm(Collect_Gradients)
    #             Gradients_Sample = self.TensorClip(Gradients_Sample, self.cfg.usrsele_compa_wkafl.cb2 * standNorm)
    #         for j in range(len(K_Gradients[i][1])):
    #             Collect_Gradients[j] += Gradients_Sample[j] * totalSim[i]
    #     return Collect_Gradients

    ##################################获取模型层数和各层的形状#############
    def getModelLayers(self,model):
        layers_shape = {}
        for layer_name, layer_values in model.items():
            layers_shape[layer_name] = layer_values.shape
        return layers_shape

    ##################################设置各层的梯度为0#####################
    def ZerosGradients_dict(self, layers_shape_dict):
        """创建state_dict格式的零梯度字典"""
        zero_gradients = {}
        for layer_name, shape in layers_shape_dict.items():
            zero_gradients[layer_name] = torch.zeros(shape)
        return zero_gradients

    ################################调整学习率###############################
    def lr_adjust(self, tau):
        tau = 0.05 * tau + 1
        lr = self.cfg.train.optimizer.lr / tau
        return lr

    #################################计算范数################################
    def L_norm_dict(self, gradient_dict):
        """计算state_dict格式梯度的L2范数"""
        norm_tensor = torch.tensor([0.])
        for layer_name, tensor in gradient_dict.items():
            norm_tensor += tensor.float().norm() ** 2
        return norm_tensor.sqrt()

    ################################# 计算角相似度 ############################
    def similarity_dict(self, user_gradients_dict, yun_gradients_dict):
        """计算两个state_dict格式梯度的余弦相似度"""
        sim = torch.tensor([0.])
        for layer_name in user_gradients_dict.keys():
            if layer_name in yun_gradients_dict:
                sim = sim + torch.sum(user_gradients_dict[layer_name] * yun_gradients_dict[layer_name])

        user_norm = self.L_norm_dict(user_gradients_dict)
        yun_norm = self.L_norm_dict(yun_gradients_dict)

        if user_norm == 0:
            print('梯度为0.')
            sim = torch.tensor([1.])
            return sim

        sim = sim / (user_norm * yun_norm)
        return sim

    def clip_gradients(self,gradients, max_norm):
        """
        对 OrderedDict 中的梯度进行裁剪

        参数:
            gradients: OrderedDict，键为层名，值为该层的梯度张量
            max_norm: 梯度裁剪的阈值（最大大允许范数）
        返回:
            OrderedDict，裁剪裁剪后的梯度（保持原键顺序）
        """
        if not gradients:
            return gradients  # 空字典直接返回

        # 1. 收集所有梯度并展平，计算总 L2 范数
        all_grads = []
        for grad in gradients.values():
            # 将每个层梯度展平为一维维张量，便于拼接
            all_grads.append(grad.view(-1))
        # 拼接所有梯度为一个大向量
        combined = torch.cat(all_grads)
        # 计算总范数
        total_norm = torch.norm(combined, p=2)  # L2 范数

        # 2. 若总范数超过阈值阈值，则按比例缩放所有梯度
        if total_norm > max_norm:
            # 计算缩放因子
            scale = max_norm / (total_norm + 1e-10)  # 加小值避免除零
            # 缩放每个梯度并保留 OrderedDict 类型
            clipped_grads = OrderedDict()
            for key, grad in gradients.items():
                clipped_grads[key] = grad * scale
            return clipped_grads
        else:
            # 范数未超阈值，直接返回原梯度（仍为 OrderedDict）
            return gradients

    def TensorClip_dict(self, gradient_dict, clip_bound):
        """对state_dict格式的梯度进行裁剪"""
        norm_tensor = self.L_norm_dict(gradient_dict)
        clipped_gradients = {}

        if clip_bound < norm_tensor:
            clip_ratio = clip_bound / norm_tensor
            for layer_name, tensor in gradient_dict.items():
                clipped_gradients[layer_name] = tensor * clip_ratio
        else:
            # 如果不需要裁剪，直接复制
            for layer_name, tensor in gradient_dict.items():
                clipped_gradients[layer_name] = tensor.clone()

        return clipped_gradients
