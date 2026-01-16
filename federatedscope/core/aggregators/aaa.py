import copy, itertools
import torch
import numpy as np
from federatedscope.core.aggregators import ClientsAvgAggregator
from federatedscope.core.trainers.utils import get_random


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

        if self.cfg.usrsele.use + self.cfg.usrsele_compa_schandagg.use \
                + self.cfg.usrsele_compa_freqandage.use + self.cfg.usrsele_compa_hfl.use >= 1:
            models_plus_dist = agg_info["client_feedback"]  # 接收到用户发送的消息，包含训练集大小，模型参量，训练集分布
            models = [models_plus_dist[i][0:2] for i in range(len(models_plus_dist))]  # 分离出模型
            dists_ratio = [models_plus_dist[i][-1] for i in range(len(models_plus_dist))]  # 分离出训练集分布
            client_aggnum = np.array([user[0] for user in models_plus_dist])  # 提取用户上传的样本数量，将列表转为数组
            dists_num = np.array(dists_ratio) * client_aggnum[:, np.newaxis]  # dists_ratio中每一行乘以对应的样本数量
            dists = dists_num / (dists_num + 10 ** (-6)).sum(0)  # 按列归一化
            dists = dists / dists.sum(1)[:, np.newaxis]  # 按列归一化后再按行归一化
            if self.cfg.data.type == 'celeba' or self.cfg.data.type == 'synthetic':
                dists = dists[:,0:2]
        else:
            models = agg_info["client_feedback"]

        recover_fun = agg_info['recover_fun'] if (
                'recover_fun' in agg_info and self.cfg.federate.use_ss) else None
        staleness = [x[1]
                     for x in agg_info['staleness']]  # (client_id, staleness)
        client_id = [x[0]
                     for x in agg_info['staleness']]
        id_staleness = agg_info['staleness']
        # print(staleness)
        self.global_num = agg_info['global_num']

        # 判断是否只使用一个算法
        dp_algorithm_judge = self.cfg.usrsele.use + self.cfg.usrsele_compa_hfl.use +\
                             self.cfg.usrsele_compa_schandagg.use + self.cfg.usrsele_compa_freqandage.use + \
                             self.cfg.usrsele_compa_kafl.use
        if dp_algorithm_judge > 1:
            raise ValueError(
                'check, two or more algorithms are used from uersele, usrsele_compa_hfl, '
                'usrsele_compa_schandagg, usrsele_compa_freqandage, usrsele_compa_kafl')

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
                weight_list = self._weight_func_ours(models, staleness)
                avg_model = self._para_weighted_avg(models,
                                                    recover_fun=recover_fun,
                                                    staleness=staleness,
                                                    weights=weight_list)
            elif self.cfg.usrsele.ablation == 'fast-w' or self.cfg.usrsele.ablation == 'combi-w':
                filtered_clients = self._distributions_filter(models, staleness, client_id, distr=dists)
                # print("第{}轮次筛选出用户为{}".format(self.global_num, filtered_clients))
                models = [models[i] for i in filtered_clients]
                print('AAAAAAAAAAAAAAAAAA', filtered_clients)
                staleness = [staleness[i] for i in filtered_clients]
                avg_model = self._para_weighted_avg(models,
                                                    recover_fun=recover_fun,
                                                    staleness=staleness)
            else:
                # 筛选出符合条件的filtered clients
                filtered_clients = self._distributions_filter(models, staleness, client_id, distr=dists)
                # print("第{}轮次筛选出用户为{}".format(self.global_num, filtered_clients))
                models = [models[i] for i in filtered_clients]
                staleness = [staleness[i] for i in filtered_clients]
                weight_list = self._weight_func_ours(models, staleness)
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
            # print('EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE, the weight is ', weights)
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
            _, filter_idx = self._client_select_comp_hfl(models, staleness, client_id, distr)
            return filter_idx
        elif self.cfg.usrsele_compa_kafl.use == True:
            print('usrsele_compa_kafl')
        elif self.cfg.usrsele_compa_freqandage.use == True:
            print('usrsele_compa_freqandage')
            filter_idx = self._client_select_comp_freandagg(client_id, len_distr)
            return filter_idx
        elif self.cfg.usrsele.use == True:
            print('usrsele')
            filter_idx = self._client_select_ours(distr_tensor, len_distr, len_class)
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

    def _weight_func_ours(self, models, staleness):
        training_set_size, training_set_size_inv = 0, 0
        for i in range(len(models)):
            sample_size, _ = models[i]
            training_set_size += sample_size
            training_set_size_inv += 1/sample_size
        weights_initial1 = [model[0]/training_set_size for model in models]
        weights_initial2 = [(1/model[0])/training_set_size_inv for model in models]
        if self.global_num == 0:
            weight_list = weights_initial1
            # weight = []
            # for i in range(len(models)):
            #     sample_size, _ = models[i]
            #     if self.cfg.federate.ignore_weight:
            #         weight.append(1.0 / len(models))
            #     else:
            #         weight.append((1/sample_size) / training_set_size_inv)
        else:
            vector_deep, out_sim, in_sim, weight = [], [], [], []
            model_num, stand_grad = len(models), self.cfg.usrsele.his_grad[0]
            if self.cfg.model.type == 'lr':
                stand_grad = torch.cat([stand_grad['fc.weight'].view(-1), stand_grad['fc.bias'].view(-1)])
            elif self.cfg.model.type == 'convnet2':
                stand_grad = torch.cat([stand_grad['fc1.weight'].view(-1), stand_grad['fc1.bias'].view(-1),
                                        stand_grad['fc2.weight'].view(-1), stand_grad['fc2.bias'].view(-1)])
            else:
                stand_grad = torch.cat([stand_grad['fc2.weight'].view(-1), stand_grad['fc2.bias'].view(-1),
                                        stand_grad['fc3.weight'].view(-1), stand_grad['fc3.bias'].view(-1)])

            for i, model in enumerate(models):
                if self.cfg.model.type == 'lr':
                    vector = torch.cat([model[1]['fc.weight'].view(-1), model[1]['fc.bias'].view(-1)])
                elif self.cfg.model.type == 'convnet2':
                    vector = torch.cat([model[1]['fc1.weight'].view(-1), model[1]['fc1.bias'].view(-1),
                                        model[1]['fc2.weight'].view(-1), model[1]['fc2.bias'].view(-1)])
                else:
                    vector = torch.cat([model[1]['fc2.weight'].view(-1), model[1]['fc2.bias'].view(-1),
                                       model[1]['fc3.weight'].view(-1), model[1]['fc3.bias'].view(-1)])
                sim = torch.cosine_similarity(vector.unsqueeze(0), stand_grad.unsqueeze(0))
                out_sim.append(sim)
                # out_sim.append(0) if sim < 0 else out_sim.append(sim)
                # vector_deep.append(vector)

            # for i in range(model_num):
            #     score = torch.tensor([0.])
            #     for j in range(model_num):
            #         if j != i:
            #             cosine_sim = torch.cosine_similarity(vector_deep[i].unsqueeze(0), vector_deep[j].unsqueeze(0))
            #             # exp_cosine_sim = torch.exp(cosine_sim)
            #             # score += exp_cosine_sim
            #             score += cosine_sim
            #             #score += torch.cosine_similarity(vector_deep[i].unsqueeze(0), vector_deep[j].unsqueeze(0))
            #     in_sim.append(score / (model_num - 1) if model_num > 1 else score)
            # print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',out_sim, in_sim)
            for i in range(model_num):
                if self.global_num <= 2/3 * self.cfg.federate.total_round_num:
                    weight.append(torch.exp(out_sim[i]) * weights_initial1[i])
                else:
                    weight.append(torch.exp(out_sim[i]) * weights_initial2[i])
                # weight.append(torch.exp((out_sim[i]+in_sim[i]) * weights_initial[i]))
                # weight.append((0.3 * torch.exp(out_sim[i]) + 0.7 * torch.exp(in_sim[i])) * weights_initial[i])
                # weight.append(0.3 * out_sim[i] + 0.7 * (score / (model_num - 1)))
            weight_list = [value / sum(weight) for value in weight]
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
            elif self.cfg.model.type == 'convnet2':
                global_model = torch.cat([stand_grad['fc1.weight'].view(-1), stand_grad['fc1.bias'].view(-1),
                                        stand_grad['fc2.weight'].view(-1), stand_grad['fc2.bias'].view(-1)])
            else:
                global_model = torch.cat([stand_grad['fc2.weight'].view(-1), stand_grad['fc2.bias'].view(-1),
                                        stand_grad['fc3.weight'].view(-1), stand_grad['fc3.bias'].view(-1)])

            lambda_deno = torch.dot(global_model, global_model)
            for i, model in enumerate(models):
                if self.cfg.model.type == 'lr':
                    vector = torch.cat([stand_grad['fc.weight'].view(-1), stand_grad['fc.bias'].view(-1)])
                elif self.cfg.model.type == 'convnet2':
                    vector = torch.cat([stand_grad['fc1.weight'].view(-1), stand_grad['fc1.bias'].view(-1),
                                              stand_grad['fc2.weight'].view(-1), stand_grad['fc2.bias'].view(-1)])
                else:
                    vector = torch.cat([stand_grad['fc2.weight'].view(-1), stand_grad['fc2.bias'].view(-1),
                                        stand_grad['fc3.weight'].view(-1), stand_grad['fc3.bias'].view(-1)])

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
        list1_values = [value for index, value in list1]
        list2 = [index for index, value in list1] #list2指从0开始编号的下标
        list3 = [client_id[index] for index in list2] #list3指用户的id
        return list2, list3

    def _client_select_ours(self, distr_tensor, len_distr, len_class):
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
            _, index_as = distr_tensor_diff.norm(2, 1).sort()
            # 筛选出filter_num个前差异最小的分布
            filter_distri = distr_tensor[index_as[:filter_num]]
            his_popu = (his_popu + filter_distri.mean(0)) / 2
            self.cfg.usrsele.his_popu = his_popu.tolist()
            return index_as[:filter_num]

        # 组合筛选方案，逐一计算组合与均匀分布的差异，挑选出L2范数最小的组合
        elif strategy == 'combi':
            # 遍历所有可能的组合数，共C_n^k，其中n=len_distr，k=filter_num
            threshold = self.cfg.usrsele.combi_thre
            fiter_index = []
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
                    fiter_index = combi
                    filter_distri = distr_tensor[list(combi)]
            his_popu = (his_popu + filter_distri.mean(0)) / 2
            self.cfg.usrsele.his_popu = his_popu.tolist()
            return fiter_index

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
        print('frelist is ', frelist, '\n' 'filter_idx is ', filter_idx, '\n' 'frelist_update is', frelist_update)
        self.cfg.usrsele_compa_freqandage.frelist = frelist_update
        return filter_idx