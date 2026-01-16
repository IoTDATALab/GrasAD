import numpy as np


def _split_according_to_prior(label, client_num, prior):
    assert client_num == len(prior)
    classes = len(np.unique(label))
    assert classes == len(np.unique(np.concatenate(prior, 0)))

    # counting
    frequency = np.zeros(shape=(client_num, classes))
    for idx, client_prior in enumerate(prior):
        for each in client_prior:
            frequency[idx][each] += 1
    sum_frequency = np.sum(frequency, axis=0)

    idx_slice = [[] for _ in range(client_num)]
    for k in range(classes):
        idx_k = np.where(label == k)[0]
        np.random.shuffle(idx_k)
        nums_k = np.ceil(frequency[:, k] / sum_frequency[k] *
                         len(idx_k)).astype(int)
        while len(idx_k) < np.sum(nums_k):
            random_client = np.random.choice(range(client_num))
            if nums_k[random_client] > 0:
                nums_k[random_client] -= 1
        assert len(idx_k) == np.sum(nums_k)
        idx_slice = [
            idx_j + idx.tolist() for idx_j, idx in zip(
                idx_slice, np.split(idx_k,
                                    np.cumsum(nums_k)[:-1]))
        ]

    for i in range(len(idx_slice)):
        np.random.shuffle(idx_slice[i])
    return idx_slice


def dirichlet_distribution_noniid_slice(label,
                                        client_num,
                                        alpha,
                                        min_size=1,
                                        prior=None):
    r"""Get sample index list for each client from the Dirichlet distribution.
    https://github.com/FedML-AI/FedML/blob/master/fedml_core/non_iid
    partition/noniid_partition.py

    Arguments:
        label (np.array): Label list to be split.
        client_num (int): Split label into client_num parts.
        alpha (float): alpha of LDA.
        min_size (int): min number of sample in each client
    Returns:
        idx_slice (List): List of splited label index slice.
    """
    if len(label.shape) != 1:
        raise ValueError('Only support single-label tasks!')

    if prior is not None:
        return _split_according_to_prior(label, client_num, prior)

    num = len(label)
    classes = len(np.unique(label))
    assert num > client_num * min_size, f'The number of sample should be ' \
                                        f'greater than' \
                                        f' {client_num * min_size}.'
    size = 0
    while size < min_size:
        idx_slice = [[] for _ in range(client_num)]
        for k in range(classes):
            # for label k
            idx_k = np.where(label == k)[0]
            np.random.shuffle(idx_k)
            prop = np.random.dirichlet(np.repeat(alpha, client_num))
            prop = (np.cumsum(prop) * len(idx_k)).astype(int)[:-1]
            idx_slice = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_slice, np.split(idx_k, prop))
            ]

            # 检查是否是最后一个类别
            if k == classes - 1:
                # 检查是否存在空元素
                empty_indices = [i for i, idx in enumerate(idx_slice) if len(idx) == 0]

                # 如果有空元素
                if empty_indices:
                    # 从后往前找有足够样本的客户端
                    donor_indices = []
                    for i in range(client_num - 1, -1, -1):
                        if i not in empty_indices and len(idx_slice[i]) >= 4:
                            donor_indices.append(i)
                            # 找到足够的供体即可停止
                            if len(donor_indices) >= len(empty_indices):
                                break

                    # 确保有足够的供体
                    if len(donor_indices) < len(empty_indices):
                        # 如果供体不足，继续下一轮循环
                        break

                    # 从供体转移样本到空客户端
                    for empty_idx, donor_idx in zip(empty_indices, donor_indices):
                        # 从供体随机选择2个样本
                        if len(idx_slice[donor_idx]) >= 4:
                            # 随机选择2个不同的索引
                            transfer_indices = np.random.choice(
                                len(idx_slice[donor_idx]), size=2, replace=False
                            )
                            # 提取要转移的样本（先排序再反向，确保按正确顺序移除）
                            sorted_indices = sorted(transfer_indices, reverse=True)
                            samples_to_transfer = [idx_slice[donor_idx][i] for i in sorted_indices]
                            # 从供体移除样本
                            for idx in sorted_indices:
                                if idx < len(idx_slice[donor_idx]):  # 确保索引有效
                                    idx_slice[donor_idx].pop(idx)
                            # 添加到空客户端
                            idx_slice[empty_idx].extend(samples_to_transfer)


        size = min([len(idx_j) for idx_j in idx_slice])
    for i in range(client_num):
        np.random.shuffle(idx_slice[i])
    return idx_slice
