from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_dp_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # nbafl(dp) related options
    # ---------------------------------------------------------------------- #
    cfg.nbafl = CN()

    # Params
    cfg.nbafl.use = False
    cfg.nbafl.mu = 0.
    cfg.nbafl.epsilon = 100.
    cfg.nbafl.w_clip = 1.
    cfg.nbafl.constant = 30.

    # ---------------------------------------------------------------------- #
    # VFL-SGDMF(dp) related options
    # ---------------------------------------------------------------------- #
    cfg.sgdmf = CN()

    cfg.sgdmf.use = False  # if use sgdmf algorithm
    cfg.sgdmf.R = 5.  # The upper bound of rating
    cfg.sgdmf.epsilon = 4.  # \epsilon in dp
    cfg.sgdmf.delta = 0.5  # \delta in dp
    cfg.sgdmf.constant = 1.  # constant

    # ---------------------------------------------------------------------- #
    # MAPA(dp) related options lyn
    # ---------------------------------------------------------------------- #
    cfg.mapa = CN()

    cfg.mapa.use = False  # if use mapa parameters lyn
    cfg.mapa.grad_var = 5.  # The variance of gradient
    cfg.mapa.clipbound = 1.  # estimation of gradient's upper bound
    cfg.mapa.redcu_ratio = 0.8  # reduction ratio of clipping bound
    cfg.mapa.L_smooth = 0.0004  # smooth constant of gradient
    cfg.mapa.inital_loss_bound = 1000. # estimation of initial loss function
    cfg.mapa.n2s = 0.8 # noise to signal ratio
    cfg.mapa.Stage_itrs = 0 # 阶段迭代次数
    cfg.mapa.Stage_lr = 0.1 #初始学习率
    cfg.mapa.style = 'sample' # client-level dp or sample-level dp
    cfg.mapa.lr = 0.01 # 学习率
    cfg.mapa.beta = 2. # dissimilarity between client distribution and population distribution

    # ---------------------------------------------------------------------- #
    # Adaclip1(dp) related options lyn
    # ---------------------------------------------------------------------- #
    cfg.adaclip1 = CN()

    cfg.adaclip1.use = False  # if use mapa parameters lyn
    cfg.adaclip1.beta_1 = 0.99  # Averaging speed parameter for the mean of gradients
    cfg.adaclip1.beta_2 = 0.9  # Averaging speed parameter for the variance of gradients
    cfg.adaclip1.h_1 = 10 ** (-12)  # Parameter 1 for estimating stand variance
    cfg.adaclip1.h_2 = 10 ** (-10)  # Parameter 2 for estimating stand variance
    cfg.adaclip1.clipbound = 0.01 # threshold for clipping gradients
    cfg.adaclip1.n2s = 0.8 # noise to signal ratio
    cfg.adaclip1.M = [] # 中间参量，均值
    cfg.adaclip1.S = [] # 中间参量，标准差
    cfg.adaclip1.lr = 0.1 # 学习率
    # ---------------------------------------------------------------------- #
    # Adaclip2(dp) related options lyn
    # ---------------------------------------------------------------------- #
    cfg.adaclip2 = CN()

    cfg.adaclip2.use = False  # if use mapa parameters lyn
    cfg.adaclip2.quatile = 0.5  # target clipped gradient
    cfg.adaclip2.c = 0.5  # Split ratio of privacy budget, unused
    cfg.adaclip2.clipbound = 0.8  # threshold for clipping gradients
    cfg.adaclip2.n2s = 0.8 # noise to signal ratio
    cfg.adaclip2.clipbound_us = 'Lin' # clipbound update style,
                                      # lin (线性) or Geo (几何)
    cfg.adaclip2.lr = 0.01 # 学习率

    # ---------------------------------------------------------------------- #
    # Fixdp-s(dp) related options lyn
    # ---------------------------------------------------------------------- #
    cfg.fixdps = CN()

    cfg.fixdps.use = False  # if use mapa parameters lyn
    cfg.fixdps.clipbound = 0.8  # threshold for clipping gradients
    cfg.fixdps.n2s = 0.8 # noise to signal ratio
    cfg.fixdps.lr = 0.1 # 学习率

    # ---------------------------------------------------------------------- #
    # Fixdp-c(dp) related options lyn
    # ---------------------------------------------------------------------- #
    cfg.fixdpc = CN()

    cfg.fixdpc.use = False  # if use mapa parameters lyn
    cfg.fixdpc.clipbound = 0.8  # threshold for clipping gradients
    cfg.fixdpc.n2s = 0.8 # noise to signal ratio
    cfg.fixdpc.lr = 0.01 # 学习率

    # ---------------------------------------------------------------------- #
    # Median(dp) related options lyn
    # ---------------------------------------------------------------------- #
    cfg.median = CN()

    cfg.median.use = False  # if use mapa parameters lyn
    cfg.median.n2s = 0.8 # noise to signal ratio

    # --用户选择参量，包括历史数据分布，历史贡献---------------------------------- #
    cfg.usrsele = CN()

    cfg.usrsele.use = False  # if use user selection lyn
    cfg.usrsele.strategy = 'fast' # 3种策略，fast, combi(组合), coor_group(分组)
    cfg.usrsele.his_popu = []  # history population
    cfg.usrsele.his_grad = [0] # history gradient
    cfg.usrsele.factor = 0.5 # selecting factor
    cfg.usrsele.combi_thre = 62. # combi策略的初始阈值
    cfg.usrsele.ablation= 'none' # fast-s, fast-w, combi-s, combi-w, none

    # ---------------------------------------------------------------------- #
    # 用户选择对比算法：Async HFL，文章Async-HFL: Efficient and Robust Asynchronous
    # Federated Learning in Hierarchical IoT Networks
    # ---------------------------------------------------------------------- #
    cfg.usrsele_compa_hfl = CN()

    cfg.usrsele_compa_hfl.use = False
    cfg.usrsele_compa_hfl.factor = 0.5 # selecting factor

    # ---------------------------------------------------------------------- #
    # 用户选择对比算法：Async HFL，文章Scheduling and Aggregation Design for
    # Asynchronous Federated Learning over Wireless Networks
    # ---------------------------------------------------------------------- #
    cfg.usrsele_compa_schandagg = CN()

    cfg.usrsele_compa_schandagg.use = False
    cfg.usrsele_compa_schandagg.factor = 0.5 # selecting factor
    cfg.usrsele_compa_schandagg.gamma = 1.5 # 大于1表示偏向旧本地模型，小于1表示偏向新本地模型

    # ---------------------------------------------------------------------- #
    # 用户选择对比算法：Async kAFL，文章KAFL: Achieving High Training Efficiency
    # for Fast-K Asynchronous Federated Learning
    # ---------------------------------------------------------------------- #
    cfg.usrsele_compa_kafl = CN()
    cfg.usrsele_compa_kafl.use = False
    cfg.usrsele_compa_kafl.rho = 0.5
    cfg.usrsele_compa_kafl.global_model = [0]
    cfg.usrsele_compa_kafl.frelist = []

    # ---------------------------------------------------------------------- #
    # 用户选择对比算法：Async freqandage，文章Device Scheduling and Update
    # Aggregation Policies for Asynchronous Federated Learning中的按频率策略
    # ---------------------------------------------------------------------- #
    cfg.usrsele_compa_freqandage = CN()
    cfg.usrsele_compa_freqandage.use = False
    cfg.usrsele_compa_freqandage.frelist = []
    cfg.usrsele_compa_freqandage.factor = 0.5
    cfg.usrsele_compa_freqandage.gamma = 1.5  # 大于1表示偏向旧本地模型，小于1表示偏向新本地模型

    # ---------------------------------------------------------------------- #
    # 用户选择对比算法：AoUPrior，文章AoU-based local update and
    # user scheduling for semi-asynchronous online federated learning
    # in wireless networks（IEEE internet of things journalpp）中的按过时性和香农熵策略
    # ---------------------------------------------------------------------- #
    cfg.usrsele_compa_aouprior = CN()
    cfg.usrsele_compa_aouprior.use = False
    cfg.usrsele_compa_aouprior.factor = 0.5

    # ---------------------------------------------------------------------- #
    # 用户选择对比算法：WKAFL，文章XXXXXX
    # user scheduling for semi-asynchronous online federated learning
    # in wireless networks（IEEE TPDS）中的按过时性和香农熵策略
    # ---------------------------------------------------------------------- #
    cfg.usrsele_compa_wkafl = CN()
    cfg.usrsele_compa_wkafl.use = False
    cfg.usrsele_compa_wkafl.cb1 = 10
    cfg.usrsele_compa_wkafl.cb2 = 4
    cfg.usrsele_compa_wkafl.threshold = 0.3


    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_dp_cfg)

def assert_dp_cfg(cfg):
    pass

register_config("dp", extend_dp_cfg)