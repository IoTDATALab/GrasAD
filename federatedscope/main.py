import os
import sys

DEV_MODE = True  # simplify the federatedscope re-setup everytime we change
# the source codes of federatedscope
os.environ["WANDB_API_KEY"] = '520da4246c7917e33fd708435ce9642fecf3c68f'
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)

from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, \
    get_server_cls
from federatedscope.core.configs.config import global_cfg, CfgNode
from federatedscope.core.auxiliaries.runner_builder import get_runner

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

if __name__ == '__main__':
    init_cfg = global_cfg.clone()
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # load clients' cfg file
    if args.client_cfg_file:
        client_cfgs = CfgNode.load_cfg(open(args.client_cfg_file, 'r'))
        # client_cfgs.set_new_allowed(True)
        client_cfgs.merge_from_list(client_cfg_opt)
    else:
        client_cfgs = None

    # federated dataset might change the number of clients
    # thus, we allow the creation procedure of dataset to modify the global
    # cfg object
    data, modified_cfg = get_data(config=init_cfg.clone(),
                                  client_cfgs=client_cfgs)
    init_cfg.merge_from_other_cfg(modified_cfg)

    # 生成每个用户不同的batch_size的配置文件yaml，其中0.2表示从用户中选取20%数据作为batch_size
    # train_num = []
    # for i in range(1,len(data)):
    #     train_num.append(len(list(data[i].train_data)))
    # with open('/home/pkl2/lyn/FederatedScope/scripts/pami_camparison_algorithms/users_dataset_size.yaml', 'a+') as fl:
    #     for i in range(len(train_num)):
    #         fl.write('client_{}:\n  dataloader:\n    batch_size: {}\n'.format(i+1, int(numpy.ceil(0.2* train_num[i]))))
    # fl.close()
    init_cfg.freeze()

    runner = get_runner(data=data,
                        server_class=get_server_cls(init_cfg),
                        client_class=get_client_cls(init_cfg),
                        config=init_cfg.clone(),
                        client_configs=client_cfgs)
    _ = runner.run()
