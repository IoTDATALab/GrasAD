from federatedscope.register import register_data
from federatedscope.core.data import BaseDataTranslator
from federatedscope.data.celeba.load_celeba_data import load_celeba_data

def call_my_data(config, file_paths):
    if config.data.type == "leaf_celeba":
        celeba = load_celeba_data()
        translator = BaseDataTranslator(config)
        fs_celeba = translator(celeba)
        return fs_celeba, config

register_data("leaf_celeba", call_my_data)
