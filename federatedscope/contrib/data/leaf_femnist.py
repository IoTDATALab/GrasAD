from federatedscope.register import register_data
from federatedscope.core.data import BaseDataTranslator
from federatedscope.data.femnist.load_femnist_data import load_femnist_data

def call_my_data(config, file_paths):
    if config.data.type == "leaf_femnist":
        femnist = load_femnist_data()
        translator = BaseDataTranslator(config)
        fs_femnist = translator(femnist)
        return fs_femnist, config

register_data("leaf_femnist", call_my_data)
