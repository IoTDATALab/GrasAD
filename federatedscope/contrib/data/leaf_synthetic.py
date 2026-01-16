from federatedscope.register import register_data
from federatedscope.core.data import BaseDataTranslator
from federatedscope.data.synthetic.load_synthetic_data import load_synthetic_data

def call_my_data(config, file_paths):
    if config.data.type == "leaf_synthetic":
        synthetic = load_synthetic_data()
        translator = BaseDataTranslator(config)
        fs_synthetic = translator(synthetic)
        return fs_synthetic, config

register_data("leaf_synthetic", call_my_data)
