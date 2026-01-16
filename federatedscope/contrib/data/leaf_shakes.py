from federatedscope.register import register_data
from federatedscope.core.data import BaseDataTranslator
from federatedscope.data.shakespeare.load_shakes_data import load_shakespeare_data

def call_my_data(config, file_paths):
    if config.data.type == "leaf_shakes":
        shakes = load_shakespeare_data()
        translator = BaseDataTranslator(config)
        fs_shakes = translator(shakes)
        return fs_shakes, config

register_data("leaf_shakes", call_my_data)
