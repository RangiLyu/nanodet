from nanodet.model.arch import build_model
from nanodet.util import cfg, get_model_complexity_info, load_config


def test_flops():
    load_config(cfg, "./config/legacy_v0.x_configs/nanodet-m.yml")

    model = build_model(cfg.model)
    input_shape = (3, 320, 320)
    get_model_complexity_info(model, input_shape)
