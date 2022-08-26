import pytest
import torch.nn as nn

from nanodet.model.module.norm import build_norm_layer


def test_build_norm_layer():
    with pytest.raises(AssertionError):
        # cfg must be a dict
        cfg = "BN"
        build_norm_layer(cfg, 3)

    with pytest.raises(AssertionError):
        # `type` must be in cfg
        cfg = dict()
        build_norm_layer(cfg, 3)

    with pytest.raises(KeyError):
        # unsupported norm type
        cfg = dict(type="FancyNorm")
        build_norm_layer(cfg, 3)

    with pytest.raises(AssertionError):
        # postfix must be int or str
        cfg = dict(type="BN")
        build_norm_layer(cfg, 3, postfix=[1, 2])

    with pytest.raises(AssertionError):
        # `num_groups` must be in cfg when using 'GN'
        cfg = dict(type="GN")
        build_norm_layer(cfg, 3)

    # test each type of norm layer in norm_cfg
    abbr_mapping = {
        "BN": "bn",
        "SyncBN": "bn",
        "GN": "gn",
    }
    module_dict = {
        "BN": nn.BatchNorm2d,
        "SyncBN": nn.SyncBatchNorm,
        "GN": nn.GroupNorm,
    }
    for type_name, module in module_dict.items():
        for postfix in ["_test", 1]:
            cfg = dict(type=type_name)
            if type_name == "GN":
                cfg["num_groups"] = 2
            name, layer = build_norm_layer(cfg, 4, postfix=postfix)
            assert name == abbr_mapping[type_name] + str(postfix)
            assert isinstance(layer, module)
            if type_name == "GN":
                assert layer.num_channels == 4
                assert layer.num_groups == cfg["num_groups"]
            elif type_name != "LN":
                assert layer.num_features == 4
