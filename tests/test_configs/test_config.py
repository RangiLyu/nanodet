# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from os.path import dirname, exists, join

from nanodet.model.arch import build_model
from nanodet.util import cfg, collect_files, load_config


def test_config_files():
    root_path = join(dirname(__file__), "../..")
    cfg_folder = join(root_path, "config")
    if not exists(cfg_folder):
        raise FileNotFoundError("Cannot find config folder.")

    cfg_paths = collect_files(cfg_folder, [".yml", ".yaml"])
    for cfg_path in cfg_paths:
        print(f"Start testing {cfg_path}")
        config = copy.deepcopy(cfg)

        # test load cfg
        load_config(config, cfg_path)
        assert "save_dir" in config
        assert "model" in config
        assert "data" in config
        assert "device" in config
        assert "schedule" in config
        assert "log" in config

        # test build model
        model = build_model(config.model)
        assert config.model.arch.name == model.__class__.__name__
