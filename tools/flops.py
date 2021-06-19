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

from nanodet.model.arch import build_model
from nanodet.util import cfg, get_model_complexity_info, load_config


def main(config, input_shape=(3, 320, 320)):
    model = build_model(config.model)
    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(
        f'{split_line}\nInput shape: {input_shape}\n'
        f'Flops: {flops}\nParams: {params}\n{split_line}'
    )


if __name__ == '__main__':
    cfg_path = r'config/nanodet-m.yml'
    load_config(cfg, cfg_path)
    main(config=cfg, input_shape=(3, 320, 320))
