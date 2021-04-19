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

import os
import argparse
import torch

from nanodet.util import convert_old_model


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Convert .pth model to onnx.')
    parser.add_argument('--file_path',
                        type=str,
                        help='Path to .pth checkpoint.')
    parser.add_argument('--out_path',
                        type=str,
                        help='Path to .ckpt checkpoint.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    file_path = args.file_path
    out_path = args.out_path
    old_check_point = torch.load(file_path)
    new_check_point = convert_old_model(old_check_point)
    torch.save(new_check_point, out_path)
    print("Checkpoint saved to:", out_path)
