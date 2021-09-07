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

import random

import cv2
import numpy as np


def random_brightness(img, delta):
    img += random.uniform(-delta, delta)
    return img


def random_contrast(img, alpha_low, alpha_up):
    img *= random.uniform(alpha_low, alpha_up)
    return img


def random_saturation(img, alpha_low, alpha_up):
    hsv_img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
    hsv_img[..., 1] *= random.uniform(alpha_low, alpha_up)
    img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return img


def normalize(meta, mean, std):
    img = meta["img"].astype(np.float32)
    mean = np.array(mean, dtype=np.float64).reshape(1, -1)
    stdinv = 1 / np.array(std, dtype=np.float64).reshape(1, -1)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    meta["img"] = img
    return meta


def _normalize(img, mean, std):
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3) / 255
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3) / 255
    img = (img - mean) / std
    return img


def color_aug_and_norm(meta, kwargs):
    img = meta["img"].astype(np.float32) / 255

    if "brightness" in kwargs and random.randint(0, 1):
        img = random_brightness(img, kwargs["brightness"])

    if "contrast" in kwargs and random.randint(0, 1):
        img = random_contrast(img, *kwargs["contrast"])

    if "saturation" in kwargs and random.randint(0, 1):
        img = random_saturation(img, *kwargs["saturation"])
    # cv2.imshow('trans', img)
    # cv2.waitKey(0)
    img = _normalize(img, *kwargs["normalize"])
    meta["img"] = img
    return meta
