# Copyright 2023 cansik.
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

import logging
import os
import time
from collections import defaultdict
from typing import Optional, Sequence

import cv2
import numpy as np
from pycocotools.coco import COCO

from .coco import CocoDataset
from .xml_dataset import get_file_list


class CocoYolo(COCO):
    def __init__(self, annotation):
        """
        Constructor of Microsoft COCO helper class for
        reading and visualizing annotations.
        :param annotation: annotation dict
        :return:
        """
        # load dataset
        super().__init__()
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        dataset = annotation
        assert type(dataset) == dict, "annotation file format {} not supported".format(
            type(dataset)
        )
        self.dataset = dataset
        self.createIndex()


class YoloDataset(CocoDataset):
    def __init__(self, class_names, **kwargs):
        self.class_names = class_names
        super(YoloDataset, self).__init__(**kwargs)

    @staticmethod
    def _find_image(
        image_prefix: str,
        image_types: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff"),
    ) -> Optional[str]:
        for image_type in image_types:
            path = f"{image_prefix}{image_type}"
            if os.path.exists(path):
                return path
        return None

    def yolo_to_coco(self, ann_path):
        """
        convert yolo annotations to coco_api
        :param ann_path:
        :return:
        """
        logging.info("loading annotations into memory...")
        tic = time.time()
        ann_file_names = get_file_list(ann_path, type=".txt")
        logging.info("Found {} annotation files.".format(len(ann_file_names)))
        image_info = []
        categories = []
        annotations = []
        for idx, supercat in enumerate(self.class_names):
            categories.append(
                {"supercategory": supercat, "id": idx + 1, "name": supercat}
            )
        ann_id = 1

        for idx, txt_name in enumerate(ann_file_names):
            ann_file = os.path.join(ann_path, txt_name)
            image_file = self._find_image(os.path.splitext(ann_file)[0])

            if image_file is None:
                logging.warning(f"Could not find image for {ann_file}")
                continue

            with open(ann_file, "r") as f:
                lines = f.readlines()

            image = cv2.imread(image_file)
            height, width = image.shape[:2]

            file_name = os.path.basename(image_file)
            info = {
                "file_name": file_name,
                "height": height,
                "width": width,
                "id": idx + 1,
            }
            image_info.append(info)
            for line in lines:
                data = [float(t) for t in line.split(" ")]
                cat_id = int(data[0])
                locations = np.array(data[1:]).reshape((len(data) // 2, 2))
                bbox = locations[0:2]

                bbox[0] -= bbox[1] * 0.5

                bbox = np.round(bbox * np.array([width, height])).astype(int)
                x, y = bbox[0][0], bbox[0][1]
                w, h = bbox[1][0], bbox[1][1]

                if cat_id >= len(self.class_names):
                    logging.warning(
                        f"Category {cat_id} is not defined in config ({txt_name})"
                    )
                    continue

                if w < 0 or h < 0:
                    logging.warning(
                        "WARNING! Find error data in file {}! Box w and "
                        "h should > 0. Pass this box annotation.".format(txt_name)
                    )
                    continue

                coco_box = [max(x, 0), max(y, 0), min(w, width), min(h, height)]
                ann = {
                    "image_id": idx + 1,
                    "bbox": coco_box,
                    "category_id": cat_id + 1,
                    "iscrowd": 0,
                    "id": ann_id,
                    "area": coco_box[2] * coco_box[3],
                }
                annotations.append(ann)
                ann_id += 1

        coco_dict = {
            "images": image_info,
            "categories": categories,
            "annotations": annotations,
        }
        logging.info(
            "Load {} txt files and {} boxes".format(len(image_info), len(annotations))
        )
        logging.info("Done (t={:0.2f}s)".format(time.time() - tic))
        return coco_dict

    def get_data_info(self, ann_path):
        """
        Load basic information of dataset such as image path, label and so on.
        :param ann_path: coco json file path
        :return: image info:
        [{'file_name': '000000000139.jpg',
          'height': 426,
          'width': 640,
          'id': 139},
         ...
        ]
        """
        coco_dict = self.yolo_to_coco(ann_path)
        self.coco_api = CocoYolo(coco_dict)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)
        return img_info
