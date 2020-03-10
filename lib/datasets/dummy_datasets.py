# Copyright (c) 2017-present, Facebook, Inc.
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
##############################################################################
"""Provide stub objects that can act as stand-in "dummy" datasets for simple use
cases, like getting all classes in a dataset. This exists so that demos can be
run without requiring users to download/install datasets first.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from utils.collections import AttrDict


def get_coco_dataset():
    """A dummy COCO dataset that includes only the 'classes' field."""
    ds = AttrDict()
    classes = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def get_tt100k_dataset():
    ds = AttrDict()
    classes = [
        '__background__', 'pl40', 'p26', 'p27', 'p5', 'ip', 'pl30', 'pn', 'w30',
        'p11', 'pl5', 'wo', 'io', 'po', 'i4', 'pl70', 'pl80', 'pl50', 'ph4', 'pl100',
        'il80', 'il70', 'il60', 'pne', 'i2', 'pg', 'p17', 'p12', 'p22', 'pl60',
        'pm30', 'pl120', 'il110', 'il90', 'p10', 'w57', 'w55', 'ph4.5', 'w13',
        'pl20', 'w59', 'i5', 'w63', 'p16', 'w32', 'pb', 'pl110', 'il100', 'ph5',
        'p3', 'w58', 'ph2', 'pm55', 'p19', 'pl25', 'pm20', 'pr40', 'ph3.5', 'p18',
        'w3', 'p8', 'ps', 'ph2.8', 'w12', 'pa14', 'p6', 'p9', 'p23', 'ph3', 'w47',
        'il50', 'pr30', 'w37', 'w46', 'pm35', 'pr100', 'i10', 'pl15', 'w34', 'i13',
        'pl10', 'p1', 'i12', 'pm2', 'pl90', 'pm10', 'pr20', 'pm40', 'w16', 'w15', 'i3',
        'ph2.5', 'p15', 'pm8', 'pa12', 'w21', 'pa13', 'pr50', 'p13', 'pa10', 'ph2.2',
        'ph4.2', 'pm5', 'i1', 'pr60', 'w42', 'pw3.2', 'p21', 'p25', 'pr70', 'w22',
        'w10', 'p4', 'p14', 'pm13', 'pw4.2', 'pm50', 'w35', 'pl0', 'p2', 'w45', 'w8',
        'w41', 'pl35', 'ph4.3', 'ph3.2', 'p20', 'pa8', 'ph2.1', 'pr80', 'pm15', 'i11',
        'w20', 'i14', 'ph4.8', 'ph1.5', 'ph2.9', 'w18', 'w5', 'w38', 'pr10', 'pw2',
        'pw3', 'pw4.5', 'p28', 'ph5.3', 'pw2.5', 'pw4', 'ph2.4', 'pw3.5', 'w66',
        'p24', 'pc', 'pl4', 'pm1.5', 'ph3.3', 'w43', 'w31', 'ph5.5', 'pm46',
        'pm25', 'w24', 'w48', 'w50', 'w26', 'w60', 'ph4.4', 'w49', 'ph2.6',
        'i15', 'p7', 'pn40', 'pl65', 'w1', 'w62', 'w44']
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds