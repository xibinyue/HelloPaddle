#!/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 - xibin.yue <xibin.yue@moji.com>
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import os


class CifarDataLoader(object):
    def __init__(self, root_dir, data_property='train', single_epoch=False):
        self.root_dir = root_dir
        self.data_property = data_property
        self.single_epoch = single_epoch

    def _load_cifar(self, data_path):
        try:
            data_dict = pickle.load(open(data_path, 'rb'))
            raw_data = data_dict['data']
            label_data = data_dict['labels']
            batch_anno = data_dict['batch_label']
            raw_data = raw_data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
            return raw_data, label_data, batch_anno
        except Exception as e:
            print e

    def reader(self):
        while True:
            if self.data_property == 'train':
                for batch_id in range(1, 6):
                    file_path = os.path.join(self.root_dir, 'data_batch_%d' % batch_id)
                    raw_data, label_data, batch_anno = self._load_cifar(file_path)
                    print 'BATCH INFO - %s' % batch_anno.upper()
                    length = raw_data.shape[0]
                    for i in range(length):
                        yield raw_data[i], label_data[i]
                if self.single_epoch:
                    break
            else:
                file_path = os.path.join(self.root_dir, 'test_batch')
                raw_data, label_data, batch_anno = self._load_cifar(file_path)
                print 'BATCH INFO - %s' % batch_anno.upper()
                length = raw_data.shape[0]
                for i in range(length):
                    yield raw_data[i], label_data[i]
                if self.single_epoch:
                    break
