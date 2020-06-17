# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 20:06:25 2020

@author: mayao
"""

import d2lzh as d2l
from mxnet import contrib, image, nd

img = image.imread('C:/Users/mayao/d2l-zh/img/catdog.jpg')
h, w = img.shape[0:2]

d2l.set_figsize()

def display_anchors(fmap_w, fmap_h, s):
    fmap = nd.zeros((1, 10, fmap_w, fmap_h)) # 1 sample, 10 channels
    anchors = contrib.nd.MultiBoxPrior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = nd.array((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img.asnumpy()).axes,
                    anchors[0] * bbox_scale)