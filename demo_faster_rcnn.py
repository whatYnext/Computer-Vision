""" Predict with pre-trained Faster RCNN models

"""

from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils

######################################################################
# Load a pretrained model
# -------------------------

net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)

######################################################################
# Pre-process an image
# --------------------
#
# Next we download an image, and pre-process with preset data transforms.
# The default behavior is to resize the short edge of the image to 600px.
# But you can feed an arbitrarily sized image.

# This function returns two results. The first is a NDArray with shape
# `(batch_size, RGB_channels, height, width)`. It can be fed into the
# model directly. The second one contains the images in numpy format to
# easy to be plotted. Since we only loaded a single image, the first dimension
# of `x` is 1.
#
# Please beware that `orig_img` is resized to short edge 600px.

im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/detection/biking.jpg?raw=true',
                          path='biking.jpg')
x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)

######################################################################
# Inference and display
# ---------------------
#
# The Faster RCNN model returns predicted class IDs, confidence scores,
# bounding boxes coordinates. Their shape are (batch_size, num_bboxes, 1),
# (batch_size, num_bboxes, 1) and (batch_size, num_bboxes, 4), respectively.
#
# We can use :py:func:`gluoncv.utils.viz.plot_bbox` to visualize the
# results. We slice the results for the first image and feed them into `plot_bbox`:

box_ids, scores, bboxes = net(x)
ax = utils.viz.plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes)

plt.show()
