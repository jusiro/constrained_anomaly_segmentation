import random
import numpy as np
import cv2

from skimage import exposure
from matplotlib import pyplot as plt
import imutils


def image_normalization(x, shape, norm='max', channels=3, histogram_matching=False, reference_image=None,
                        mask=False, channel_first=True):

    # Histogram matching to reference image
    if histogram_matching:
        x_norm = exposure.match_histograms(x, reference_image)
        x_norm[x == 0] = 0
        x = x_norm

    # image resize
    x = imutils.resize(x, height=shape)
    #x = resize_image_canvas(x, shape)

    # Grayscale image -- add channel dimension
    if len(x.shape) < 3:
        x = np.expand_dims(x, -1)

    if mask:
        x = (x > 200)

    # channel first
    if channel_first:
        x = np.transpose(x, (2, 0, 1))
    if not mask:
        if norm == 'max':
            x = x / 255.0
        elif norm == 'zscore':
            x = (x - 127.5) / 127.5

    # numeric type
    x.astype('float32')
    return x


def plot_image(x, y=None, denorm_intensity=False, channel_first=True):
    if len(x.shape) < 3:
        x = np.expand_dims(x, 0)
    # channel first
    if channel_first:
        x = np.transpose(x, (1, 2, 0))
    if denorm_intensity:
        if self.norm == 'zscore':
            x = (x*127.5) + 127.5
            x = x.astype(int)

    plt.imshow(x)

    if y is not None:
        y = np.expand_dims(y[0, :, :], -1)
        plt.imshow(y, cmap='jet', alpha=0.1)

    plt.axis('off')
    plt.show()


def augment_input_batch(batch):
    masks = np.zeros(batch.shape)

    for i in np.arange(0, batch.shape[0]):
        (batch[i, :, :, :], masks[i, :, :, :]) = augment_input_context(batch[i, :, :, :])

    return batch, masks


def augment_input_context(x):
    im = x.copy()
    mask = np.zeros(im.shape)

    # Randomize anomaly size
    w = random.randint(0, im.shape[2] // 10)

    # Random center-cropping
    xx = random.randint(0, im.shape[2] - w)
    yy = random.randint(0, im.shape[1] - w)

    # Get intensity
    i = np.percentile(im, 99) + np.std(im)

    # Inset anomaly
    im[:, xx-w:xx+w, yy-w:yy+w] = 0
    mask[:, xx-w:xx+w, yy-w:yy+w] = 1

    # keep only skull
    im[x==0] = 0
    mask[x == 0] = 0

    return im, mask


def resize_image_canvas(image, input_shape, pcn_eat_hz=0.):

    # cut a bit on the sides
    if pcn_eat_hz > 0:
        h, w = image.shape
        px_eat_hz = int(pcn_eat_hz * w)
        image = image[:, px_eat_hz:w - px_eat_hz]

    h, w = image.shape
    ratio_h = input_shape[1] / h
    ratio_w = input_shape[2] / w

    img_res = np.zeros((input_shape[1], input_shape[2]), dtype=image.dtype)

    if ratio_w > ratio_h:
        img_res_h = imutils.resize(image, height=input_shape[1])
        left_margin = (input_shape[2] - img_res_h.shape[1]) // 2
        img_res[:, left_margin:left_margin + img_res_h.shape[1]] = img_res_h
    else:
        img_res_w = imutils.resize(image, width=input_shape[2])
        top_margin = (input_shape[1] - img_res_w.shape[0]) // 2
        img_res[top_margin:top_margin + img_res_w.shape[0], :] = img_res_w

    return img_res

