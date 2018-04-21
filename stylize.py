# Copyright (c) 2015-2018 Anish Athalye. Released under GPLv3.

from IPython import embed
import vgg

import tensorflow as tf
import numpy as np

from sys import stderr
import time

from PIL import Image
from closed_form_matting import getLaplacian
from edge_detection import TF_Canny

CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
#STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1')
#STYLE_LAYERS = ('relu3_1', 'relu4_1', 'relu5_1')
RGB2GRAY_VEC = [0.299, 0.587, 0.114]

STYLE_WEIGHT = {'relu1_1':[0,1], 'relu2_1':[0.25,0.75], 'relu3_1':[0.5,0.5],"relu4_1":[0.75,0.25],"relu5_1":[1,0]}


try:
    reduce
except NameError:
    from functools import reduce

def get_style_layers_weights(style_layer_weight_exp):
    layer_weight = 1.0
    style_layers_weights = {}
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] = layer_weight
        layer_weight *= style_layer_weight_exp

    # normalize style layer weights
    layer_weights_sum = 0
    for style_layer in STYLE_LAYERS:
        layer_weights_sum += style_layers_weights[style_layer]
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] /= layer_weights_sum
    return style_layers_weights
        
def get_content_features(shape, vgg_weights, vgg_mean_pixel, pooling, content):
    content_features = {}
    g = tf.Graph()
    with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net = vgg.net_preloaded(vgg_weights, image, pooling)
        content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])
        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={image: content_pre})
    return content_features

def get_content_gray(content):
    content_gray = rgb2gray(content)  # mathematically the same thing
    return content_gray

def get_content_edge(content):
    g = tf.Graph()
    with g.as_default(), tf.Session() as sess:
        image = tf.placeholder('float', shape=content.shape)
        rgb2gray_vec = tf.constant(RGB2GRAY_VEC)
        image_gray = tf.tensordot(image, rgb2gray_vec, axes=[[2], [0]])
        x_tensor = tf.expand_dims(tf.expand_dims(image_gray, axis=0),-1)
        edge_result = TF_Canny(x_tensor, return_raw_edges=False)
        edge_detection = edge_result.eval(feed_dict={image:content})
    return edge_detection
    
def get_style_features(styles, vgg_weights, vgg_mean_pixel, pooling):
    style_features = [{} for _ in styles]
    style_shapes = [(1,) + style.shape for style in styles]
    for i in range(len(styles)):
        g = tf.Graph()
        with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:
            image = tf.placeholder('float', shape=style_shapes[i])
            net = vgg.net_preloaded(vgg_weights, image, pooling)
            style_pre = np.array([vgg.preprocess(styles[i], vgg_mean_pixel)])
            for layer in STYLE_LAYERS:
                features = net[layer].eval(feed_dict={image: style_pre})
                features = np.reshape(features, (-1, features.shape[3]))
                gram = np.matmul(features.T, features) / features.size
                style_features[i][layer] = gram
    return style_features

def get_affine_loss(output_image, content_laplacian, affine_weight):
    loss_affine = 0.0
    output_t = output_image / 255.
    for Vc in tf.unstack(output_t, axis=-1):
        Vc_ravel = tf.reshape(tf.transpose(Vc), [-1])
        loss_affine += tf.matmul(tf.expand_dims(Vc_ravel, 0), tf.sparse_tensor_dense_matmul(content_laplacian, tf.expand_dims(Vc_ravel, -1)))
    return loss_affine * affine_weight

def get_content_loss(content_weight, content_weight_blend, content_features, net):
    content_layers_weights = {}
    content_layers_weights['relu4_2'] = content_weight_blend
    content_layers_weights['relu5_2'] = 1.0 - content_weight_blend

    content_loss = 0
    content_losses = []
    for content_layer in CONTENT_LAYERS:
        content_losses.append(content_layers_weights[content_layer] * content_weight * (2 * tf.nn.l2_loss(
            net[content_layer] - content_features[content_layer]) /
                                                                                        content_features[
                                                                                            content_layer].size))
    content_loss += reduce(tf.add, content_losses)
    return content_loss
    
def get_split_from_layer(layer,i, n):
    _, h, w, c = map(lambda i: i.value, layer.get_shape())
    h_begin = max(int(h*(i*1./n)),0)
    h_end = min(int(h*((i+1)*1./n)),h)
    return layer[:, h_begin:h_end]
    
    
def get_style_loss(styles, net, style_features, style_layers_weights, style_weight, style_blend_weights):
    style_loss = 0
    for i in range(len(styles)):
        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            _, height, width, number = map(lambda i: i.value, layer.get_shape())
            size = height * width * number
            feats = tf.reshape(layer, (-1, number))
            gram = tf.matmul(tf.transpose(feats), feats) / size
            style_gram = STYLE_WEIGHT[style_layer][i] * style_features[i][style_layer] if len(styles)==2 else style_features[i][style_layer]
            style_losses.append(
                style_layers_weights[style_layer] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
        style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)
    return style_loss

def get_tv_loss(image, tv_weight, shape):
    tv_y_size = _tensor_size(image[:, 1:, :, :])
    tv_x_size = _tensor_size(image[:, :, 1:, :])
    tv_loss = tv_weight * 2 * (
        (tf.nn.l2_loss(image[:, 1:, :, :] - image[:, :shape[1] - 1, :, :]) /
         tv_y_size) +
        (tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, :shape[2] - 1, :]) /
         tv_x_size))
    return tv_loss

def get_color_loss(image, content_gray, color_weight):
    rgb2gray_vec = tf.constant(RGB2GRAY_VEC)
    image_gray = tf.tensordot(image, rgb2gray_vec, axes=[[3], [0]])
    #color_loss = color_weight * tf.reduce_sum(tf.abs(image_gray - content_gray)) / _tensor_size(image_gray)
    color_loss = color_weight * tf.nn.l2_loss(image_gray - content_gray) / _tensor_size(image_gray)
    return color_loss

def get_edge_loss(image, content_edge, edge_weight):
    rgb2gray_vec = tf.constant(RGB2GRAY_VEC)
    image_gray = tf.tensordot(image, rgb2gray_vec, axes=[[3], [0]])
    x_tensor = tf.expand_dims(image_gray,-1)
    image_edge = TF_Canny(x_tensor, return_raw_edges=False)
    return edge_weight * tf.nn.l2_loss(image_edge-content_edge)/_tensor_size(image_edge)
    
def stylize(network, initial, initial_noiseblend, content, styles, preserve_colors, iterations,
        content_weight, content_weight_blend, style_weight, color_weight, affine_weight, edge_weight, style_layer_weight_exp, style_blend_weights, tv_weight,
        learning_rate, beta1, beta2, epsilon, pooling,
        print_iterations=None, checkpoint_iterations=None):
    """
    Stylize images.

    This function yields tuples (iteration, image); `iteration` is None
    if this is the final image (the last iteration).  Other tuples are yielded
    every `checkpoint_iterations` iterations.

    :rtype: iterator[tuple[int|None,image]]
    """
    shape = (1,) + content.shape
    #content = gray2rgb(rgb2gray(content)) # make sure it is gray scale!
    style_layers_weights = get_style_layers_weights(style_layer_weight_exp)
    vgg_weights, vgg_mean_pixel, _ = vgg.load_net(network)
    content_gray = get_content_gray(content)
    content_features = get_content_features(shape, vgg_weights, vgg_mean_pixel, pooling, content)
    content_edge = get_content_edge(content)
    style_features = get_style_features(styles, vgg_weights, vgg_mean_pixel, pooling)

    # make stylized image using backpropogation
    with tf.Graph().as_default():
        #content_laplacian = tf.to_float(getLaplacian(content / 255.))
        initial = tf.random_normal(shape) * 0.256 + vgg_mean_pixel
        image = tf.Variable(initial)
        image_pre = image - vgg_mean_pixel
        net = vgg.net_preloaded(vgg_weights, image_pre, pooling)
        content_loss =  get_content_loss(content_weight, content_weight_blend, content_features, net) if content_weight!=0 else 0
        style_loss = get_style_loss(styles, net, style_features, style_layers_weights, style_weight, style_blend_weights)
        tv_loss = get_tv_loss(image, tv_weight, shape) if tv_weight!=0 else 0
        color_loss = get_color_loss(image,content_gray,color_weight) if color_weight!=0 else 0
        affine_loss = get_affine_loss(image, content_laplacian, affine_weight) if affine_weight!=0 else 0
        edge_loss = get_edge_loss(image, content_edge, edge_weight) if edge_weight!=0 else 0
        #loss = content_loss + style_loss + tv_loss
        loss = color_loss + content_loss + style_loss + tv_loss + affine_loss + edge_loss

        # optimizer setup
        train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

        def print_progress():
            if content_weight != 0: stderr.write('    content loss: %g\n' % content_loss.eval())
            stderr.write('    style loss: %g\n' % style_loss.eval())
            if tv_weight != 0: stderr.write('       tv loss: %g\n' % tv_loss.eval())
            if color_weight != 0: stderr.write('  color loss: %g\n' % color_loss.eval())
            if affine_weight!= 0: stderr.write('    affine loss: %g\n' % affine_loss.eval())
            if edge_loss!=0 : stderr.write('    edge loss: %g\n' % edge_loss.eval())
            stderr.write('    total loss: %g\n' % loss.eval())
        # optimization
        for iteration, img_out in do_optimization(image, content, loss, shape, vgg_mean_pixel, print_iterations,print_progress, iterations, train_step, checkpoint_iterations, preserve_colors):
            yield iteration, img_out

def do_optimization(image, content, loss, shape, vgg_mean_pixel, print_iterations,print_progress, iterations, train_step, checkpoint_iterations, preserve_colors):
    best_loss = float('inf')
    best = None
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        stderr.write('Optimization started...\n')
        if (print_iterations and print_iterations != 0):
            print_progress()
        iteration_times = []
        start = time.time()
        for i in range(iterations):
            iteration_start = time.time()
            if i > 0:
                elapsed = time.time() - start
                # take average of last couple steps to get time per iteration
                remaining = np.mean(iteration_times[-10:]) * (iterations - i)
                stderr.write('Iteration %4d/%4d (%s elapsed, %s remaining)\n' % (
                    i + 1,
                    iterations,
                    hms(elapsed),
                    hms(remaining)
                ))
            else:
                stderr.write('Iteration %4d/%4d\n' % (i + 1, iterations))
            train_step.run()

            last_step = (i == iterations - 1)
            if last_step or (print_iterations and i % print_iterations == 0):
                print_progress()

            if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                this_loss = loss.eval()
                if this_loss < best_loss:
                    best_loss = this_loss
                    best = image.eval()

                img_out = best.reshape(shape[1:])

                if preserve_colors and preserve_colors == True:
                    img_out =  preserve_color(content, img_out)
                yield (
                    (None if last_step else i),
                    img_out
                )
            iteration_end = time.time()
            iteration_times.append(iteration_end - iteration_start)

def preserve_color(content, img_out):
    original_image = np.clip(content, 0, 255)
    styled_image = np.clip(img_out, 0, 255)

    # Luminosity transfer steps:
    # 1. Convert stylized RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
    # 2. Convert stylized grayscale into YUV (YCbCr)
    # 3. Convert original image into YUV (YCbCr)
    # 4. Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
    # 5. Convert recombined image from YUV back to RGB

    # 1
    styled_grayscale = rgb2gray(styled_image)
    styled_grayscale_rgb = gray2rgb(styled_grayscale)

    # 2
    styled_grayscale_yuv = np.array(Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert('YCbCr'))

    # 3
    original_yuv = np.array(Image.fromarray(original_image.astype(np.uint8)).convert('YCbCr'))

    # 4
    w, h, _ = original_image.shape
    combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
    combined_yuv[..., 0] = styled_grayscale_yuv[..., 0]
    combined_yuv[..., 1] = original_yuv[..., 1]
    combined_yuv[..., 2] = original_yuv[..., 2]

    # 5
    img_out = np.array(Image.fromarray(combined_yuv, 'YCbCr').convert('RGB'))
    return img_out

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], RGB2GRAY_VEC)

def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb

def hms(seconds):
    seconds = int(seconds)
    hours = (seconds // (60 * 60))
    minutes = (seconds // 60) % 60
    seconds = seconds % 60
    if hours > 0:
        return '%d hr %d min' % (hours, minutes)
    elif minutes > 0:
        return '%d min %d sec' % (minutes, seconds)
    else:
        return '%d sec' % seconds
