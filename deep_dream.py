from __future__ import print_function
import os
import numpy as np
from matplotlib import pyplot as plt
from IPython import embed
import tensorflow as tf
import vgg
import PIL
from lapnorm import LapNorm
from functools import partial
from tqdm import tqdm
from IPython import embed

def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)),session=sess)
        return wrapper
    return wrap

# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)
    
class DeepDream(object):

    # start with a gray image with a little noise
    def __init__(self, sess):
        self.sess = sess
    
    @staticmethod
    def render_deepdream(t_obj, t_input, img0=None,
                         iter_n=20, step=1.5, octave_n=3, octave_scale=2.0, lap_n=4):
        if img0 is None:
            img0 = np.random.uniform(size=(224*4, 224*4, 3)) + 100.0
            img0 = np.float32(img0)
        t_score = tf.reduce_mean(t_obj)  # defining the optimization objective
        t_grad = tf.gradients(t_score, t_input)[0]  # behold the power of automatic differentiation!

        # split the image into a number of octaves
        img = img0
        octaves = []
        for i in range(octave_n - 1):
            hw = img.shape[:2]
            lo = resize(img, np.int32(np.float32(hw) / octave_scale))
            hi = np.float32(img - resize(lo, hw))
            img = lo
            octaves.append(hi)

        lap_norm_func = tffunc(np.float32)(partial(LapNorm.lap_normalize, scale_n=lap_n))
        # generate details octave by octave
        for octave in range(octave_n):
            if octave > 0:
                hi = octaves[-octave]
                img = resize(img, hi.shape[:2]) + hi
            for i in tqdm(range(iter_n)):
                g = DeepDream.calc_grad_tiled(img, t_input, t_grad)
                g = lap_norm_func(g)
                img += g * (step / (np.abs(g).mean() + 1e-7))
            DeepDream.showarray(img / 255.0)

    
            
    @staticmethod
    def calc_grad_tiled(img, t_inp, t_grad, tile_size=224):
        print(img.shape)
        '''Compute the value of tensor t_grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile boundaries over
        multiple iterations.'''
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        pairs = []
        for y in range(0, max(h-sz//2, sz),sz):
            for x in range(0, max(w-sz//2, sz),sz):
                sub = img_shift[y:y+sz,x:x+sz]
                print(x,y,sz, sub.shape)
                pairs.append((x,y,sub))
        subs = np.array([sub for _,_,sub in pairs])
        grads = sess.run(t_grad, {t_inp:subs})
        print(grads.shape)
        for i,(x,y,sub) in enumerate(pairs):
            grad[y:y+sz,x:x+sz] = grads[i]
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)


    @staticmethod
    def showarray(a):
        a = np.uint8(np.clip(a, 0, 1) * 255)
        plt.imshow(a)
        plt.show()


    @staticmethod
    def visstd(a, s=0.1):
        '''Normalize the image range for visualization'''
        return (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5

    '''
    @staticmethod
    def render_naive(t_obj, t_inp, img0=img_noise, iter_n=20, step=1.0):
        t_score = tf.reduce_mean(t_obj)  # defining the optimization objective
        t_grad = tf.gradients(t_score, t_inp)[0]  # behold the power of automatic differentiation!
        img = img0.copy()
        for i in range(iter_n):
            g, score = sess.run([t_grad, t_score], {t_inp: img})
            # normalizing the gradient, so the same step size should work
            g /= g.std() + 1e-8  # for different layers and networks
            img += g * step
            print(score, end=' ')
        showarray(visstd(img))


    @staticmethod
    def render_multiscale(t_obj, t_inp, img0=img_noise, iter_n=10, step=1.0, octave_n=4, octave_scale=2.0):
        t_score = tf.reduce_mean(t_obj)  # defining the optimization objective
        t_grad = tf.gradients(t_score, t_inp)[0]  # behold the power of automatic differentiation!
    
        img = img0.copy()
        for octave in range(octave_n):
            if octave > 0:
                hw = np.float32(img.shape[:2]) * octave_scale
                img = resize(img, np.int32(hw))
            for i in range(iter_n):
                g = calc_grad_tiled(img, t_inp, t_grad)
                # normalizing the gradient, so the same step size should work
                g /= g.std() + 1e-8  # for different layers and networks
                img += g * step
                print('.', end=' ')
            showarray(visstd(img))
    '''
#model_fn = 'inception5h/tensorflow_inception_graph.pb'

# creating TensorFlow session and loading the model
#graph = tf.Graph()
#sess = tf.InteractiveSession(graph=graph)
#with tf.gfile.FastGFile(model_fn, 'rb') as f:
#    graph_def = tf.GraphDef()
#    graph_def.ParseFromString(f.read())
#t_input = tf.placeholder(np.float32, name='input') # define the input tensor
#imagenet_mean = 117.0
#t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
#tf.import_graph_def(graph_def, {'input':t_preprocessed})

LAYER = "prob"
CHANNEL = 679

graph = tf.Graph()
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session()
inp = tf.placeholder(np.float32, shape=(None,224,224,3), name='input') # define the input tensor
vgg_weights, vgg_mean_pxl, classes = vgg.load_net("imagenet-vgg-verydeep-19.mat")
inp_pre = inp - vgg_mean_pxl
net = vgg.net_preloaded(vgg_weights, inp_pre, pooling="max")

print(f"class={classes[CHANNEL]}")
img0 = PIL.Image.open('examples/1-content.jpg')
img0 = np.float32(img0)
deepdream = DeepDream(sess)
deepdream.render_deepdream(net[LAYER][:,:,:,CHANNEL], inp)
