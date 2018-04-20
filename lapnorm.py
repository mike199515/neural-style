import numpy as np
import tensorflow as tf
class LapNorm:
    k = np.float32([1,4,6,4,1])
    k = np.outer(k, k)
    k5x5 = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)

    @staticmethod
    def lap_split(img):
        '''Split the image into lo and hi frequency components'''
        with tf.name_scope('split'):
            lo = tf.nn.conv2d(img, LapNorm.k5x5, [1,2,2,1], 'SAME')
            lo2 = tf.nn.conv2d_transpose(lo, LapNorm.k5x5*4, tf.shape(img), [1,2,2,1])
            hi = img-lo2
        return lo, hi

    @staticmethod
    def lap_split_n(img, n):
        '''Build Laplacian pyramid with n splits'''
        levels = []
        for i in range(n):
            img, hi = LapNorm.lap_split(img)
            levels.append(hi)
        levels.append(img)
        return levels[::-1]

    @staticmethod
    def lap_merge(levels):
        '''Merge Laplacian pyramid'''
        img = levels[0]
        for hi in levels[1:]:
            with tf.name_scope('merge'):
                img = tf.nn.conv2d_transpose(img, LapNorm.k5x5*4, tf.shape(hi), [1,2,2,1]) + hi
        return img

    @staticmethod
    def normalize_std(img, eps=1e-10):
        '''Normalize image by making its standard deviation = 1.0'''
        with tf.name_scope('normalize'):
            std = tf.sqrt(tf.reduce_mean(tf.square(img)))
            return img/tf.maximum(std, eps)

    @staticmethod
    def lap_normalize(img, scale_n=4):
        '''Perform the Laplacian pyramid normalization.'''
        img = tf.expand_dims(img,0)
        tlevels = LapNorm.lap_split_n(img, scale_n)
        tlevels = list(map(LapNorm.normalize_std, tlevels))
        out = LapNorm.lap_merge(tlevels)
        return out[0,:,:,:]
