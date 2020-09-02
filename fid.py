#!/usr/bin/env python3
''' Calculates the Frechet Inception Distance (FID) to evalulate GANs.

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectivly.

See --help to see further details.
'''

from __future__ import absolute_import, division, print_function
import numpy as np
import os
import gzip, pickle
import tensorflow as tf
from scipy.misc import imread, imresize
from scipy import linalg
import pathlib
import urllib
import tqdm


class InvalidFIDException(Exception):
    pass


def create_inception_graph(pth):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile( pth, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString( f.read())
        _ = tf.import_graph_def( graph_def, name='FID_Inception_Net')
#-------------------------------------------------------------------------------


# code for handling inception net derived from
#   https://github.com/openai/improved-gan/blob/master/inception_score/model.py
def _get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    layername = 'FID_Inception_Net/pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != []:
              shape = [s.value for s in shape]
              new_shape = []
              for j, s in enumerate(shape):
                if s == 1 and j == 0:
                  new_shape.append(None)
                else:
                  new_shape.append(s)
              o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return pool3
#-------------------------------------------------------------------------------


def get_activations(images, sess, batch_size=50, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    d0 = images.shape[0]
    if batch_size > d0:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = d0
    n_batches = d0//batch_size
    n_used_imgs = n_batches*batch_size
    pred_arr = np.empty((n_used_imgs,2048))
    loader_bar = tqdm.tqdm(range(n_batches))

    for i in loader_bar:
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start = i*batch_size
        end = start + batch_size
        batch = images[start:end]
        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size,-1)
    loader_bar.close()
    if verbose:
        print(" done")
    return pred_arr
#-------------------------------------------------------------------------------


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
#-------------------------------------------------------------------------------


def calculate_activation_statistics(images, sess, batch_size=50, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations(images, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma
    

#------------------
# The following methods are implemented to obtain a batched version of the activations.
# This has the advantage to reduce memory requirements, at the cost of slightly reduced efficiency.
# - Pyrestone
#------------------


def load_image_batch(files):
    """Convenience method for batch-loading images
    Params:
    -- files    : list of paths to image files. Images need to have same dimensions for all files.
    Returns:
    -- A numpy array of dimensions (num_images,hi, wi, 3) representing the image pixel values.
    """
    return np.array([imread(str(fn)).astype(np.float32) for fn in files])

def get_activations_from_files(files, sess, batch_size=50, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files      : list of paths to image files. Images need to have same dimensions for all files.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    d0 = len(files)
    if batch_size > d0:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = d0
    n_batches = d0//batch_size
    n_used_imgs = n_batches*batch_size
    pred_arr = np.empty((n_used_imgs,2048))
    loader_bar = tqdm.tqdm(range(n_batches))
    for i in loader_bar:
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start = i*batch_size
        end = start + batch_size
        #batch = load_image_batch(files[start:end])
        batch = _load_all_files(files[start:end], imsize=(args.imsize, args.imsize))
        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size,-1)
        del batch #clean up memory
    loader_bar.close()
    if verbose:
        print(" done")
    return pred_arr
    
def calculate_activation_statistics_from_files(files, sess, batch_size=50, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files      : list of paths to image files. Images need to have same dimensions for all files.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations_from_files(files, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_activate_error_from_files(files1, files2, sess, batch_size=50,
        verbose=False):
    act1 = get_activations_from_files(files1, sess, batch_size, verbose)
    act2 = get_activations_from_files(files2, sess, batch_size, verbose)
    error = np.mean(((act1-act2)*(act1-act2)).sum(axis=1))
    return error 
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# The following functions aren't needed for calculating the FID
# they're just here to make this module work as a stand-alone script
# for calculating FID scores
#-------------------------------------------------------------------------------
def check_or_download_inception(inception_path):
    ''' Checks if the path to the inception file is valid, or downloads
        the file if it is not present. '''
    INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    if inception_path is None:
        inception_path = './'
    inception_path = pathlib.Path(inception_path)
    model_file = inception_path / 'classify_image_graph_def.pb'
    if not model_file.exists():
        print("Downloading Inception model")
        from urllib import request
        import tarfile
        fn, _ = request.urlretrieve(INCEPTION_URL)
        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb', str(model_file.parent))
    return str(model_file)



def _load_all_filenames(fullpath):
    print(fullpath)
    images = []
    for path, subdirs, files in os.walk(fullpath):
        for name in files:
            if os.path.splitext(name)[-1].lower() in ['.jpg', '.png', '.jpeg']:
                filename = os.path.join(path, name)
                # print('filename', filename)
                # print('path', path, '\nname', name)
                # print('filename', filename)
                if os.path.isfile(filename):
                    images.append(filename)
    print('images number:', len(images))
    return images

def _load_all_files(files, imsize=(299,299)):
    # the data should be in (0,255) with shape (batch, height, width, channel)
    # images = np.stack([imresize(imread(str(image), mode='RGB'), imsize, interp='lanczos').astype(np.float32) for image in files])
    images = np.stack([imresize(imread(str(image), mode='RGB'), imsize, interp='lanczos').astype(np.float32) for image in files])
    #images = images.transpose((0,3,1,2))
    #images /= 255    
    return images



def _handle_path(path, sess, low_profile=False):
    if path.endswith('.npz') or path.endswith('.np'):
        if path.endswith('npz'):
            f = np.load(path)    
            m, s = f['mu'][:], f['sigma'][:]
            f.close()
        else:
            f = np.load(path).item()
            m, s = f['mu'][:], f['sigma'][:]
            f.close()
    else:
        # path = pathlib.Path(path)
        
        #files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        files = _load_all_filenames(path)
        if low_profile:
            m, s = calculate_activation_statistics_from_files(files, sess, batch_size=args.batch_size)
        else:
            x = _load_all_files(files, imsize=args.imsize)
            m, s = calculate_activation_statistics(x, sess)
            del x #clean up memory
        #save mu and sigma
        np.savez_compressed(path, mu=m, sigma=s)

    return m, s


def calculate_fid_given_paths(paths, inception_path, low_profile=False):
    ''' Calculates the FID of two paths. '''
    inception_path = check_or_download_inception(inception_path)

    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)

    create_inception_graph(str(inception_path))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        m1, s1 = _handle_path(paths[0], sess, low_profile=low_profile)
        np.savez_compressed(paths[0], mu=m1, sigma=s1)
        m2, s2 = _handle_path(paths[1], sess, low_profile=low_profile)
        #np.savez_compressed(paths[0], mu=m1, sigma=s1)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value

def calculate_ipd_given_path(paths, inception_path, low_profile=False):
    inception_path = check_or_download_inception(inception_path)
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)
    files1 = _load_all_filenames(paths[0])
    files2 = _load_all_filenames(paths[1])
    print(paths[0], len(files1), paths[1], len(files2))
    # check files
    if  not len(files1)==len(files2):
        folder1 = os.path.split(files1[0])[0]
        folder2 = os.path.split(files2[0])[0]
        ext1 = os.path.splitext(files1[0])[-1]
        ext2 = os.path.splitext(files2[0])[-1]
        filename1 = [os.path.split(_f)[-1][:-4] for _f in files1]
        filename2 = [os.path.split(_f)[-1][:-4] for _f in files2]
        filenames = [_f for _f in filename1 if _f in filename2]
        files1 = [os.path.join(folder1, _f+ext1) for _f in filenames]
        files2 = [os.path.join(folder2, _f+ext2) for _f in filenames]
    print(paths[0], len(files1), paths[1], len(files2))
    create_inception_graph(str(inception_path))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ipd = calculate_activate_error_from_files(files1, files2, sess)
    return ipd

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=str, nargs=2,
        help='Path to the generated images or to .npz statistic files')
    parser.add_argument("-i", "--inception", type=str, default=None,
        help='Path to Inception model (will be downloaded if not provided)')
    parser.add_argument("--gpu", default="", type=str,
        help='GPU to use (leave blank for CPU only)')
    parser.add_argument("--imsize", default=256, type=int,
        help='image size, default (256,256)')
    parser.add_argument("--batch_size", default=64, type=int,
        help='batch size for evaluation, default 64')
    parser.add_argument("--lowprofile", action="store_true",
        help='Keep only one batch of images in memory at a time. This reduces memory footprint, but may decrease speed slightly.')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(args)
    # fid_value = calculate_fid_given_paths(args.path, args.inception, low_profile=args.lowprofile)
    # print("FID: ", fid_value)
    ipd = calculate_ipd_given_path(args.path, args.inception, low_profile=args.lowprofile)
    print("IPD: ", ipd)
