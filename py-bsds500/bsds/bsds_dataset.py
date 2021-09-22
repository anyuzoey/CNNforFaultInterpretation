import os
import numpy as np
from PIL import Image
from skimage.util import img_as_float
from skimage.color import rgb2grey
from skimage.io import imread
from scipy.io import loadmat


class BSDSDataset (object):
    """
    BSDS dataset wrapper

    Given the path to the root of the BSDS dataset, this class provides
    methods for loading images, ground truths and evaluating predictions

    Attribtes:

    bsds_path - the root path of the dataset
    data_path - the path of the data directory within the root
    images_path - the path of the images directory within the data dir
    gt_path - the path of the groundTruth directory within the data dir
    train_sample_names - a list of names of training images
    val_sample_names - a list of names of validation images
    test_sample_names - a list of names of test images
    """
    def __init__(self, bsds_path):
        """
        Constructor

        :param bsds_path: the path to the root of the BSDS dataset
        """
        self.bsds_path = bsds_path
        self.data_path = os.path.join(bsds_path, 'BSDS500', 'data')
        self.images_path = os.path.join(self.data_path, 'images')
        self.gt_path = os.path.join(self.data_path, 'groundTruth')

        self.train_sample_names = self._sample_names(self.images_path, 'train')
        self.val_sample_names = self._sample_names(self.images_path, 'val')
        self.test_sample_names = self._sample_names(self.images_path, 'test')

    @staticmethod
    def _sample_names(dir, subset):
        names = []
        files = os.listdir(os.path.join(dir, subset))
        for fn in files:
            dir, filename = os.path.split(fn)
            name, ext = os.path.splitext(filename)
            if ext.lower() == '.jpg':
                names.append(os.path.join(subset, name))
        return names

    def read_image(self, name):
        """
        Load the image identified by the sample name (you can get the names
        from the `train_sample_names`, `val_sample_names` and
        `test_sample_names` attributes)
        :param name: the sample name
        :return: a (H,W,3) array containing the image, scaled to range [0,1]
        """
        path = os.path.join(self.images_path, name + '.jpg')
        return img_as_float(imread(path))

    def get_image_shape(self, name):
        """
        Get the shape of the image identified by the sample name (you can
        get the names from the `train_sample_names`, `val_sample_names` and
        `test_sample_names` attributes)
        :param name: the sample name
        :return: a tuple of the form `(height, width, channels)`
        """
        path = os.path.join(self.images_path, name + '.jpg')
        img = Image.open(path)
        return img.height, img.width, 3

    def ground_truth_mat(self, name):
        """
        Load the ground truth Matlab file identified by the sample name
        (you can get the names from the `train_sample_names`,
        `val_sample_names` and `test_sample_names` attributes)
        :param name: the sample name
        :return: the `groundTruth` entry from the Matlab file
        """
        path = os.path.join(self.gt_path, name + '.mat')
        return self.load_ground_truth_mat(path)

    def segmentations(self, name):
        """
        Load the ground truth segmentations identified by the sample name
        (you can get the names from the `train_sample_names`,
        `val_sample_names` and `test_sample_names` attributes)
        :param name: the sample name
        :return: a list of (H,W) arrays, each of which contains a
        segmentation ground truth
        """
        path = os.path.join(self.gt_path, name + '.mat')
        return self.load_segmentations(path)

    def boundaries(self, name):
        """
        Load the ground truth boundaries identified by the sample name
        (you can get the names from the `train_sample_names`,
        `val_sample_names` and `test_sample_names` attributes)
        :param name: the sample name
        :return: a list of (H,W) arrays, each of which contains a
        boundary ground truth
        """
        path = os.path.join(self.gt_path, name + '.mat')
        return self.load_boundaries(path)

    @staticmethod
    def load_ground_truth_mat(path):
        """
        Load the ground truth Matlab file at the specified path
        and return the `groundTruth` entry.
        :param path: path
        :return: the 'groundTruth' entry from the Matlab file
        """
        gt = loadmat(path)
        return gt['groundTruth']

    @staticmethod
    def load_segmentations(path):
        """
        Load the ground truth segmentations from the Matlab file
        at the specified path.
        :param path: path
        :return: a list of (H,W) arrays, each of which contains a
        segmentation ground truth
        """
        gt = BSDSDataset.load_ground_truth_mat(path)
        num_gts = gt.shape[1]
        return [gt[0,i]['Segmentation'][0,0].astype(np.int32) for i in range(num_gts)]

    @staticmethod
    def load_boundaries(path):
        """
        Load the ground truth boundaries from the Matlab file
        at the specified path.
        :param path: path
        :return: a list of (H,W) arrays, each of which contains a
        boundary ground truth
        """
        gt = BSDSDataset.load_ground_truth_mat(path)
        num_gts = gt.shape[1]
        return [gt[0,i]['Boundaries'][0,0] for i in range(num_gts)]


class BSDSHEDAugDataset (object):
    """
    BSDS HED augmented dataset wrapper

    Given the path to the root of the BSDS dataset, this class provides
    methods for loading images, ground truths and evaluating predictions

    The augmented dataset can be downloaded from:

    http://vcl.ucsd.edu/hed/HED-BSDS.tar

    See their repo for more information:

    http://github.com/s9xie/hed

    Attribtes:

    bsds_dataset - standard BSDS dataset
    root_path - the root path of the dataset
    """

    AUG_SCALES = [
        '', '_scale_0.5', '_scale_1.5'
    ]

    AUG_ROTS = [
        '0.0', '22.5', '45.0', '67.5', '90.0', '112.5', '135.0', '157.5', '180.0', '202.5', '225.0', '247.5',
        '270.0', '292.5', '315.0', '337.5'
    ]

    AUG_FLIPS = [
        '1_0', '1_1'
    ]

    ALL_AUGS = [(s, r, f) for f in AUG_FLIPS for r in AUG_ROTS for s in AUG_SCALES]

    def __init__(self, bsds_dataset, root_path):
        """
        Constructor

        :param bsds_dataset: the standard BSDS dataset
        :param root_path: the path to the root of the augmented dataset
        """
        self.bsds_dataset = bsds_dataset
        self.root_path = root_path


        self.sample_name_to_fold = {}
        for name in bsds_dataset.train_sample_names:
            self.sample_name_to_fold[name] = 'train'
        for name in bsds_dataset.val_sample_names:
            self.sample_name_to_fold[name] = 'train'
        for name in bsds_dataset.test_sample_names:
            self.sample_name_to_fold[name] = 'test'

    def _data_path(self, data_type, scale, rot, flip, name, ext):
        fold = self.sample_name_to_fold[name]
        if data_type not in {'data', 'gt'}:
            raise ValueError("data_type should be 'data' or 'gt', not {}".format(data_type))
        if scale not in self.AUG_SCALES:
            raise ValueError("scale should be one of {}, not {}".format(self.AUG_SCALES, scale))
        if rot not in self.AUG_ROTS:
            raise ValueError("rot should be one of {}, not {}".format(self.AUG_ROTS, rot))
        if flip not in self.AUG_FLIPS:
            raise ValueError("flip should be one of {}, not {}".format(self.AUG_FLIPS, flip))
        return os.path.join(self.root_path, fold, 'aug_{}{}'.format(data_type, scale), '{}_{}'.format(rot, flip),
                            '{}{}'.format(os.path.split(name)[1], ext))

    @classmethod
    def augment_names(cls, names):
        """
        Add augmentation parameters to the supplied list of names. Converts a
        sequence of names into a sequence of tuples that provide the name along
        with augmentation parameters. Each name is combined will all possible
        combinations of augmentation parameters. By default, there are 96
        possible augmentations, so the resulting list will be 96x the length
        of `names`.

        The tuples returned can be used as parameters for the `read_image`,
        `image_shape` and `mean_boundaries` methods.

        :param names: a sequence of names
        :return: list of `(name, scale_aug, rotate_aug, flip_aug)` tuples
        """
        return [(n, s, r, f) for n in names for (s, r, f) in cls.ALL_AUGS]

    def read_image(self, name, scale, rot, flip):
        """
        Load the image identified by the sample name and augmentation
        parameters.
        The sample name `name` should come from the `train_sample_names`,
        `val_sample_names` and `test_sample_names` attributes of a
        `BSDSDataset` instance.
        The `scale`, `rot` and `flip` augmentation parameters should
        come from `AUG_SCALES`, `AUG_ROTS` and `AUG_FLIPS` attributes
        of the `BSDSHEDAugDataset` class
        :param name: the sample name
        :param scale: augmentation scale
        :param rot: augmentation rotation
        :param flip: augmentation flip
        :return: a tuple of the form `(height, width, channels)`
        """
        path = self._data_path('data', scale, rot, flip, name, '.jpg')
        return img_as_float(imread(path)).astype(np.float32)

    def get_image_shape(self, name, scale, rot, flip):
        """
        Get the shape of the image identified by the sample name
        and augmentation parameters.
        The sample name `name` should come from the `train_sample_names`,
        `val_sample_names` and `test_sample_names` attributes of a
        `BSDSDataset` instance.
        The `scale`, `rot` and `flip` augmentation parameters should
        come from `AUG_SCALES`, `AUG_ROTS` and `AUG_FLIPS` attributes
        of the `BSDSHEDAugDataset` class
        :param name: the sample name
        :param scale: augmentation scale
        :param rot: augmentation rotation
        :param flip: augmentation flip
        :return: a (H,W,3) array containing the image, scaled to range [0,1]
        """
        path = self._data_path('data', scale, rot, flip, name, '.jpg')
        img = Image.open(path)
        return img.height, img.width, 3

    def mean_boundaries(self, name, scale, rot, flip):
        """
        Load the ground truth boundaries identified by the sample name
        and augmentation parameters.

        See the `read_image` method for more information on the sample
        name and augmentation parameters

        :param name: the sample name
        :param scale: augmentation scale
        :param rot: augmentation rotation
        :param flip: augmentation flip
        :return: a list of (H,W) arrays, each of which contains a
        boundary ground truth
        """
        path = self._data_path('gt', scale, rot, flip, name, '.png')
        return self.load_mean_boundaries(path)

    @staticmethod
    def load_mean_boundaries(path):
        """
        Load the ground truth boundaries from the Matlab file
        at the specified path.
        :param path: path
        :return: a list of (H,W) arrays, each of which contains a
        boundary ground truth
        """
        return rgb2grey(img_as_float(imread(path))).astype(np.float32)
