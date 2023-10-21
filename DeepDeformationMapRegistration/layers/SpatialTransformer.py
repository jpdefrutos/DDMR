import tensorflow.keras.layers as kl
import tensorflow.keras.backend as K
import tensorflow as tf
import neurite as ne

import h5py
from DeepDeformationMapRegistration.utils.constants import IMG_SHAPE, DISP_MAP_SHAPE


class SpatialTransformer(kl.Layer):
    """
    Adapted SpatialTransformer layer taken from VoxelMorph v0.1
    https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/tf/layers.py
    Removed unused options to ease portability

    N-D Spatial Transformer Tensorflow / Keras Layer

    The Layer can handle ONLY dense transforms.
    Transforms are meant to give a 'shift' from the current position.
    Therefore, a dense transform gives displacements (not absolute locations) at each voxel.

    If you find this function useful, please cite:
      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.

    Originally, this code was based on voxelmorph code, which
    was in turn transformed to be dense with the help of (affine) STN code
    via https://github.com/kevinzakka/spatial-transformer-network

    Since then, we've re-written the code to be generalized to any
    dimensions, and along the way wrote grid and interpolation functions
    """

    def __init__(self,
                 interp_method='linear',
                 indexing='ij',
                 single_transform=False,
                 fill_value=None,
                 add_identity=True,
                 shift_center=True,
                 **kwargs):
        """
        Parameters:
            interp_method: 'linear' or 'nearest'
            single_transform: whether a single transform supplied for the whole batch
            indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian)
                'xy' indexing will have the first two entries of the flow
                (along last axis) flipped compared to 'ij' indexing
            fill_value (default: None): value to use for points outside the domain.
                If None, the nearest neighbors will be used.
            add_identity (default: True): whether the identity matrix is added
                to affine transforms.
            shift_center (default: True): whether the grid is shifted to the center
                of the image when converting affine transforms to warp fields.
        """
        self.interp_method = interp_method
        self.fill_value = fill_value
        self.add_identity = add_identity
        self.shift_center = shift_center
        self.ndims = None
        self.inshape = None
        self.single_transform = single_transform

        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing

        super(self.__class__, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'interp_method': self.interp_method,
            'indexing': self.indexing,
            'single_transform': self.single_transform,
            'fill_value': self.fill_value,
            'add_identity': self.add_identity,
            'shift_center': self.shift_center,
        })
        return config

    def build(self, input_shape):
        """
        input_shape should be a list for two inputs:
        input1: image.
        input2: transform Tensor
            if affine:
                should be a N x N+1 matrix
                *or* a N*N+1 tensor (which will be reshape to N x (N+1) and an identity row added)
            if not affine:
                should be a *vol_shape x N
        """

        if len(input_shape) > 2:
            raise Exception('Spatial Transformer must be called on a list of length 2.'
                            'First argument is the image, second is the transform.')

        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.inshape = input_shape
        vol_shape = input_shape[0][1:-1]
        trf_shape = input_shape[1][1:]

        # the transform is an affine iff:
        # it's a 1D Tensor [dense transforms need to be at least ndims + 1]
        # it's a 2D Tensor and shape == [N+1, N+1] or [N, N+1]
        #   [dense with N=1, which is the only one that could have a transform shape of 2, would be of size Mx1]
        is_matrix = len(trf_shape) == 2 and trf_shape[0] in (self.ndims, self.ndims + 1) and trf_shape[
            1] == self.ndims + 1
        assert not (len(trf_shape) == 1 or is_matrix), "Invalid transformation. Expected a dense displacement map"

        # check sizes
        if trf_shape[-1] != self.ndims:
            raise Exception('Offset flow field size expected: %d, found: %d'
                            % (self.ndims, trf_shape[-1]))

        # confirm built
        self.built = True

    def call(self, inputs):
        """
        Parameters
            inputs: list with two entries
        """

        # check shapes
        assert len(inputs) == 2, "inputs has to be len 2, found: %d" % len(inputs)
        vol = inputs[0]
        trf = inputs[1]

        # necessary for multi_gpu models...
        vol = K.reshape(vol, [-1, *self.inshape[0][1:]])
        trf = K.reshape(trf, [-1, *self.inshape[1][1:]])

        # prepare location shift
        if self.indexing == 'xy':  # shift the first two dimensions
            trf_split = tf.split(trf, trf.shape[-1], axis=-1)
            trf_lst = [trf_split[1], trf_split[0], *trf_split[2:]]
            trf = tf.concat(trf_lst, -1)

        # map transform across batch
        if self.single_transform:
            fn = lambda x: self._single_transform([x, trf[0, :]])
            return tf.map_fn(fn, vol, dtype=tf.float32)
        else:
            return tf.map_fn(self._single_transform, [vol, trf], dtype=tf.float32)

    def _single_transform(self, inputs):
        return self._transform(inputs[0], inputs[1], interp_method=self.interp_method, fill_value=self.fill_value)

    def _transform(self, vol, loc_shift, interp_method='linear', indexing='ij', fill_value=None):
        """
        transform (interpolation N-D volumes (features) given shifts at each location in tensorflow

        Essentially interpolates volume vol at locations determined by loc_shift.
        This is a spatial transform in the sense that at location [x] we now have the data from,
        [x + shift] so we've moved data.

        Parameters:
            vol: volume with size vol_shape or [*vol_shape, nb_features]
            loc_shift: shift volume [*new_vol_shape, N]
            interp_method (default:'linear'): 'linear', 'nearest'
            indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian).
                In general, prefer to leave this 'ij'
            fill_value (default: None): value to use for points outside the domain.
                If None, the nearest neighbors will be used.

        Return:
            new interpolated volumes in the same size as loc_shift[0]

        Keyworks:
            interpolation, sampler, resampler, linear, bilinear
        """

        # parse shapes

        if isinstance(loc_shift.shape, (tf.compat.v1.Dimension, tf.TensorShape)):
            volshape = loc_shift.shape[:-1].as_list()
        else:
            volshape = loc_shift.shape[:-1]
        nb_dims = len(volshape)

        # location should be mesh and delta
        mesh = ne.utils.volshape_to_meshgrid(volshape, indexing=indexing)  # volume mesh
        loc = [tf.cast(mesh[d], 'float32') + loc_shift[..., d] for d in range(nb_dims)]

        # test single
        return ne.utils.interpn(vol, loc, interp_method=interp_method, fill_value=fill_value)


if __name__ == "__main__":
    output_file = './spatialtransformer.h5'

    in_dm = tf.keras.Input(DISP_MAP_SHAPE)
    in_image = tf.keras.Input(IMG_SHAPE)
    pred = SpatialTransformer(interp_method='linear', indexing='ij', single_transform=False)([in_image, in_dm])

    model = tf.keras.Model(inputs=[in_image, in_dm], outputs=pred)

    model.save(output_file)
    print(f"SpatialTransformer layer saved in: {output_file}")
