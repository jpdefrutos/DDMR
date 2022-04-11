import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

PYCHARM_EXEC = os.getenv('PYCHARM_EXEC') == 'True'

import tensorflow as tf
import voxelmorph as vxm
from voxelmorph.tf.modelio import LoadableModel, store_config_args
from tensorflow.keras.layers import UpSampling3D


class WeaklySupervised(LoadableModel):

    @store_config_args
    def __init__(self, inshape, all_labels: [list, tuple], nb_unet_features=None, int_steps=5, bidir=False,
                 int_downsize=1, outshape=None, **kwargs):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            all_labels: List of all labels included in training segmentations.
            hot_labels: List of labels to output as one-hot maps.
            nb_unet_features: Unet convolutional features. See VxmDense documentation for more information.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Dowsampling of the displacement map. Integer
            kwargs: Forwarded to the internal VxmDense model.
        """

        mov_segm = tf.keras.Input((*inshape, len(all_labels)), name='mov_segmentations_input')

        fix_img = tf.keras.Input((*inshape, 1), name='fix_image_input')
        mov_img = tf.keras.Input((*inshape, 1), name='mov_image_input')

        input_model = tf.keras.Model(inputs=[mov_img, fix_img], outputs=[mov_img, fix_img])

        vxm_model = vxm.networks.VxmDense(inshape=inshape,
                                          nb_unet_features=nb_unet_features,
                                          input_model=input_model,
                                          int_steps=int_steps,
                                          bidir=bidir,
                                          int_downsize=int_downsize,
                                          **kwargs)

        pred_segm = vxm.layers.SpatialTransformer(interp_method='linear', indexing='ij', name='interp_segm')(
            [mov_segm, vxm_model.references.pos_flow])

        inputs = [mov_img, fix_img, mov_segm] # mov_img, mov_segm, fix_segm
        model_outputs = vxm_model.outputs
        if outshape is not None:
            scale_factors = [o//i for i, o in zip(inshape, outshape)]
            upsampling_layer = UpSampling3D(scale_factors)  # Doesn't perform trilinear, only nearest
            # Image
            model_outputs[0] = upsampling_layer(model_outputs[0])
            # Segmentation
            pred_segm = upsampling_layer(pred_segm)
            # Displacement map
            model_outputs[1] = upsampling_layer(scale_factors)(model_outputs[1])
            model_outputs[1] = tf.multiply(model_outputs[1], tf.cast(scale_factors, model_outputs[1].dtype))

        # Just renaming
        pred_fix_image = tf.identity(model_outputs[0], name='pred_fix_image')
        pred_dm = tf.identity(model_outputs[1], name='pred_dm')
        pred_segm = tf.identity(pred_segm, name='pred_fix_segm')
        outputs = [pred_fix_image, pred_segm, pred_dm]

        self.references = LoadableModel.ReferenceContainer()
        self.references.pred_segm = pred_segm
        self.references.pred_img = vxm_model.outputs[0]
        self.references.pos_flow = vxm_model.references.pos_flow

        super(WeaklySupervised, self).__init__(inputs=inputs, outputs=outputs)

    def get_registration_model(self):
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, mov_img, mov_segm, fix_segm):
        return self.get_registration_model().predict([mov_segm, fix_segm, mov_img])

    def apply_transform(self, mov_img, mov_segm, fix_segm, interp_method='linear'):
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=mov_img.shape[1:], name='input_img')
        pred_img = vxm.layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs, pred_img).predict([mov_segm, fix_segm, mov_img])

