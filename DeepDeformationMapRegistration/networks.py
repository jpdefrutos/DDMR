import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

PYCHARM_EXEC = os.getenv('PYCHARM_EXEC') == 'True'

import tensorflow as tf
import voxelmorph as vxm
from voxelmorph.tf.modelio import LoadableModel, store_config_args


class VxmWeaklySupervised(LoadableModel):

    @store_config_args
    def __init__(self, inshape, all_labels: [list, tuple], nb_unet_features=None, int_steps=5, bidir=False, **kwargs):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            all_labels: List of all labels included in training segmentations.
            hot_labels: List of labels to output as one-hot maps.
            nb_unet_features: Unet convolutional features. See VxmDense documentation for more information.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            kwargs: Forwarded to the internal VxmDense model.
        """

        fix_segm = tf.keras.Input((*inshape, len(all_labels)), name='fix_segmentations_input')
        mov_segm = tf.keras.Input((*inshape, len(all_labels)), name='mov_segmentations_input')

        mov_img = tf.keras.Input((*inshape, 1), name='mov_image_input')

        unet_input_model = tf.keras.Model(inputs=[mov_segm, fix_segm], outputs=[mov_segm, fix_segm])

        vxm_model = vxm.networks.VxmDense(inshape=inshape,
                                          nb_unet_features=nb_unet_features,
                                          input_model=unet_input_model,
                                          int_steps=int_steps,
                                          bidir=bidir, **kwargs)

        pred_img = vxm.layers.SpatialTransformer(interp_method='linear', indexing='ij', name='pred_fix_img')(
            [mov_img, vxm_model.references.pos_flow])

        inputs = [mov_segm, fix_segm, mov_img] # mov_img, mov_segm, fix_segm
        outputs = [pred_img] + vxm_model.outputs

        self.references = LoadableModel.ReferenceContainer()
        self.references.pred_segm = vxm_model.outputs[0]
        self.references.pred_img = pred_img
        self.references.pos_flow = vxm_model.references.pos_flow

        super().__init__(inputs=inputs, outputs=outputs)

    def get_registration_model(self):
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, mov_img, mov_segm, fix_segm):
        return self.get_registration_model().predict([mov_segm, fix_segm, mov_img])

    def apply_transform(self, mov_img, mov_segm, fix_segm, interp_method='linear'):
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=mov_img.shape[1:], name='input_img')
        pred_img = vxm.layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs, pred_img).predict([mov_segm, fix_segm, mov_img])
