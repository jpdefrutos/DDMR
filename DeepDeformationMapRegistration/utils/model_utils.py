import os
import requests
from datetime import datetime
from email.utils import parsedate_to_datetime, formatdate
from DeepDeformationMapRegistration.utils.constants import ANATOMIES, MODEL_TYPES, ENCODER_FILTERS, DECODER_FILTERS, IMG_SHAPE
import voxelmorph as vxm
from DeepDeformationMapRegistration.utils.logger import LOGGER


# taken from: https://lenon.dev/blog/downloading-and-caching-large-files-using-python/
def download(url, destination_file):
    headers = {}

    if os.path.exists(destination_file):
        mtime = os.path.getmtime(destination_file)
        headers["if-modified-since"] = formatdate(mtime, usegmt=True)

    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()

    if response.status_code == requests.codes.not_modified:
        return

    if response.status_code == requests.codes.ok:
        with open(destination_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=1048576):
                f.write(chunk)

        last_modified = response.headers.get("last-modified")
        if last_modified:
            new_mtime = parsedate_to_datetime(last_modified).timestamp()
            os.utime(destination_file, times=(datetime.now().timestamp(), new_mtime))


def get_models_path(anatomy: str, model_type: str, output_root_dir: str):
    assert anatomy in ANATOMIES.keys(), 'Invalid anatomy'
    assert model_type in MODEL_TYPES.keys(), 'Invalid model type'
    anatomy = ANATOMIES[anatomy]
    model_type = MODEL_TYPES[model_type]
    url = 'https://github.com/jpdefrutos/DDMR/releases/download/trained_models_v1/' + anatomy + '_' + model_type + '.h5'
    file_path = os.path.join(output_root_dir, 'models', anatomy, model_type + '.h5')
    if not os.path.exists(file_path):
        LOGGER.info(f'Model not found. Downloading from {url}... ')
        os.makedirs(os.path.split(file_path)[0], exist_ok=True)
        download(url, file_path)
        LOGGER.info(f'... downloaded model. Stored in {file_path}')
    else:
        LOGGER.info(f'Found model: {file_path}')
    return file_path


def load_model(weights_file_path: str, trainable: bool = False, return_registration_model: bool=True):
    assert os.path.exists(weights_file_path), f'File {weights_file_path} not found'
    assert weights_file_path.endswith('h5'), 'Invalid file extension. Expected .h5'

    ret_val = vxm.networks.VxmDense(inshape=IMG_SHAPE[:-1],
                                    nb_unet_features=[ENCODER_FILTERS, DECODER_FILTERS],
                                    int_steps=0)
    ret_val.load_weights(weights_file_path, by_name=True)
    ret_val.trainable = trainable

    if return_registration_model:
        ret_val = (ret_val, ret_val.get_registration_model())

    return ret_val


def get_spatialtransformer_model():
    url = 'https://github.com/jpdefrutos/DDMR/releases/download/trained_models_v1/spatialtransformer.h5'
    file_path = os.path.join(os.getcwd(), 'models', 'spatialtransformer.h5')
    if not os.path.exists(file_path):
        LOGGER.info(f'Model not found. Downloading from {url}... ')
        os.makedirs(os.path.split(file_path)[0], exist_ok=True)
        download(url, file_path)
        LOGGER.info(f'... downloaded model. Stored in {file_path}')
    else:
        LOGGER.info(f'Found model: {file_path}')
    return file_path
