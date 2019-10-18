#from .unet_model import *
#from .pretrained import *

import segmentation_models_pytorch as smp


def get_model(model_name, encoder_name, num_classes):
    if model_name == 'UNet':
        return smp.UNet(encoder_name, encoder_weights='imagenet', classes=num_classes, activation=None)
    elif model_name == 'FPN':
        return smp.FPN(encoder_name, encoder_weights='imagenet', classes=num_classes, activation=None)


