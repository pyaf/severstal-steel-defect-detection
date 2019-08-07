#from .unet_model import *
from .pretrained import *



def get_model(model_name, encoder_name, num_classes):
    if model_name == "UNet":
        return UNet(encoder=encoder_name, classes=num_classes)


