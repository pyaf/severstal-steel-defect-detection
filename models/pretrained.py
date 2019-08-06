import segmentation_models_pytorch as smp


def UNet(encoder="resnet34", classes=4, pretrained="imagenet"):
    model = smp.Unet(encoder, encoder_weights=pretrained, classes=classes, activation=None)
    return model
