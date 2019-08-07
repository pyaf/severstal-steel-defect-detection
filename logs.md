# Logs for the competition

# ADPOS diabetic retina

# Models on training:

# 6 Aug

Dataloader on npy files is very slow: 11 minutes for one loop.
with make_mask, cv2.imread: 1:25 :)
with make_mask, np.load image: 3:17

`68_UNet_f1_test`: org image training, with resnet34 arch unet, lr=5e-5
LB: 0.88

# 8 Aug

`88_se_resnext_101x4d_unet`: same as above with se_resnext_101x4d encoder, with 1e-4
batch size 4, 2, grad acc = 32, 30 min for train epoch *.*





OLD kaggle competition:
https://www.kaggle.com/iafoss/unet34-dice-0-8://www.kaggle.com/iafoss/unet34-dice-0-87
install latest of segmentation_models.pytorch


# Questions and Ideas:

# TODO:

* Img size 512 and 1024?


# Revelations:




# Things to check, just before starting the model training:

* train_df_name
* model_name
* fold and total fold (for val %)
* npy_folder_name for dataloader's __getitem__() function
* are you resampling images?
* self.size, self.top_lr, self.std, self.mean -> insta trained weights used so be careful
* self.ep2unfreeze
*



# Observations:

# NOTES:

# Files informations:

* data/train_png: png version of dicom images
* data/npy_masks: contains indices where masks are 1
* data/npy_train: npy files of `train_png`
* data/npy_train_256: same as above, cv2.resized to 256
* data/npy_masks_256: masks cv2.resized to 256, new value points generated in between 0 and 1, so thresholded at 0.5, not indices


# remarks shortcut/keywords

# Experts speak


