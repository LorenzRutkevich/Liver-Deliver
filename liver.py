#!/usr/bin/env python
# coding: utf-8

# In[7]:


#-------------------------------------------------------------------------------------------------------------------#
#  Author: 			Lorenz Rutkevich															                    #
#  Created for:     Bundeswettbewerb Künstliche Intelligenz (BWKI)                                                  #
#                   https://www.bw-ki.de/                                                                           #
#                                                                                                                   #
#  Sources:                                                                                                         #
#                                                                                                                   #
#  - Data (original): https://competitions.codalab.org/competitions/17094                                           #
#  - Data (JPG-Format): https://www.kaggle.com/datasets/harshwardhanbhangale/lits-dataset-256x256-imgs              #
#  - U Net Paper: https://arxiv.org/pdf/1512.03385.pdf                                                              #
#  - R2Unet Paper: https://arxiv.org/abs/1802.06955                                                                 #
#                                                                                                                   #
#                                                                                                                   #
#                                                                                                                   #
#-------------------------------------------------------------------------------------------------------------------#


# In[1]:


#---------------------------------------------------------------------------------#
#                                   Imports                                       #
#---------------------------------------------------------------------------------#
import pickle
import os
import keras.layers
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from PIL import ImageEnhance
import tensorflow as tf
from tqdm import tqdm
import argparse
from keras.layers import *
import random
import cv2
from keras.metrics import *
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import plot_model
import keras.backend as K

###################
from models import *
from resunet import *
###################

import sys # bug fix for argparse
sys.argv=['']
del sys


#---------------------------------------------------------------------------------#
#                                   Arguments                                     #
#---------------------------------------------------------------------------------#

args = argparse.ArgumentParser()
args.add_argument('--train_img', type=str, default='/home/lorenz/U-Net/archive/train_images/') # path to train images
args.add_argument('--train_mask', type=str, default='/home/lorenz/U-Net/archive/train_masks/') # path to train masks
args.add_argument('--val_img', type=str, default='/home/lorenz/U-Net/archive/val_images/') # path to validation images
args.add_argument('--val_mask', type=str, default='/home/lorenz/U-Net/archive/val_masks/') # path to the validation masks
args.add_argument('--epochs', type=int, default=20) # number of epochs
args.add_argument('--batch_size', type=int, default=32) # batch size
args.add_argument('--predict', type=bool, default=False) # Change to True for prediction
args.add_argument('--augment', type=bool, default=False) # Change to True to augment the data
args.add_argument('--test_img', type=str, default='/home/lorenz/U-Net/archive/test_images/') # path to the image to predict
args.add_argument('--test_mask', type=str, default='/home/lorenz/U-Net/archive/test_masks/') # path to the mask to predict
args.add_argument('--img_width', type=int, default=128) # width of the image
args.add_argument('--img_height', type=int, default=128) # height of the image
args.add_argument('--img_channels', type=int, default=1) # channels of the image (RGB = 3), binary = 1, default RGB
args.add_argument('--save_augmentations', type=bool, default=True) # save the augmented images
args.add_argument('--base_dir', type=str, default='/home/lorenz/U-Net/archive') # path to the parent directory of the training data
args.add_argument('--model', type=str, default='AttResUnet(small)') # model to use
args.add_argument('--show_summary', type=bool, default=True) # show the summary of the model
args.add_argument('--measure', type=bool, default=True) # measure the predicted tumors
args.add_argument('--skip_train', type=int, default=000) # skip the training for the specified number of images
args.add_argument('--skip_val', type=int, default=000) # amount of images to skip in the validation set
args.add_argument('--skip_test', type=int, default=0) # amount of test images to skip
args.add_argument('--skip_paths', type=bool, default=False) # for direct loading of the paths
args.add_argument('--save_predictions', type=bool, default=True) # save predictions
args = args.parse_args() # parse the arguments to the args variable

from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())

if tf.test.gpu_device_name(): 
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF for better performance")

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session( config=config)


# In[2]:




#if args.skip_loading:
#    tf.config.run_functions_eagerly(True)

#---------------------------------------------------------------------------------#
#                                     Metrics                                     #
#---------------------------------------------------------------------------------#

def dice_score(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)
    #return 2 * (intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)    
    # 
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))    


# In[4]:


#---------------------------------------------------------------------------------#
#                                   Model Selection                               #
# The parameters are: input_shape=(img_width, height, channels), num_classes=x    #
#                        default: input_shape=(128,128,1)                         #
#---------------------------------------------------------------------------------#
if args.model == 'Unet': # Total params: 9,413,569
    model = Unet()
    c = True

elif args.model == 'AttUnet': # Total params: 9,718,565
    model = AttUnet()
    c = True

elif args.model == 'AttResUnet': # Total params: 4,478,574
    model = attention_res_unet()
    c = True

elif args.model == 'AttResUnet(small)': # Total params: 2,454,009
    model = att_res_unet()
    c = True 

elif args.model == 'RecUnet': # Total params: 20,423,137
    model = RecurrentUnet()
    c = True

elif args.model == 'DoubleUnet': # Total params: 18,061,281
    model = double_unet()
    c = True

elif args.model == 'SmallUnet': # Total params: 7,760,097
    model = small_unet()
    c = True

elif args.model == 'ResUnet': # Total params: 4,200,737
    model = res_unet()
    c = True

elif args.model == 'NestedUnet': # Total params: 51,922,497
    model = NestedUnet()
    c = True

elif args.model == 'DoubleAttResUnet': # Total params: 11,005,225
    model = att_res_unet_pp()
    c = True
    
else:
    print(f'Model "{args.model}" not found.\n'
          'Please choose one of the following models:\n'
          'Unet, AttUnet, RecUnet, DoubleUnet, ResUnet, NestedUnet, AttResUnet, AttResUnet(small), DoubleAttResUnet, SmallUnet')
    exit()
    
if c:
    model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy', keras.metrics.AUC(), dice_score, f1_m, precision_m, recall_m])
    plot_model(model, to_file=f'{args.model}.png', show_shapes=True)
    if args.show_summary:
        model.summary()


# In[11]:


#---------------------------------------------------------------------------------#
#           Loading in the paths if split correctly as shown below                #
#                                                                                 #
#           (base_dir)                                                            #
#           ----> Data                                                            #
#                     (sub_dirs)                                                  #
#                --> train_images                                                 #
#                --> train_masks                                                  #
#                --> val_images                                                   #
#                --> val_masks                                                    #
#                                                                                 #
#           Checking if the data is compatible and cutting things out if needed   #
#---------------------------------------------------------------------------------#

def get_paths(base_dir):
    if os.path.exists(base_dir):
        global train_images, train_masks, val_images, val_masks
        train_images = os.path.join(base_dir, 'train_images')
        train_masks = os.path.join(base_dir, 'train_masks')
        val_images = os.path.join(base_dir, 'val_images')
        val_masks = os.path.join(base_dir, 'val_masks')
        print(f"Summary: Train Data => Images: {len(os.listdir(train_images))}, Masks {len(os.listdir(train_masks))}\n "
              f"Val Data => Images: {len(os.listdir(val_images))}, Masks {len(os.listdir(val_masks))}")
        if len(os.listdir(train_images)) == len(os.listdir(train_masks)) and len(os.listdir(val_images)) == len(os.listdir(val_masks)):
            return train_images, train_masks, val_images, val_masks
        else:
            print("Error: Data is not equally sized, removing some images")
            if len(os.listdir(train_images)) > len(os.listdir(train_masks)):
                for i in os.listdir(train_images):
                    if not os.path.exists(os.path.join(train_masks, i)):
                        os.remove(os.path.join(train_images, i))
            elif len(os.listdir(train_images)) < len(os.listdir(train_masks)):
                for i in os.listdir(train_masks):
                    if not os.path.exists(os.path.join(train_images, i)):
                        os.remove(os.path.join(train_masks, i))
            if len(os.listdir(val_images)) > len(os.listdir(val_masks)):
                for i in os.listdir(val_images):
                    if not os.path.exists(os.path.join(val_masks, i)):
                        os.remove(os.path.join(val_images, i))
            elif len(os.listdir(val_images)) < len(os.listdir(val_masks)):
                for i in os.listdir(val_masks):
                    if not os.path.exists(os.path.join(val_images, i)):
                        os.remove(os.path.join(val_masks, i))
    else:
        raise Exception(f"{base_dir} does not exist")

# Always needs to be defined
# Parental folder where data to augment or data to load is in
# if you load in data without augmenting, the subfolders need to be named
# "train_images_augmented", "train_masks_augmented", "val_images_augmented", "val_masks_augmented", "test_images", "test_masks"
#########################
base_dir = args.base_dir
#########################
if args.augment:
    if args.skip_paths is False:
        get_paths(base_dir)



#---------------------------------------------------------------------------------#
#           Loading in the absolute paths if given as arguments                   #
#---------------------------------------------------------------------------------#

    if args.skip_paths:
        print("Loading data paths as stated in args")
        train_images = args.train_img
        train_masks = args.train_mask
        val_images = args.val_img
        val_masks = args.val_mask

#---------------------------------------------------------------------------------#
#                       Changing the values of the mask                           #
#---------------------------------------------------------------------------------#
def make_mask_visible(mask):
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return mask

#---------------------------------------------------------------------------------#
#                       Showing example masks and images                          #
#---------------------------------------------------------------------------------#

def show_example(img, mask):
  #  mask = make_mask_visible(mask)
    plt.figure(figsize=(12,12))
    plt.subplot(1,2,1)
    plt.title('Image',fontsize=14)
    plt.axis('off')
    plt.imshow(img, cmap='bone')
    plt.subplot(1,2,2)
    plt.title('Mask', fontsize=14)
    plt.axis('off')
    plt.imshow(mask, cmap='bone')
    plt.show()

#---------------------------------------------------------------------------------#
#              Summarizing the data if augmentation is not to be skipped          #
#---------------------------------------------------------------------------------#
if args.augment:
    for i in range (0, 2):
            show_example(img=np.array(Image.open(os.path.join(val_images, os.listdir(val_images)[i]))),
               mask=np.array(Image.open(os.path.join(val_masks, os.listdir(val_masks)[i]))))


# In[ ]:



def resize_img(img, mask, size):
    img = np.array(Image.fromarray(img).resize(size))
    mask = np.array(Image.fromarray(mask).resize(size))
    return img, mask

def rotate_img(img, mask, angle):
    img = np.array(Image.fromarray(img).rotate(angle))
    mask = np.array(Image.fromarray(mask).rotate(angle))
    return img, mask

def flip_img(img, mask):
    img = np.array(Image.fromarray(img).transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    mask = np.array(Image.fromarray(mask).transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    return img, mask

def make_gray(img):
    img = np.array(Image.fromarray(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def denoise(img):
    img = np.array(Image.fromarray(img))
    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    return img

def random_brightness(img):
    img = Image.fromarray(img)
    if np.random.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
    return np.array(img)

def random_contrast(img):
    img = Image.fromarray(img)
    if np.random.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))
    return np.array(img)

def sharpen(img):
    img = Image.fromarray(img)
    if np.random.random() < 0.5:
        img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.9, 1.1))
    return np.array(img)

def train_augmentations(img, mask):# resizing to 128, 128
    img, mask = resize_img(img, mask, (args.img_height, args.img_width))
    img, mask = rotate_img(img, mask, random.randint(5, 360)) # random rotation between 0 and 360 degrees
    img, mask = flip_img(img, mask) # flipping the image and mask
    img = denoise(img) # denoising the image
    img = random_brightness(img) # applying random brightness values
    img = make_gray(img) # changing rgb to binary (3 to 1 channels)
    img = random_contrast(img) # applying random contrast values
    img = sharpen(img) # sharpening the image
    #############################
   # mask = make_mask_visible(mask) # changing the mask values to 0 and 1
    return img, mask

def val_augmentations(img, mask):
    img, mask = resize_img(img, mask, (args.img_height, args.img_width)) # resizing to 128, 128
    img, mask = rotate_img(img, mask, random.randint(5, 360)) # random rotation between 0 and 360 degrees
    img, mask = flip_img(img, mask) # flipping the image and mask
    img = make_gray(img)
    img = random_contrast(img)
    img = denoise(img)
    img = sharpen(img)
    #############################
   # mask = make_mask_visible(mask)
    return img, mask

if args.augment:
    train_images_list, train_masks_list = [], []
    val_images_list, val_masks_list = [], []
    test_images_list, test_masks_list = [], []

    for i in tqdm(range(len(os.listdir(train_images))-args.skip_train), colour='#d44367', desc='Creating Train Data'):
        if i % 3 == 0:
            img = np.array(Image.open(os.path.join(train_images, os.listdir(train_images)[i])))
            mask = np.array(Image.open(os.path.join(train_masks, os.listdir(train_masks)[i])))
            img, mask = train_augmentations(img, mask)
            train_images_list.append(img)
            train_masks_list.append(mask)
        else:
            img = np.array(Image.open(os.path.join(train_images, os.listdir(train_images)[i])))
            mask = np.array(Image.open(os.path.join(train_masks, os.listdir(train_masks)[i])))
            img, mask = resize_img(img, mask, (args.img_height, args.img_width))
            img = make_gray(img)
            img = denoise(img)
    #        mask = make_mask_visible(mask)
            #############################
            train_images_list.append(img)
            train_masks_list.append(mask)
        if args.save_augmentations:
            if not os.path.exists(os.path.join(base_dir, 'train_images_augmented')):
                os.mkdir(os.path.join(base_dir, 'train_images_augmented'))
                os.mkdir(os.path.join(base_dir, 'train_masks_augmented'))
            plt.imsave(os.path.join(base_dir, 'train_images_augmented', os.listdir(train_images)[i]), img, cmap='gray')
            plt.imsave(os.path.join(base_dir, 'train_masks_augmented', os.listdir(train_masks)[i]), mask, cmap='gray')
        #-----Training data above -----# #-----Validation data below -----#
    for i in tqdm(range(len(os.listdir(val_images))-args.skip_val), colour='#3397da', desc='Creating Validation Data'):
        if i % 3 == 0:
            img = np.array(Image.open(os.path.join(val_images, os.listdir(val_images)[i])))
            mask = np.array(Image.open(os.path.join(val_masks, os.listdir(val_masks)[i])))
            img, mask = val_augmentations(img, mask)
            val_images_list.append(img)
            val_masks_list.append(mask)
        else:
            img = np.array(Image.open(os.path.join(val_images, os.listdir(val_images)[i])))
            mask = np.array(Image.open(os.path.join(val_masks, os.listdir(val_masks)[i])))
            img, mask = resize_img(img, mask, (args.img_height, args.img_width))
            img = make_gray(img)
     #       mask = make_mask_visible(mask)
            #############################
            img = denoise(img)
            val_images_list.append(img)
            val_masks_list.append(mask)
        if args.save_augmentations:
            if not os.path.exists(os.path.join(base_dir, 'val_images_augmented')):
                os.mkdir(os.path.join(base_dir, 'val_images_augmented'))
                os.mkdir(os.path.join(base_dir, 'val_masks_augmented'))
            plt.imsave(os.path.join(base_dir, 'val_images_augmented', os.listdir(val_images)[i]), img, cmap='gray')
            plt.imsave(os.path.join(base_dir, 'val_masks_augmented', os.listdir(val_masks)[i]), mask, cmap='gray')
    
    created_test_dir = False        
    if args.save_augmentations:
        if not os.path.exists(os.path.join(base_dir, 'test_images')):
            os.mkdir(os.path.join(base_dir, 'test_images'))
            os.mkdir(os.path.join(base_dir, 'test_masks'))
            created_test_dir = True

    # create test data
    for i in tqdm(range(int(len(train_images_list)*0.075)), colour='#33a068', desc='Creating Test Data from Training Set (Images)'):
        if i % 2 == 0:
            if created_test_dir:
                plt.imsave(os.path.join(base_dir, 'test_images', os.listdir(train_images)[i]), train_images_list[i], cmap='gray')
            test_images_list.append(train_images_list.pop(i))
    for i in tqdm(range(int(len(train_masks_list)*0.075)), colour='#33a068', desc='Creating Test Data from Training Set (Masks)'):
        if i % 2 == 0:
            if created_test_dir:
                plt.imsave(os.path.join(base_dir, 'test_masks', os.listdir(train_masks)[i]), train_masks_list[i], cmap='gray')
            test_masks_list.append(train_masks_list.pop(i))
    for i in tqdm(range(int(len(val_images_list)*0.075)), colour='#33a068', desc='Creating Test Data from Val Set (Images)'):
        if i % 2 == 0:
            if created_test_dir:
                plt.imsave(os.path.join(base_dir, 'test_images', os.listdir(val_images)[i]), val_images_list[i], cmap='gray')
            test_images_list.append(val_images_list.pop(i))
    for i in tqdm(range(int(len(val_masks_list)*0.075)), colour='#33a068', desc='Creating Test Data from Val Set (Masks)'):
        if i % 2 == 0:
            if created_test_dir:
                plt.imsave(os.path.join(base_dir, 'test_masks', os.listdir(val_masks)[i]), val_masks_list[i], cmap='gray')
            test_masks_list.append(val_masks_list.pop(i))


    train_images = np.array(train_images_list)
    train_masks = np.array(train_masks_list)
    val_images = np.array(val_images_list)
    val_masks = np.array(val_masks_list)
    test_images = np.array(test_images_list)
    test_masks = np.array(test_masks_list)

else:
    def load_augmented_data(skip_test, skip_val, skip_train):
        train_images = np.array([np.array(Image.open(os.path.join(base_dir, 'train_images_augmented', os.listdir(os.path.join(base_dir, 'train_images_augmented'))[i]))) for i in tqdm(range(len(os.listdir(os.path.join(base_dir, 'train_images_augmented')))-skip_train), colour='#d44367', desc='Loading Training Data (images)')])
        train_masks = np.array([np.array(Image.open(os.path.join(base_dir, 'train_masks_augmented', os.listdir(os.path.join(base_dir, 'train_masks_augmented'))[i]))) for i in tqdm(range(len(os.listdir(os.path.join(base_dir, 'train_masks_augmented')))-skip_train), colour='#d44367', desc='Loading Training Data (masks)')])
        val_images = np.array([np.array(Image.open(os.path.join(base_dir, 'val_images_augmented', os.listdir(os.path.join(base_dir, 'val_images_augmented'))[i]))) for i in tqdm(range(len(os.listdir(os.path.join(base_dir, 'val_images_augmented')))-skip_val), colour='#3397da', desc='Loading Validation Data (images)')])
        val_masks = np.array([np.array(Image.open(os.path.join(base_dir, 'val_masks_augmented', os.listdir(os.path.join(base_dir, 'val_masks_augmented'))[i]))) for i in tqdm(range(len(os.listdir(os.path.join(base_dir, 'val_masks_augmented')))-skip_val), colour='#3397da', desc='Loading Validation Data (masks)')])
        test_images = np.array([np.array(Image.open(os.path.join(base_dir, 'test_images', os.listdir(os.path.join(base_dir, 'test_images'))[i]))) for i in tqdm(range(len(os.listdir(os.path.join(base_dir, 'test_images')))-skip_test), colour='#33a068', desc='Loading Test Data (images)')])
        test_masks = np.array([np.array(Image.open(os.path.join(base_dir, 'test_masks', os.listdir(os.path.join(base_dir, 'test_masks'))[i]))) for i in tqdm(range(len(os.listdir(os.path.join(base_dir, 'test_masks')))-skip_test), colour='#33a068', desc='Loading Test Data (masks)')])
        train_images, train_masks = train_images[:,:,:,0], train_masks[:,:,:,0] # removing third dimension (1 --> amount of data) 
        val_images, val_masks = val_images[:,:,:,0], val_masks[:,:,:,0]
        test_images, test_masks = test_images[:,:,:,0], test_masks[:,:,:,0]
        train_images, train_masks = train_images / 255, train_masks / 255 # normalizing data
        val_images, val_masks = val_images / 255, val_masks / 255
        test_images, test_masks = test_images / 255, test_masks / 255
        #train_masks, val_masks, test_masks = make_mask_visible(train_masks), make_mask_visible(val_masks), make_mask_visible(test_masks)
        return train_images, train_masks, val_images, val_masks, test_images, test_masks

    train_images, train_masks, val_images, val_masks, test_images, test_masks = load_augmented_data(skip_test=args.skip_test, skip_val=args.skip_val, skip_train=args.skip_train)

#---------------------------------------------------------------------------------#
#                           Summarizing the data                                  #
#---------------------------------------------------------------------------------#
print(f"Train data shape: {train_images.shape}, train masks shape: {train_masks.shape} => Amount, Height, Width, Channels\n"
    f"Val data shape: {val_images.shape}, val masks shape: {val_masks.shape} => Amount, Height, Width, Channels\n"
    f"Test data shape: {test_images.shape}, test masks shape: {test_masks.shape} => Amount, Height, Width, Channels\n"
    f"Type: {type(train_images)}")
    
#---------------------------------------------------------------------------------#
#                         Showing augmented images                                #
#---------------------------------------------------------------------------------#


for i in range (0, 2):
    show_example(val_images[i], val_masks[i])


# In[13]:


#---------------------------------------------------------------------------------#
#         Training the model, saving the model and saving its history             #
#---------------------------------------------------------------------------------#
def show_plots(loss, accuracy, auc, dice_score, f1m, precision, recall):
    plt.plot(loss)
    plt.plot(accuracy)
    plt.plot(auc)
    plt.plot(dice_score)
    plt.plot(f1m)
    plt.plot(precision)
    plt.plot(recall)
    plt.title('model loss, accuracy, f1, precision, recall and AUC')
    plt.ylabel('loss, accuracy, dice_score , f1, precision, recall and AUC')
    plt.xlabel('epoch')
    plt.legend(['loss', 'accuracy', 'auc', 'dice_score', 'f1', 'precision', 'recall'], loc='upper left')
    plt.show()
    plt.savefig('loss_accuracy_auc.png')

checkpoint_path = './checkpoints/'
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

     
def train_model(model, epochs, batch_size, train_images, train_masks, val_images, val_masks, test_images, test_masks, save_model=True, save_history=True, save_predictions=True, save_plots=True, save_weights=True, save_model_name='model', save_history_name='history', save_predictions_name='predictions', save_plots_name='plots', save_weights_name='weights'):
    # creating callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, mode='min', min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath='model.h5', save_best_only=True, save_weights_only=True, verbose=1, mode='min'),
        tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch', profile_batch=2, embeddings_freq=0, embeddings_metadata=None)
        ]
    # training the model
    history = model.fit(train_images, train_masks, batch_size=batch_size, epochs=epochs, validation_data=(val_images, val_masks), callbacks=callbacks)
    # saving the model
    if save_model:
        model.save(f'{save_model_name}.h5')
    # saving the history
    if save_history:
        with open(f'{save_history_name}.pkl', 'wb') as f:
            pickle.dump(history.history, f)
    # saving the predictions
    if save_predictions:
        predictions = model.predict(test_images)
        np.save(f'{save_predictions_name}.npy', predictions)
    # saving the plots
    if save_plots:
        show_plots(history.history['loss'], history.history['accuracy'], history.history['auc'], history.history['dice_score'], history.history['f1_m'], history.history['precision_m'], history.history['recall_m'])
    # saving the weights
    if save_weights:
        model.save_weights(f'{save_weights_name}.h5')
    return history
        

history = train_model(model, epochs=args.epochs, batch_size=args.batch_size, train_images=train_images, train_masks=train_masks, val_images=val_images, val_masks=val_masks, test_images=test_images, test_masks=test_masks)

model.save(os.path.join(checkpoint_path, '{args.model}.h5'))
# saving the history
with open(f'history.pkl', 'wb') as f:
    pickle.dump(history.history, f)


    

#---------------------------------------------------------------------------------#
#                               Saving log files                                  #
#---------------------------------------------------------------------------------#

if not os.path.exists('./logs'):
    os.mkdir('./logs')

with open(os.path.join('./logs', f'{args.model}_history.pickle'), 'wb') as f:
    pickle.dump(history.history, f)





# In[ ]:


#---------------------------------------------------------------------------------#
#                           Predicting on the test images                         #
#---------------------------------------------------------------------------------#

def predict(model, test_images):
    predictions = model.predict(test_images)
   # predictions[predictions > 0.5] = 1
   # predictions[predictions <= 0.5] = 0
    
    predictions = np.array([cv2.applyColorMap(np.uint8(255 * predictions[i]), cv2.COLORMAP_HOT) for i in range(len(predictions))])
    #predictions = np.array([cv2.cvtColor(predictions[i], cv2.COLOR_BGR2RGB) for i in range(len(predictions))])
   #  uncomment for RGB, returns other colors however
    return predictions

if not os.path.exists('./predictions'):
    os.mkdir('./predictions')

model = tf.keras.models.load_model(os.path.join(checkpoint_path, f'{args.model}'), custom_objects={'dice_score': dice_score, 'K': K, 'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
for i in range(50, 105):
    predictions = predict(model, test_images[i:i+1])
    show_example(test_images[i], predictions[0])
    show_example(test_images[i], test_masks[i])
    cv2.imwrite(os.path.join('./predictions', f'{args.model}_{i}.png'), predictions[0])







# In[ ]:


#---------------------------------------------------------------------------------#
#              If a tumor can be seen, the amount of pixels it covers             #
#        will printed, as well as the amount of liver and picture it covers.      #
#              Uses masks that have been saved in the prediction folder           #
#---------------------------------------------------------------------------------#
def measure_tumor_size_in_pixels(mask):
    return np.sum(mask) / 255

# image has a dpi of 100 in x and 100 in y (whole image)
# 100 pixels per inch. 1 inch = 2.54 cm, 100 pixels per 2.54 cm, 1 pixel =(2.54/100) cm, 1 pixel = 0.0254 cm
# the amount of pixels in an average liver on the image is 4.176 pixels
    
def tumor_percentage(mask):
    return measure_tumor_size_in_pixels(mask) / (mask.shape[0] * mask.shape[1])


def tumor_liver_percentage(mask):
    return measure_tumor_size_in_pixels(mask) / 4167

if args.measure:
    for i in tqdm(range(len(os.listdir('./predictions/'))), colour='red', desc='Measuring Tumor Size'):
        mask = np.array(Image.open(os.path.join('./predictions', os.listdir(os.path.join('./predictions'))[i])))
        plt.imshow(mask, cmap='gray')
        plt.title(f'Mask_{i}', fontsize=14)
        plt.axis('off')
        plt.show()
        print(f'Tumor size in pixels on mask_{i}: {measure_tumor_size_in_pixels(mask)}')
        print(f'Percentage of the whole mask_{i} covered by the tumor: {tumor_percentage(mask) * 100}%')
        print(f'Percentage, the tumor covers of the liver on mask_{i} (roughly): {tumor_liver_percentage(mask) * 100}%')
        plt.show()

