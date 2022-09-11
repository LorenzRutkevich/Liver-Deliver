import keras.backend as K
import numpy as np
from PIL import Image
import cv2


def tumor_size(mask):
    return np.sum(mask)

def tumor_size_percentage(mask):
    return np.sum(mask) / (mask.shape[1] * mask.shape[1])


def tumor_surface_percentage(mask):
    return np.sum(mask) / 4167

def colorize_mask(mask):
    mask *= 255
    mask = mask.astype(np.uint8)
    mask = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
#  mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask[mask < 255] = 0
    return mask
    
def make_mask_visible(mask):
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return mask

def make_gray(img):
    img = np.array(Image.fromarray(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def predict(img, model):
    img = np.array(img)
    img = make_gray(img)
    img = img.reshape(1, img.shape[0], img.shape[1])
    img = img / 255
    mask = model.predict(img)
#  mask = make_mask_visible(mask)
    return mask

def predict_(img, model):
    img = np.array(img)
    img = make_gray(img)
    img = img.reshape(1, img.shape[0], img.shape[1])
    img = img / 255
    mask = model.predict(img)
    return mask

def att_res_pred(img, model):
    img = np.array(img)
    img = make_gray(img)
    img = img.reshape(1, img.shape[0], img.shape[1])
    img = img / 255
    mask = model.predict(img)
    return mask

def att_pred(img, model):
    img = np.array(img)
    img = make_gray(img)
    img = img.reshape(1, img.shape[0], img.shape[1])
    img = img / 255
    mask = model.predict(img)
    return mask

def dice_score(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

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
