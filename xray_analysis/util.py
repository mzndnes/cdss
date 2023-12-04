import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.preprocessing import image
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow import keras
import matplotlib.cm as cm
import tensorflow as tf
import csv
random.seed(a=None, version=2)


def get_mean_std_per_batch(image_path, df, H=320, W=320):
    sample_data = []
    for idx, img in enumerate(df["Image"].values):
        img_name = image_path + img
        sample_data.append(
            np.array(image.load_img(img_name, target_size=(H, W))))

    mean = np.mean(sample_data[0])
    std = np.std(sample_data[0])

    writer = open("mean_std.txt", 'w')
    writer.write(str(mean))
    writer.write(" ")
    writer.write(str(std))
    writer.close()
    return mean, std


def load_image(img, preprocess=True,W=320,H=320):
    """Load and preprocess image."""
    reader = open("mean_std.txt", 'r')
    line=reader.readline()
    stat=line.split(" ")
    mean=np.array(float(stat[0]))
    std=np.array(float(stat[1]))
    # mean, std = get_mean_std_per_batch(image_path, df, H=H, W=W)
    x = image.load_img(img, target_size=(H, W))
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x


def grad_cam(model, image, cls, layer_name, H=320, W=320):
    """GradCAM method for visualizing input saliency."""

    conv_output = model.get_layer(layer_name).output
    gradient_function = keras.models.Model([model.input], [conv_output, model.output])

    with tf.GradientTape() as tape:
        conv_output, preds = gradient_function(image)
        loss = preds[:, cls]

    grads = tape.gradient(loss, conv_output)
    output, grads_val = conv_output[0, :], grads[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))

    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam


def compute_gradcam(model, img_bytes, pid, hid, encounter_id, labels,
                    layer_name='bn'):

    preprocessed_input = load_image(img_bytes)
    preds = model.predict(preprocessed_input)


    top4_disease = np.take(labels, np.argsort(preds[0])[::-1][:4])
    cur_path = os.getcwd()
    cur_path=os.path.join(cur_path,"xray_heatmap")
    file_name = 'img_'+str(pid) + '_' + str(hid) + '_' + str(encounter_id)
    cdss_url='http://202.51.3.39/xray_heatmap/'
    j = 0
    imgs=list()
    for i in range(len(labels)):
        if labels[i] in top4_disease:
            gradcam = grad_cam(model, preprocessed_input, i, layer_name)
            img_name = file_name + labels[i] + '.png'
            img_url=cdss_url+img_name
            img_name = os.path.join(cur_path, img_name)
            imgs.append(img_url)
            alpha = min(0.5, preds[0][i])
            save_gradcam(img_bytes, gradcam, alpha, img_name)
            j += 1

    return preds[0], imgs
    # return preds[0]
def save_gradcam(img, heatmap, alpha,img_name):

    # Load the original image
    img = keras.utils.load_img(img)
    # img = keras.utils.img_to_array(img)
    # img=image.load_img(img, target_size=(H, W))
    img = keras.utils.img_to_array(img)
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(img_name)
