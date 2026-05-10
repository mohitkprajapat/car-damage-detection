import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from src import config


# new — find last conv layer and compute grad-cam heatmap
def get_heatmap(model, img_arr):
    # find last conv layer
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, (Conv2D, DepthwiseConv2D)):
            last_conv = layer.name
            break
        # check inside nested models (e.g. mobilenetv2 base)
        if hasattr(layer, 'layers'):
            for sub in reversed(layer.layers):
                if isinstance(sub, (Conv2D, DepthwiseConv2D)):
                    last_conv = sub.name
                    # build sub-model to expose this layer
                    break
            if last_conv:
                break

    # build grad model up to last conv layer output + final predictions
    grad_model = _build_grad_model(model, last_conv)

    img_tensor = tf.cast(img_arr, tf.float32)

    with tf.GradientTape() as tape:
        fmaps, preds = grad_model(img_tensor)
        pred_idx = tf.argmax(preds[0])
        score = preds[:, pred_idx]

    grads = tape.gradient(score, fmaps)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

    cam = fmaps[0] @ pooled[..., tf.newaxis]
    cam = tf.squeeze(cam)
    cam = tf.nn.relu(cam)

    # resize and normalize
    hm = cam.numpy()
    hm = np.maximum(hm, 0)
    if hm.max() != 0:
        hm = hm / hm.max()

    hm = np.uint8(255 * hm)
    hm = np.array(Image.fromarray(hm).resize(config.img_shape, Image.BILINEAR))
    hm = hm / 255.0
    return hm


def _build_grad_model(model, last_conv_name):
    # walk model layers to find output of last conv
    for layer in model.layers:
        if layer.name == last_conv_name:
            return tf.keras.Model(model.inputs, [layer.output, model.output])
        if hasattr(layer, 'layers'):
            for sub in layer.layers:
                if sub.name == last_conv_name:
                    # expose via sub-model
                    output = layer.output if hasattr(layer, 'output') else model.layers[1].output
                    sub_out = tf.keras.Model(layer.input, sub.output)(output)
                    # rebuild: input -> conv_out, input -> final_out
                    return tf.keras.Model(model.inputs, [sub_out, model.output])
    # fallback: use last layer before dense head
    raise ValueError(f"Could not find layer: {last_conv_name}")


# new — overlay heatmap on original image
def overlay_heatmap(orig_arr, hm):
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_hm = jet_colors[np.uint8(hm * 255)]
    jet_hm = np.uint8(jet_hm * 255)

    orig = np.uint8(orig_arr[0])
    overlay = np.uint8(jet_hm * 0.4 + orig * 0.6)
    return Image.fromarray(overlay)


# new — full pipeline: path -> prediction + grad-cam dict
def analyze(model, img_path, class_labels):
    img = load_img(img_path, target_size=config.img_shape)
    arr = img_to_array(img)
    arr_exp = np.expand_dims(arr, axis=0)

    preds = model.predict(arr_exp, verbose=0)[0]
    pred_idx = np.argmax(preds)
    pred_class = class_labels[pred_idx]
    confidence = float(preds[pred_idx])

    hm = get_heatmap(model, arr_exp)
    overlay = overlay_heatmap(arr_exp, hm)

    return {
        "overlay": overlay,
        "pred_class": pred_class,
        "confidence": confidence,
        "probs": {class_labels[i]: float(preds[i]) for i in range(len(class_labels))}
    }
