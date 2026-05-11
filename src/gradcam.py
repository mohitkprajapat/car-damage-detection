import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.cm as cm
from PIL import Image

from src import config


# find the last conv layer searching nested models too
# returns the conv layer object (inside backbone if nested)
# and the backbone model it lives in (or None if top-level)
def _find_last_conv(model):
    last_conv, last_backbone = None, None
    for layer in model.layers:
        if isinstance(layer, (Conv2D, DepthwiseConv2D)):
            last_conv, last_backbone = layer, None
        elif hasattr(layer, "layers"):
            for sub in layer.layers:
                if isinstance(sub, (Conv2D, DepthwiseConv2D)):
                    last_conv, last_backbone = sub, layer
    return last_conv, last_backbone


# build a single flat grad_model: outer_input -> [conv_output, predictions]
# works by building backbone sub-model from its own input->conv_output,
# then stitching with the outer model's symbolic graph
def _make_grad_model(model):
    conv_layer, backbone = _find_last_conv(model)
    if conv_layer is None:
        raise ValueError("No Conv2D/DepthwiseConv2D found in model")

    if backbone is not None:
        # backbone is a nested functional model (e.g. ResNet50, VGG16, MobileNetV2)
        # build sub: backbone.input -> conv.output  (clean internal graph)
        conv_sub = tf.keras.Model(backbone.input, conv_layer.output)
        # stitch: outer_input -> [conv_sub(backbone.input_in_outer), model.output]
        # backbone.input in the outer graph = output of whichever layer feeds it
        # that tensor is backbone's inbound node input
        backbone_input_tensor = backbone.input  # symbolic, belongs to backbone's own graph

        # get the tensor from the OUTER model that feeds into backbone
        # it's stored in backbone._inbound_nodes[0].input_tensors
        try:
            outer_feed = backbone._inbound_nodes[0].input_tensors
            if isinstance(outer_feed, list):
                outer_feed = outer_feed[0]
        except (IndexError, AttributeError):
            outer_feed = None

        if outer_feed is not None:
            conv_out_tensor = conv_sub(outer_feed)
            grad_model = tf.keras.Model(model.input, [conv_out_tensor, model.output])
            return grad_model

    # top-level conv or fallback
    grad_model = tf.keras.Model(model.inputs, [conv_layer.output, model.output])
    return grad_model


# new — compute grad-cam heatmap
def get_heatmap(model, img_arr):
    grad_model = _make_grad_model(model)
    img_tensor = tf.cast(img_arr, tf.float32)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        tape.watch(conv_out)
        pred_idx = tf.argmax(preds[0])
        score = preds[:, pred_idx]

    grads = tape.gradient(score, conv_out)

    if grads is None:
        raise RuntimeError(
            "Grad-CAM: gradient is None. "
            "The conv layer output is not connected to the model output in the gradient graph."
        )

    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = conv_out[0] @ pooled[..., tf.newaxis]
    cam = tf.squeeze(cam)
    cam = tf.nn.relu(cam)

    hm = cam.numpy()
    hm = np.maximum(hm, 0)
    if hm.max() != 0:
        hm = hm / hm.max()

    hm = np.uint8(255 * hm)
    hm = np.array(Image.fromarray(hm).resize(config.img_shape, Image.BILINEAR))
    hm = hm / 255.0
    return hm


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
        "probs": {class_labels[i]: float(preds[i]) for i in range(len(class_labels))},
    }
