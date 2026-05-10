import tensorflow as tf
from tensorflow.keras.applications import VGG16, MobileNetV2, ResNet50
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D


# original notebook code — do not modify
def transfer_model(hp):
    models = ['resnet50', 'mobnetv2', 'vgg16', 'resnet50-165', 'mobnetv2-143', 'vgg16-15']
    input_image = (224, 224, 3)
    inputs = tf.keras.Input(shape=input_image)
    base = hp.Choice('base model', models)
    bm_list = base.split('-')
    base_mod = bm_list[0]

    if len(bm_list) == 2:
        lay_num = int(bm_list[1])
    else:
        lay_num = None

    if base_mod == 'mobnetv2':
        mob_net_fine_tune_at = lay_num
        base_model = MobileNetV2(
           input_shape=input_image,
           include_top=False,
           weights='imagenet')
        base_model.trainable = True
        if mob_net_fine_tune_at is not None:
            for layer in base_model.layers[:mob_net_fine_tune_at]:
                layer.trainable = False
        else:
            base_model.trainable = False
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    elif base_mod == 'resnet50':
        resnet_fine_tune_at = lay_num
        base_model = ResNet50(
           input_shape=input_image,
           include_top=False,
           weights='imagenet')
        base_model.trainable = True
        if resnet_fine_tune_at is not None:
            for layer in base_model.layers[:resnet_fine_tune_at]:
                layer.trainable = False
        else:
            base_model.trainable = False
        x = tf.keras.applications.resnet.preprocess_input(inputs)

    elif base_mod == 'vgg16':
        vggnet_fine_tune_at = lay_num
        base_model = VGG16(
           input_shape=input_image,
           include_top=False,
           weights='imagenet')
        base_model.trainable = True
        if vggnet_fine_tune_at is not None:
            for layer in base_model.layers[:vggnet_fine_tune_at]:
                layer.trainable = False
        else:
            base_model.trainable = False
        x = tf.keras.applications.vgg16.preprocess_input(inputs)

    x = base_model(x)

    dropout_rate = 0.2
    L2_reg_rate = 0.01

    x = GlobalAveragePooling2D()(x)
    x = Dropout(rate=dropout_rate)(x)

    x = Dense(units=512, 
              kernel_regularizer=tf.keras.regularizers.L2(l2=L2_reg_rate), 
              activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=dropout_rate)(x)

    x = Dense(units=128, 
              kernel_regularizer=tf.keras.regularizers.L2(l2=L2_reg_rate), 
              activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=dropout_rate)(x)

    outputs = Dense(units=3, 
                    kernel_regularizer=tf.keras.regularizers.L2(l2=L2_reg_rate), 
                    activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.01
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy', 'recall', 'f1_score', 'precision'])
    return model
