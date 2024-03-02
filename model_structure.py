import tensorflow as tf 
import os

from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
"""
def ChannelAttention(in_planes, ratio):
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        max_pool = tf.keras.layers.GlobalMaxPooling1D()

        fc = tf.keras.Sequential([
            tf.keras.layers.Conv1D( in_planes // ratio, 1, use_bias=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(in_planes, 1, use_bias=False)
        ])
        sigmoid = tf.keras.layers.Activation('sigmoid')

        avg_out = fc(avg_pool(x))
        max_out = fc(max_pool(x))
        out = avg_out + max_out
        return sigmoid(out)

def SpatialAttention(kernel_size):

        conv1 = tf.keras.layers.Conv1D(1, kernel_size, padding='same', use_bias=False)
        sigmoid = tf.keras.layers.Activation('sigmoid')

        avg_out = tf.reduce_mean(x, axis=1, keepdims=True)
        max_out = tf.reduce_max(x, axis=1, keepdims=True)
        x = tf.concat([avg_out, max_out], axis=1)
        x = conv1(x)
        return sigmoid(x)



def AttentionCNN(input_window_length):
        conv = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding1D(padding=(4, 5)),
            tf.keras.layers.Conv1D(30, 10, strides=1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.ZeroPadding1D(padding=(3, 4)),
            tf.keras.layers.Conv1D(30, 8, strides=1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.ZeroPadding1D(padding=(2, 3)),
            tf.keras.layers.Conv1D(40, 6, strides=1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.ZeroPadding1D(padding=(2, 2)),
            tf.keras.layers.Conv1D(50, 5, strides=1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.ZeroPadding1D(padding=(2, 2)),
            tf.keras.layers.Conv1D(50, 5, strides=1),
            tf.keras.layers.ReLU()
        ])

        ca = ChannelAttention(in_planes=50, ratio=4)
        sa = SpatialAttention(kernel_size=7)

        dense = tf.keras.Sequential([
            tf.keras.layers.Flatten()(sa * ca * conv(x),
            tf.keras.layers.Dense(1024, activation='relu')
        ])
        output_layer = tf.keras.layers.Dense(1, activation='linear')(dense)
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        return model
"""
def create_fc_layer_2d(in_planes=50, ratio=4):
    fc_layer_2d = models.Sequential([
        layers.Conv2D(in_planes // ratio, (1, 1), use_bias=False),
        layers.ReLU(),
        layers.Conv2D(in_planes, (1, 1), use_bias=False)
    ])
    return fc_layer_2d

def SpatialAttention(inputlayer):
    def spatial_attention(x):
        avg_pool = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(x)
        max_pool = tf.keras.layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(x)
        avg_out = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        max_out = tf.reduce_max(x, axis=[1, 2], keepdims=True)
        x = tf.concat([avg_pool, max_pool], axis=-1)

        conv1 = layers.Conv2D(1, 7, padding='same', strides=1, use_bias=False)
        sigmoid = layers.Activation('sigmoid')

        x = conv1(x)
        return sigmoid(x)

    return spatial_attention(inputlayer)


def resnet_layer(inputs, num_channels=50):
    x = layers.Conv2D(num_channels, (5, 1), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(num_channels, (5, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)

    # 调整输入张量的通道数以匹配输出张量
    if inputs.shape[-1] != num_channels:
        inputs = layers.Conv2D(num_channels, (1, 1), padding='same')(inputs)

    x = layers.add([x, inputs])
    x = layers.Activation('relu')(x)
    return x


def conv(input_layer, filters, kernel_size, strides, padding):
    conv_layer = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding)(input_layer)
    conv_layer = tf.keras.layers.BatchNormalization()(conv_layer)
    output_layer = tf.keras.layers.Activation("relu")(conv_layer)
    return output_layer


def create_model(input_window_length):

    '''Specifies the structure of a seq2point model using Keras' functional API.

    Returns:
    model (tensorflow.keras.Model): The uncompiled seq2point model.
    '''
    

    input_layer = tf.keras.layers.Input(shape=(input_window_length,))
    reshape_layer = tf.keras.layers.Reshape((1, input_window_length, 1))(input_layer)


    #default
    conv_layer_1 = tf.keras.layers.Convolution2D(filters=30, kernel_size=(10, 1), strides=(1, 1), padding="same", activation="relu")(reshape_layer)
    #conv_layer_1 = tf.keras.layers.BatchNormalization()(conv_layer_1)
    conv_layer_2 = tf.keras.layers.Convolution2D(filters=30, kernel_size=(8, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_1)
    conv_layer_3 = tf.keras.layers.Convolution2D(filters=40, kernel_size=(6, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_2)
    conv_layer_4 = tf.keras.layers.Convolution2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_3)
    conv_layer_5 = tf.keras.layers.Convolution2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_4)

    """conv_layer_1 = conv(reshape_layer, filters=30, kernel_size=(10, 1), strides=(1, 1), padding="same")
    conv_layer_2 = conv(conv_layer_1, filters=30, kernel_size=(8, 1), strides=(1, 1), padding="same")
    conv_layer_3 = conv(conv_layer_2, filters=40, kernel_size=(6, 1), strides=(1, 1), padding="same")
    conv_layer_4 = conv(conv_layer_3, filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same")
    conv_layer_5 = conv(conv_layer_4, filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same")
    """
    #残差块
    #conv_layer_5 = resnet_layer(conv_layer_3)

    # 对于自适应平均池化
    adaptive_avg_pool = tf.keras.layers.GlobalAveragePooling2D()(conv_layer_5)
    adaptive_avg_pool = tf.keras.layers.Reshape((1, 1, 50))(adaptive_avg_pool)
    # 对于自适应最大池化
    adaptive_max_pool = tf.keras.layers.GlobalMaxPooling2D()(conv_layer_5)
    adaptive_max_pool = tf.keras.layers.Reshape((1, 1, 50))(adaptive_max_pool)

    fc_layer_2d = create_fc_layer_2d()
    channelattention = tf.keras.activations.sigmoid(fc_layer_2d(adaptive_avg_pool)+fc_layer_2d(adaptive_max_pool))
    layer_6 = conv_layer_5 * channelattention

    spatial_attention = SpatialAttention(layer_6)
    layer_7 = layer_6 * spatial_attention

    flatten_layer = tf.keras.layers.Flatten()(conv_layer_5)
    label_layer = tf.keras.layers.Dense(1024, activation="relu")(flatten_layer)
    output_layer = tf.keras.layers.Dense(1, activation="linear")(label_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model




def save_model(model, network_type, algorithm, appliance, save_model_dir):

    """ Saves a model to a specified location. Models are named using a combination of their 
    target appliance, architecture, and pruning algorithm.

    Parameters:
    model (tensorflow.keras.Model): The Keras model to save.
    network_type (string): The architecture of the model ('', 'reduced', 'dropout', or 'reduced_dropout').
    algorithm (string): The pruning algorithm applied to the model.
    appliance (string): The appliance the model was trained with.

    """
    
    #model_path = "saved_models/" + appliance + "_" + algorithm + "_" + network_type + "_model.h5"
    model_path = save_model_dir

    if not os.path.exists (model_path):
        open((model_path), 'a').close()

    model.save(model_path)

def load_model(model, network_type, algorithm, appliance, saved_model_dir):

    """ Loads a model from a specified location.

    Parameters:
    model (tensorflow.keras.Model): The Keas model to which the loaded weights will be applied to.
    network_type (string): The architecture of the model ('', 'reduced', 'dropout', or 'reduced_dropout').
    algorithm (string): The pruning algorithm applied to the model.
    appliance (string): The appliance the model was trained with.

    """

    #model_name = "saved_models/" + appliance + "_" + algorithm + "_" + network_type + "_model.h5"
    model_name = saved_model_dir
    print("PATH NAME: ", model_name)

    model = tf.keras.models.load_model(model_name)
    num_of_weights = model.count_params()
    print("Loaded model with ", str(num_of_weights), " weights")
    return model