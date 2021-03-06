import tensorflow as tf
import keras
from tensorflow.keras.models import Model

###################################################### link Neural Network: https://keras.io/api/applications/  ###################################
# image shape: (224, 224, 1)
# number of class: 4
# including top: fully connected layers
# activation in final layer: softmax

def EfficientNetV2M(image_length=224, training=False):
    inputs = keras.Input(shape=(image_length, image_length, 1))
    base_model = tf.keras.applications.EfficientNetV2M(include_top=True,
                                                        weights=None,
                                                        input_shape=(224, 224, 1),
                                                        classes=4,
                                                        classifier_activation="softmax")
    output = base_model(inputs, training=training)
    x = Model(inputs, output)
    return x



def InceptionResNetV2(image_length=224, training=False):
    inputs = keras.Input(shape=(image_length, image_length, 1))
    base_model = tf.keras.applications.InceptionResNetV2(include_top=True,
                                                        weights=None,
                                                        input_shape=(224, 224, 1),
                                                        classes=4,
                                                        classifier_activation='softmax')
    output = base_model(inputs, training=training)
    x = Model(inputs, output)
    return x

def MobileNetV2(image_length=224, training=False):
    inputs = keras.Input(shape=(image_length, image_length, 1))
    base_model = tf.keras.applications.MobileNetV2(include_top=True,
                                                    weights=None,
                                                    input_shape=(224, 224, 1),
                                                    classes=4,
                                                    classifier_activation="softmax",
                                                )
    output = base_model(inputs, training=training)
    x = Model(inputs, output)
    return x

def ResNet152V2(image_length=224, training=False):
    inputs = keras.Input(shape=(image_length, image_length, 1))
    base_model = tf.keras.applications.ResNet152V2(include_top=True,
                                                        weights=None,
                                                        input_shape=(224, 224, 1),
                                                        classes=4,
                                                        classifier_activation='softmax')
    output = base_model(inputs, training=training)
    x = Model(inputs, output)
    return x
  
