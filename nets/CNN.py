import tensorflow as tf
import keras

# link Neural Network: https://keras.io/api/applications/

def EfficientNetV2M(image_length=224, training=False):
  inputs = keras.Input(shape=(image_length, image_length, 1))
  base_model = tf.keras.applications.EfficientNetV2M(include_top=True,
                                                      weights=None,
                                                      input_shape=(224, 224, 1),
                                                      classes=4,
                                                      classifier_activation="softmax",
                                                      include_preprocessing=True,)
  x = base_model(inputs, training=training)
  return x

def NASNetLarge(image_length=224, training=False):
  inputs = keras.Input(shape=(image_length, image_length, 1))
  base_model = tf.keras.applications.NASNetLarge(include_top=True,
                                                      weights=None,
                                                      input_shape=(224, 224, 1),
                                                      classes=4,
                                                      classifier_activation="softmax",
                                                      include_preprocessing=True,)
  x = base_model(inputs, training=training)
  return x

def InceptionResNetV2(image_length=224, training=False):
  inputs = keras.Input(shape=(image_length, image_length, 1))
  base_model = tf.keras.applications.InceptionResNetV2(include_top=True,
                                                      weights=None,
                                                      input_shape=(224, 224, 1),
                                                      classes=4,
                                                      classifier_activation="softmax",
                                                      include_preprocessing=True,)
  x = base_model(inputs, training=training)
  return x

def ResNet152V2(image_length=224, training=False):
  inputs = keras.Input(shape=(image_length, image_length, 1))
  base_model = tf.keras.applications.ResNet152V2(include_top=True,
                                                      weights=None,
                                                      input_shape=(224, 224, 1),
                                                      classes=4,
                                                      classifier_activation="softmax",
                                                      include_preprocessing=True,)
  x = base_model(inputs, training=training)
  return x
  
