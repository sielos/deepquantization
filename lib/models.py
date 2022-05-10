import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
import numpy as np
assert float(tf.__version__[:3]) >= 2.3


def get_mnist_model():
    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(units=100, activation='sigmoid'),
        tf.keras.layers.Dense(units=10, activation='sigmoid'),
        tf.keras.layers.Dense(10)
    ])

    # Train the digit classification model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])
    model.fit(
        train_images,
        train_labels,
        epochs=5,
        validation_data=(test_images, test_labels)
    )

    return model, train_images, train_labels


def get_cifar10_model():
    # # Neuron Network Metadata
    num_classes = 10
    input_shape = (32, 32, 3)

    # get cifar10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)


    model_cifar10 = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(units=100, activation='sigmoid'),
        tf.keras.layers.Dense(units=10, activation='sigmoid'),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model_cifar10.summary()

    batch_size = 128
    epochs = 15

    model_cifar10.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model_cifar10.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    return model_cifar10, x_train, y_train


def get_cad_model():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import Adam
    # # Neuron Network Metadata
    num_classes = 2
    input_shape = (224, 224, 3)

    # get cifar10 dataset
    train_path = 'Data/train'
    valid_path = 'Data/valid'
    test_path = 'Data/test'

    train_batches = ImageDataGenerator(rescale=1./255, preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory=train_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=20)
    valid_batches = ImageDataGenerator(rescale=1./255, preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory=valid_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)

    imgs, labels = next(train_batches)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(units=100, activation='sigmoid'),
        tf.keras.layers.Dense(units=10, activation='sigmoid'),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.summary()

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=train_batches, steps_per_epoch=len(train_batches), validation_data=valid_batches,
                      validation_steps=len(valid_batches), epochs=12, verbose=2)


    train_batches = ImageDataGenerator(rescale=1./255, preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory=train_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=400)
    imgs, labels = next(train_batches)

    return model, imgs, labels
