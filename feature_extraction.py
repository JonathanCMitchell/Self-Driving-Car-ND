import pickle
import tensorflow as tf
import numpy as np
from keras.layers import Dense, Input, Flatten
from keras.models import Model
from helpers import preprocess_data

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
# CIFAR 10 classification set:
# VGG
# flags.DEFINE_string('training_file', 'vgg_cifar10_100_bottleneck_features_train.p',
#                     "Bottleneck features training file (.p)")
# flags.DEFINE_string('validation_file', 'vgg_cifar10_bottleneck_features_validation.p',
#                     "Bottleneck features validation file (.p)")

# Resnet
flags.DEFINE_string('training_file', "resnet_cifar10_100_bottleneck_features_train.p",
                    "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', "resnet_cifar10_bottleneck_features_validation.p",
                    "Bottleneck features validation file (.p)")
# Inception
# flags.DEFINE_string('training_file', 'inception_cifar10_100_bottleneck_features_train.p',
#                     "Bottleneck features training file (.p)")
#
# flags.DEFINE_string('validation_file', 'inception_cifar10_100_bottleneck_features_validation.p', \
#                     "Bottleneck features validation file (.p)")

# Traffic Sign classification set:
# flags.DEFINE_string('training_file', 'vgg_traffic_100_bottleneck_features_train.p',
#                     "Bottleneck features training file (.p)")
# flags.DEFINE_string('validation_file', 'vgg_traffic_bottleneck_features_validation.p',
#                     "Bottleneck features validation file (.p)")




flags.DEFINE_integer('epochs', 50, 'The number of epochs')
flags.DEFINE_integer('learning_rate', 0.001, 'The learning rate of the gradient descent optimizer')
flags.DEFINE_integer('batch_size', 256, 'The batch size for gradient descent updates')


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.
    Note: bottleneck refers to all the layers before the final output layer that actually does the classification
    Because each image is reused multiple times during training, we simply cache these bottleneck values on disk
    so we don't have to repeatedly calculate what class each of these images are again and again...


    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file: ", training_file)
    print("Validation file: ", validation_file)

    with open(training_file, 'rb') as t:
        train_data = pickle.load(t)
    with open(validation_file, 'rb') as t:
        validation_data = pickle.load(t)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']
    print('X_train shape: ', X_train.shape)

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data


    print('before load bottleneck')
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print('after load bottleneck')
    X_normalized, y_one_hot = preprocess_data(X_train, y_train)
    X_val_normalized, y_val_one_hot = preprocess_data(X_val, y_val)

    print('X_train.shape:', X_train.shape)

    print('X_train.shape[1:]: ', X_train.shape[1:])
    input_shape = X_train.shape[1:]
    print('y_train shape: ', y_train.shape)
    print('type X_train: ', type(X_train))
    print('dtype X_train: ', X_train.dtype)

    print('dtype X_train[0]: ', X_train[0].dtype)
    print('X_val.shape: ', y_train.shape)


    nb_classes = len(np.unique(y_train))

    input = Input(shape=X_train.shape[1:])
    x = Flatten()(input)
    x = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=input, output=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(X_normalized, y_one_hot, batch_size=FLAGS.batch_size, nb_epoch=FLAGS.epochs, validation_data= (X_val_normalized, y_val_one_hot), shuffle=True)
    test_score = model.evaluate(X_val_normalized, y_val_one_hot)
    print('metrics names \n:', model.metrics_names)
    print('test_score: \n', test_score) #[loss, accuracy]
# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
