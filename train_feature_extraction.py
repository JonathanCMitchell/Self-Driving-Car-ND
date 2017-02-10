import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle


n_classes = 43
# TODO: Load traffic signs data.
training_file = 'train.p'
with open(training_file, mode = 'rb') as f:
    train = pickle.load(f)

X_train = train['features']
y_train = train['labels']
# TODO: Split data into training and validation sets.
X_train, X_validation, y_train,  y_validation = train_test_split(X_train, y_train, \
                                            train_size = 0.80, test_size = 0.20)

print('X_train size: ', len(X_train[:]))
print('X_validation size: ', len(X_validation[:]))
print('y_train size: ', len(y_train))
print('y_validation size: ', len(y_validation))

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int32, [None])
# Resize operation (32, 32, 3) => (227, 227, 3)
resized = tf.image.resize_images(x, (227, 227))
one_hot_y = tf.one_hot(y, n_classes)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
fc8W = tf.Variable(tf.truncated_normal([4096, n_classes], stddev=1e-2), name='fc8w') # (4096, 43)
fc8B = tf.Variable(tf.zeros([n_classes]), name = 'fc8b') # 43
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8B)



# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

# Hyperparameters:
EPOCHS = 10
BATCH_SIZE = 256
rate = 0.001
beta = 0.001

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate= rate)

vars_to_train = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if var.name.startswith('fc8')]

training_operation = optimizer.minimize(loss_operation, var_list=vars_to_train)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TODO: Train and evaluate the feature extraction model.


# Define evaluate function
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset: offset + BATCH_SIZE], y_data[offset: offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# Define training function
saver = tf.train.Saver()
save_file = 'new-net-epoch10'
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    X_train, y_train = shuffle(X_train, y_train)
    num_examples = len(X_train)

    print('Training...')
    for i in range(EPOCHS):
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x_train, batch_y_train = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict= {x: batch_x_train, y: batch_y_train})

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
    saver.save(sess, save_file)
    print('Model saved in file %s', save_file)


