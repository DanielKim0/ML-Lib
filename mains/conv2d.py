import tensorflow as tf
from comp.activation import sigmoid
from data_gen.mnist_gen import MNISTGen
from metrics.class_accuracy import ClassAccuracy
from metrics.cross_entropy import CrossEntropy
from models.sequential import SequentialModel
from layers.core.conv2d import Conv2DLayer
from layers.core.dense import DenseLayer
from layers.pooling import PoolingLayer
from layers.flatten import FlattenLayer
from optimizers.sgd import SGD

SPLIT = 6000

def main():
    # data generation
    gen = MNISTGen()
    (X_train, y_train), (X_test, y_test) = gen.create_batch()
    X_train = tf.split(X_train, int(X_train.shape[0]/SPLIT), axis=0)[0]
    y_train = tf.split(y_train, int(y_train.shape[0]/SPLIT), axis=0)[0]

    # model creation, with LeNet architecture
    model = SequentialModel([
        Conv2DLayer(6, 5, act=sigmoid, padding="same"),
        PoolingLayer(2, [2, 2], mode="average"),
        Conv2DLayer(16, 5, act=sigmoid),
        PoolingLayer(2, [2, 2], mode="average"),
        FlattenLayer(),
        DenseLayer(120, act=sigmoid),
        DenseLayer(84, act=sigmoid),
        DenseLayer(10, act=sigmoid),
    ])

    loss = CrossEntropy()
    opt = SGD(.3)

    # model fitting
    print(model)
    model.fit(X, y, loss, opt)
    print(model)

    # model prediction
    result = model.predict(X)
    print(loss.compare(y, result))

    # saving and loading
    model.save("test_lenet.sequential")
    loaded = SequentialModel()
    loaded.load("test_lenet.sequential")
    
    # comparing saved and loaded models
    result_loaded = loaded.predict(X)
    tf.debugging.assert_equal(result, result_loaded)

    # test data
    X_train = tf.split(X_train, int(X_train.shape[0]/SPLIT), axis=0)[1]
    y_train = tf.split(y_train, int(y_train.shape[0]/SPLIT), axis=0)[1]
    result = model.predict(X_new)
    print(loss.compare(y_new, result))


if __name__ == '__main__':
    main()
