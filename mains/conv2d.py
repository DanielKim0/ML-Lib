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

def main():
    # data generation
    gen = MNISTGen()
    (X_train, y_train), (X_test, y_test) = gen.create_batch()
    X_train = tf.split(X_train, int(X_train.shape[0]/6000), axis=0)[0]
    y_train = tf.split(y_train, int(y_train.shape[0]/6000), axis=0)[0]

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
    # model.save("test.logistic")

    # model fitting
    # print(model)
    model.fit(X_train, y_train, loss, opt)
    # print(model)

if __name__ == '__main__':
    main()
