import tensorflow as tf

from data_gen.mnist_gen import MNISTGen
from models.sequential import SequentialModel
from metrics.class_accuracy import *
from metrics.cross_entropy import *
from optimizers.sgd import SGD
from layers.core.conv2d import *
from layers.core.dense import *
from layers.pooling import *
from layers.flatten import *
from comp.activation import *

def main():
    # data generation
    gen = MNISTGen()
    (X_train, y_train), (X_test, y_test) = gen.create_batch()
    X_train = tf.split(X_train, X_train.shape[0], axis=0)[0]
    y_train = tf.split(y_train, y_train.shape[0], axis=0)[0]

    # model initialization, LeNet architecture
    model = SequentialModel([
        Conv2DLayer(6, 5, act=sigmoid, padding="same"),
        Pooling(2, [2, 2], mode="average"),
        Conv2DLayer(16, 5, act=sigmoid),
        Pooling(2, [2, 2], mode="average"),
        Flatten(),
        DenseLayer(120, act="sigmoid"),
        DenseLayer(84, act="sigmoid"),
        DenseLayer(10, act="sigmoid"),
    ])
    loss = CrossEntropy()
    opt = SGD(.03)
    # model.save("test.logistic")

    # model fitting
    # print(model)
    model.fit(X_train, y_train, loss, opt)
    # print(model)

if __name__ == '__main__':
    main()
