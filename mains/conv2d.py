import tensorflow as tf

from data_gen.mnist_gen import MNISTGen
from models.sequential import SequentialModel
from metrics.class_accuracy import *
from metrics.cross_entropy import *
from optimizers.sgd import SGD
from layers.core.conv2d import *
from layers.core.dense import *
from layers.pooling import *
from comp.activation import *

def main():
    # data generation
    gen = MNISTGen()
    (X_train, y_train), (X_test, y_test) = gen.create_batch()

    # model initialization
    model = SequentialModel([
        Conv2DLayer((nodes), kernel_size)
    ])
    loss = MSE()
    opt = SGD(.03)
    # model.save("test.logistic")

    # model fitting
    # print(model)
    model.fit(X, y, loss, opt)
    # print(model)

if __name__ == '__main__':
    main()
