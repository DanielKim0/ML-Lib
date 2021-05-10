import tensorflow as tf

from data_gen.polynomial_gen import PolynomialGen
from models.sequential import SequentialModel
from metrics.mse import MSE
from optimizers.sgd import SGD
from layers.core.dense import DenseLayer
from comp.activation import *

def main():
    # data generation
    gen = PolynomialGen(5, 3)
    X, y = gen.create_batch(512)

    # model initialization
    model = SequentialModel([
        DenseLayer(32, act=relu),
        DenseLayer(16, act=relu),
        DenseLayer(1)
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
