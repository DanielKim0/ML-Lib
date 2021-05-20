import tensorflow as tf

from data_gen.polynomial_gen import PolynomialGen
from models.sequential import SequentialModel
from metrics.mse import MSE
from optimizers.sgd import SGD
from layers.core.dense import DenseLayer
from comp.activation import relu

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

    # model fitting
    print(model)
    model.fit(X, y, loss, opt)
    print(model)

    # model prediction
    result = model.predict(X)
    print(loss.compare(y, result))

    # saving and loading
    model.save("test.sequential")
    loaded = SequentialModel()
    loaded.load("test.sequential")
    
    # comparing saved and loaded models
    result_loaded = loaded.predict(X)
    tf.debugging.assert_equal(result, result_loaded)

    # test data
    X_new, y_new = gen.create_batch(512)
    result = model.predict(X_new)
    print(loss.compare(y_new, result))

if __name__ == '__main__':
    main()
