import tensorflow as tf
from data_gen.polynomial_gen import PolynomialGen
from metrics.mse import MSE
from models.mlp_generic_model import MLPGenericModel
from optimizers.sgd import SGD

def main():
    # data generation
    gen = PolynomialGen(5, 3)
    X, y = gen.create_batch(512)

    # model initialization
    model = MLPGenericModel()
    loss = MSE()
    opt = SGD(.03)

    # model fitting
    print(model)
    model.fit(X, y, [32, 16, 1], loss, opt)
    print(model)

    # model prediction
    result = model.predict(X)
    print(loss.compare(y, result))

    # saving and loading
    model.save("test_lin.mlpgeneric")
    loaded = MLPGenericModel()
    loaded.load("test_lin.mlpgeneric")
    
    # comparing saved and loaded models
    result_loaded = loaded.predict(X)
    tf.debugging.assert_equal(result, result_loaded)

    # test data
    X_new, y_new = gen.create_batch(512)
    result = model.predict(X_new)
    print(loss.compare(y_new, result))

if __name__ == '__main__':
    main()
