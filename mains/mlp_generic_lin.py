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
    # model.save("test.logistic")

    # model fitting
    # print(model)
    model.fit(X, y, [32, 16, 1], loss, opt)
    # print(model)

if __name__ == '__main__':
    main()
