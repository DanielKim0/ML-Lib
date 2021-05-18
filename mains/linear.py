import tensorflow as tf
from data_gen.linear_gen import LinearGen
from metrics.mse import MSE
from models.linear_model import LinearModel

def main():
    # data generation
    gen = LinearGen([10, 6.8], 5, 0.01)
    X, y = gen.create_batch(512)

    # model initialization
    model = LinearModel()
    loss = MSE()
    # model.save("test.linear")

    # model fitting
    print(model)
    model.fit(X, y, True)
    print(model)

    # model prediction
    result = model.predict(X)
    print(loss.compare(y, result))

    # saving and loading
    model.save("test.linear")
    loaded = LinearModel()
    loaded.load("test.linear")

    # comparing saved and loaded models
    result_loaded = loaded.predict(X)
    tf.debugging.assert_equal(result, result_loaded)

    # test data
    X_new, y_new = gen.create_batch(512)
    result = model.predict(X_new)
    print(loss.compare(y_new, result))

if __name__ == '__main__':
    main()

