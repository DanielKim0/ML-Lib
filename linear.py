import tensorflow as tf

from data_gen.linear_gen import LinearGen
from models.linear_model import LinearModel
from metrics.mse import MSE

def main():
    gen = LinearGen([10, 6.8], 5, 0.01)
    X, y = gen.create_batch(512)
    model = LinearModel()
    model.save("test.linear")

    print(model)
    model.fit(X, y, True)
    print(model)

    result = model.predict(X)
    loss = MSE().compare(y, result)
    print(loss)

    model.save("test.linear")
    loaded = LinearModel()
    loaded.load("test.linear")
    tf.debugging.assert_equal(result, loaded.predict(X))

    X_new, y_new = gen.create_batch(512)
    result = model.predict(X_new)
    print(loss.compare(y_new, result))

if __name__ == '__main__':
    main()

