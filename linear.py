import tensorflow as tf

from data_gen.linear_gen import LinearGen
from models.linear_model import LinearModel
from metrics.mse import MSE

def main():
    X, y = LinearGen().create_batch([10, 6.8], 5, 0.01, 512)
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

if __name__ == '__main__':
    main()

