import tensorflow as tf

from data_gen.linear_gen import LinearGen
from models.linear_tf_model import LinearTFModel
from metrics.mse import MSE
from optimizers.sgd import SGD

def main():
    # data generation
    gen = LinearGen([10, 6.8], 5, 0.01)
    X, y = gen.create_batch(512)

    # model initialization
    model = LinearTFModel()
    loss = MSE()
    opt = SGD(0.3)
    # model.save("test.lineartf")

    # model fitting
    print(model)
    model.fit(X, y, loss, opt, num_epochs=16)
    print(model)

    # model prediction
    result = model.predict(X)
    print(loss.compare(y, result))
    
    # saving and loading
    model.save("test.lineartf")
    loaded = LinearTFModel()
    loaded.load("test.lineartf")
    
    # comparing saved and loaded models
    result_loaded = loaded.predict(X)
    tf.debugging.assert_equal(result, result_loaded)

    # test data
    X_new, y_new = gen.create_batch(512)
    result = model.predict(X_new)
    print(loss.compare(y_new, result))

if __name__ == '__main__':
    main()
