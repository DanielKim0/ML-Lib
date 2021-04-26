import tensorflow as tf

from data_gen.linear_gen import LinearGen
from models.linear_tf_model import LinearTFModel
from metrics.mse import MSE

def main():
    gen = LinearGen([10, 6.8], 5, 0.01)
    X, y = gen.create_batch(512)
    model = LinearTFModel()
    loss = MSE()

    print(model)
    model.fit(X, y, loss, num_epochs=16)
    model.save("test.lineartf")
    print(model)
    
    loaded = LinearTFModel()
    loaded.load("test.lineartf")
    
    result = model.predict(X)
    print(MSE().compare(y, result))

    result_loaded = loaded.predict(X)
    tf.debugging.assert_equal(result, result_loaded)

    X_new, y_new = gen.create_batch(512)
    result = model.predict(X_new)
    print(loss.compare(y_new, result))

if __name__ == '__main__':
    main()
