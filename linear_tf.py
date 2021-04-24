import tensorflow as tf

from data_gen.linear_gen import LinearGen
from models.linear_tf_model import LinearTFModel
from metrics.mse import MSE

def main():
    X, y = LinearGen().create_batch([10, 6.8], 5, 0.01, 512)
    model = LinearTFModel()
    loss = MSE()

    print(model)
    model.fit(X, y, loss, num_epochs=16)
    model.save("test.linear")
    print(model)
    
    loaded = LinearTFModel()
    loaded.load("test.linear")
    
    result = model.predict(X)
    print(MSE().compare(y, result))

    result_loaded = loaded.predict(X)
    tf.debugging.assert_equal(result, result_loaded)

if __name__ == '__main__':
    main()
