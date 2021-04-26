import tensorflow as tf

from data_gen.logistic_gen import LogisticGen
from models.logistic_model import LogisticModel
from metrics.log_likelihood import *

def main():
    w_true = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    b_true = [1.5, 3, 2.5, 0.5]
    gen = LogisticGen(w_true, b_true, 5)
    X, y = gen.create_batch(512)
    model = LogisticModel()
    loss = CrossEntropy()

    print(model)
    model.fit(X, y, len(b_true), loss, num_epochs=16, lr=0.3)
    model.save("test.logistic")
    print(model)
    
    loaded = LogisticModel()
    loaded.load("test.logistic")
    
    result = model.predict(X)
    print(tf.reduce_mean(loss.compare(y, result)))
    print(ClassAccuracy().compare(y, result))

    result_loaded = loaded.predict(X)
    tf.debugging.assert_equal(result, result_loaded)

    X_new, y_new = gen.create_batch(512)
    result = model.predict(X_new)
    print(tf.reduce_mean(loss.compare(y_new, result)))
    print(ClassAccuracy().compare(y_new, result))

if __name__ == '__main__':
    main()
