import tensorflow as tf

from data_gen.logistic_gen import LogisticGen
from models.logistic_model import LogisticModel
from metrics.cross_entropy import CrossEntropy
from metrics.class_accuracy import ClassAccuracy

def main():
    # data generation
    w_true = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    b_true = [1.5, 3, 2.5, 0.5]
    gen = LogisticGen(w_true, b_true, .1)
    X, y = gen.create_batch(512)

    # model initialization
    model = LogisticModel()
    loss = CrossEntropy()
    # model.save("test.logistic")

    # model fitting
    print(model)
    model.fit(X, y, len(b_true), loss, num_epochs=16, lr=0.3)
    print(model)
    
    # model prediction
    result = model.predict(X)
    print(tf.reduce_mean(loss.compare(y, result)))
    print(ClassAccuracy().compare(y, result))

    # saving and loading
    model.save("test.logistic")
    loaded = LogisticModel()
    loaded.load("test.logistic")
    
    # comparing saved and loaded models
    result_loaded = loaded.predict(X)
    tf.debugging.assert_equal(result, result_loaded)

    # test data
    X_new, y_new = gen.create_batch(512)
    result = model.predict(X_new)
    print(tf.reduce_mean(loss.compare(y_new, result)))
    print(ClassAccuracy().compare(y_new, result))

if __name__ == '__main__':
    main()
