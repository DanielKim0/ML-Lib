import tensorflow as tf
from models.mlp_generic_model import MLPGenericModel
from data_gen.logistic_gen import LogisticGen
from metrics.cross_entropy import CrossEntropy
from metrics.class_accuracy import ClassAccuracy

def main2():
    # data generation
    w_true = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    b_true = [1.5, 3, 2.5, 0.5]
    gen = LogisticGen(w_true, b_true, .1)
    X, y = gen.create_batch(512)
    print(y)

    # model initialization
    model = MLPGenericModel()
    loss = CrossEntropy(default_logits=True)
    opt = SGD(.1)
    # model.save("test.logistic")

    # model fitting
    # print(model)
    model.fit(X, y, [16, len(w_true[0])], loss, opt, num_epochs=16)
    # print(model)

if __name__ == '__main__':
    main()
