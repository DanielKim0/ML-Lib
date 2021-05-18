import tensorflow as tf
from comp.activation import relu
from comp.models import linreg
from comp.functions import split_func
from metrics.class_accuracy import ClassAccuracy
from utils.compress import *
from .tf_model import TFModel


class MLPGenericModel(TFModel):
    def __init__(self):
        super().__init__()

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def save(self):
        super().save()

    def load(self):
        pass

    def validate_fit(self):
        pass

    def validate_predict(self):
        pass

    def build_model(self, dims, w_mean, w_stddev):
        # dims: [inputs, ..., hiddens, ..., outputs], with inputs inferred from X in fit()
        self.w = []
        self.b = []
        
        for i in range(len(dims)-1):
            self.w.append(tf.Variable(tf.random.normal((dims[i], dims[i+1]), mean=w_mean, stddev=w_stddev), trainable=True))
            self.b.append(tf.Variable(tf.zeros(dims[i+1]), trainable=True))

        def net(X, w, b, act):
            for i in range(len(w)-1):
                X = act[i](tf.matmul(X, w[i]) + b[i])
            return tf.matmul(X, w[-1]) + b[-1]
        self.model = net

    def fit(self, X, y, dims, loss, opt, act=relu, batch_size=16, num_epochs=32, mean=0, stddev=.1):
        # data casts
        X = tf.cast(tf.constant(X), tf.float32)

        # setting instance variables
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dims = dims
        self.dims.insert(0, X.shape[1])
        self.opt = opt
        self.loss = loss
        if not isinstance(act, list):
            self.act = [act] * (len(self.dims) - 1)
        else:
            self.act = act

        # build, validate, fit
        self.build_model(self.dims, mean, stddev)
        # self.validate_fit(X, y, act)
        super().fit(X, y)

    def train_epoch(self, X, y):
        for X_batch, y_batch in self.data_iter(X, y):
            self.train_step(X_batch, y_batch)
        train_l = self.loss.compare(y, self.model(X, self.w, self.b, self.act))
        print(f"epoch {self.curr_epoch}, loss {float(tf.reduce_mean(train_l)):f}")

    def train_step(self, X, y):
        # print(self.model(X, self.w, self.b, self.act))
        with tf.GradientTape() as g:
            l = self.loss.compare(y, self.model(X, self.w, self.b, self.act))
        grads = g.gradient(l, self.w + self.b)
        # print(grads)
        self.opt.update(self.w, grads[:int(len(grads)/2)], self.batch_size)
        self.opt.update(self.b, grads[int(len(grads)/2):], self.batch_size)

    def predict(self, X):
        return self.model(X, self.w, self.b)
