import tensorflow as tf

from .tf_model import TFModel
from comp.activation import *
from utils.compress import *
from comp.models import linreg
from comp.functions import *
from metrics.class_accuracy import *

class MLPModel(TFModel):
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

    def build_model(self, inputs, hiddens, outputs, w_mean, w_stddev):
        self.w1 = tf.Variable(tf.random.normal((inputs, hiddens), mean=w_mean, stddev=w_stddev), trainable=True)
        self.b1 = tf.Variable(tf.zeros(hiddens), trainable=True)
        self.w2 = tf.Variable(tf.random.normal((hiddens, outputs), mean=w_mean, stddev=w_stddev), trainable=True)
        self.b2 = tf.Variable(tf.zeros(outputs), trainable=True)

        def net(X, w1, b1, w2, b2, act):
            h = act(tf.matmul(X, w1) + b1)
            o = tf.matmul(h, w2) + b2
            return o
        self.model = net

    def fit(self, X, y, hiddens, outputs, loss, opt, act=tanh, batch_size=16, num_epochs=32, mean=0, stddev=1):
        # data casts
        X = tf.cast(tf.constant(X), tf.float32)

        # setting instance variables
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.hiddens = hiddens
        self.opt = opt
        self.loss = loss
        self.act = act

        # build, validate, fit
        self.build_model(X.shape[1], self.hiddens, outputs, mean, stddev)
        # self.validate_fit(X, y, classes)
        super().fit(X, y)

    def train_epoch(self, X, y):
        for X_batch, y_batch in self.data_iter(X, y):
            self.train_step(X_batch, y_batch)
        train_l = self.loss.compare(y, self.model(X, self.w1, self.b1, self.w2, self.b2, self.act), from_logits=True)
        print(self.model(X, self.w1, self.b1, self.w2, self.b2, self.act))
        print(f"epoch {self.curr_epoch}, loss {float(tf.reduce_mean(train_l)):f}, accuracy {float(ClassAccuracy().compare(y, self.model(X, self.w1, self.b1, self.w2, self.b2, self.act)))}")

    def train_step(self, X, y):
        with tf.GradientTape() as g:
            l = self.loss.compare(y, self.model(X, self.w1, self.b1, self.w2, self.b2, self.act), from_logits=True)
        dw1, db1, dw2, db2 = g.gradient(l, [self.w1, self.b1, self.w2, self.b2])
        self.opt.update([self.w1, self.b1, self.w2, self.b2], [dw1, db1, dw2, db2], self.batch_size)

    def predict(self, X):
        return self.model(X, self.w, self.b)
