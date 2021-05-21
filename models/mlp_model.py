import tensorflow as tf
from comp.activation import relu
from comp.models import linreg
from comp.functions import split_func
from metrics.class_accuracy import ClassAccuracy
from utils.compress import *
from .tf_model import TFModel


class MLPModel(TFModel):
    def __init__(self):
        super().__init__()

    def __str__(self):
        s = "Multilayer Perceptron Model\n"
        if self.model_fit:
            s += "Currently fit\n"
            s += f"w1.shape: {self.w1.shape}\n"
            s += f"w2.shape: {self.w2.shape}\n"
            s += f"epochs: {self.num_epochs}/{self.curr_epoch}\n"
            s += f"loss: {self.loss}\n"
            s += f"opt: {self.opt}\n"
            s += f"act: {self.act}\n"
            s += f"batch_size: {self.batch_size}\n"
        else:
            s += "Currently not fit\n"
        return s

    def __repr__(self):
        if self.model_fit:
            s = f"MLPModel(model_fit={True}, w1.shape={self.w1.shape}, w2.shape={self.w2.shape}, loss={self.loss}, opt={self.opt}, act={self.act}, batch_size={self.batch_size}, num_epochs={self.num_epochs}, curr_epoch={self.curr_epoch})"
        else:
            s = f"MLPModel(model_fit={False})"
        return s

    def save(self, path):
        super().save()
        data = {
            "loss": self.loss,
            "hiddens": self.hiddens,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "curr_epoch": self.curr_epoch,
            "opt": self.opt,
            "act": self.act,
        }
        arrays = {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }
        compress_files(path, data, arrays)

    def load(self, path):
        data, arrays = uncompress_files(path)

        self.loss = data["loss"]
        self.opt = data["opt"]
        self.act = data["act"]
        self.num_epochs = data["num_epochs"]
        self.curr_epoch = data["curr_epoch"]
        self.hiddens = data["hiddens"]
        self.model = self.create_net()

        self.w1 = arrays["w1"]
        self.b1 = arrays["b1"]
        self.w2 = arrays["w2"]
        self.b2 = arrays["b2"]

    def validate_fit(self, X, y, batch_size, num_epochs):
        if X.shape[0] != y.shape[0]:
            raise ValueError("")
        if batch_size <= 0 or not isinstance(batch_size, int):
            raise ValueError("")
        if num_epochs <= 0 or not isinstance(num_epochs, int):
            raise ValueError("")

    def validate_predict(self, X):
        if X.shape[1] != self.w.shape[0]:
            raise ValueError("")

    def create_net(self):
        def net(X, w1, b1, w2, b2, act):
            h = act(tf.matmul(X, w1) + b1)
            o = tf.matmul(h, w2) + b2
            return o
        return net

    def build_model(self, inputs, hiddens, outputs, w_mean, w_stddev):
        self.w1 = tf.Variable(tf.random.normal((inputs, hiddens), mean=w_mean, stddev=w_stddev), trainable=True)
        self.b1 = tf.Variable(tf.zeros(hiddens), trainable=True)
        self.w2 = tf.Variable(tf.random.normal((hiddens, outputs), mean=w_mean, stddev=w_stddev), trainable=True)
        self.b2 = tf.Variable(tf.zeros(outputs), trainable=True)
        self.model = self.create_net()

    def fit(self, X, y, hiddens, outputs, loss, opt, act=relu, batch_size=16, num_epochs=32, mean=0, stddev=.1):
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
        self.validate_model(stddev)
        self.build_model(X.shape[1], self.hiddens, outputs, mean, stddev)
        self.validate_fit(X, y, batch_size, num_epochs, opt, loss)
        super().fit(X, y)

    def train_epoch(self, X, y):
        for X_batch, y_batch in self.data_iter(X, y):
            self.train_step(X_batch, y_batch)
        train_l = self.loss.compare(y, self.model(X, self.w1, self.b1, self.w2, self.b2, self.act), from_logits=True)
        print(f"epoch {self.curr_epoch}, loss {float(tf.reduce_mean(train_l)):f}, accuracy {float(ClassAccuracy().compare(y, self.model(X, self.w1, self.b1, self.w2, self.b2, self.act)))}")

    def train_step(self, X, y):
        with tf.GradientTape() as g:
            l = self.loss.compare(y, self.model(X, self.w1, self.b1, self.w2, self.b2, self.act), from_logits=True)
        dw1, db1, dw2, db2 = g.gradient(l, [self.w1, self.b1, self.w2, self.b2])
        self.opt.update([self.w1, self.b1, self.w2, self.b2], [dw1, db1, dw2, db2], self.batch_size)

    def predict(self, X):
        self.validate_predict(X)
        return self.model(X, self.w1, self.b1, self.w2, self.b2, self.act)
