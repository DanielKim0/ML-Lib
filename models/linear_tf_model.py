from .tf_model import TFModel
import tensorflow as tf
from utils.compress import *
from comp.models import linreg

class LinearTFModel(TFModel):
    def __init__(self):
        super().__init__()

    def __str__(self):
        s = "Linear TF Model\n"
        if self.model_fit:
            s += "Currently fit\n"
            s += f"w.shape: {self.w.shape[0]}\n"
            s += f"loss: {self.loss}\n"
            s += f"opt: {self.opt}\n"
            s += f"epochs: {self.num_epochs}/{self.curr_epoch}\n"
        else:
            s += "Currently not fit\n"
        return s

    def __repr__(self):
        if self.model_fit:
            s = f"LinearTFModel(model_fit={False})"
        else:
            s = f"LinearTFModel(model_fit={True}, w={self.w}, b={self.b}, loss={self.loss}, opt={self.opt}, batch_size={self.batch_size}, num_epochs={self.num_epochs}, curr_epoch={self.curr_epoch}, model={self.model}, update={self.update})"
        return s

    def save(self, path):
        super().save()
        data = {
            "loss": self.loss,
            "opt": self.opt,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "curr_epoch": self.curr_epoch,
            "model": self.model,
        }
        arrays = {
            "w": self.w,
            "b": self.b,
        }
        compress_files(path, data, arrays)

    def load(self, path):
        data, arrays = uncompress_files(path)
        self.loss = data["loss"]
        self.opt = data["opt"]
        self.num_epochs = data["num_epochs"]
        self.curr_epoch = data["curr_epoch"]
        self.model = data["model"]
        self.w = arrays["w"]
        self.b = arrays["b"]

    def validate_fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("")
        if y.shape[1] != 1:
            raise ValueError("")

    def validate_predict(self, X):
        if X.shape[1] != self.w.shape[0]:
            raise ValueError("")

    def build_model(self, w_size, w_mean, w_stddev):
        self.w = tf.Variable(tf.random.normal(shape=(w_size, 1), mean=w_mean, stddev=w_stddev), trainable=True)
        self.b = tf.Variable(tf.zeros(1), trainable=True)
        self.model = linreg

    def fit(self, X, y, loss, opt, batch_size=16, num_epochs=32, mean=0, stddev=0.01):
        # data casts
        X = tf.cast(tf.constant(X), tf.float32)
        y = tf.cast(tf.constant(y), tf.float32)

        # setting instance variables
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.opt = opt
        self.loss = loss
        
        # build, validate, fit
        self.build_model(X.shape[1], mean, stddev)
        self.validate_fit(X, y)
        super().fit(X, y)

    def train_epoch(self, X, y):
        for X_batch, y_batch in self.data_iter(X, y):
            self.train_step(X_batch, y_batch)
        train_l = self.loss.compare(self.model(X, self.w, self.b), y)
        print(f'epoch {self.curr_epoch}, loss {float(tf.reduce_mean(train_l)):f}')

    def train_step(self, X, y):
        with tf.GradientTape() as g:
            l = self.loss.compare(y, self.model(X, self.w, self.b))
        dw, db = g.gradient(l, [self.w, self.b])
        self.opt.update([self.w, self.b], [dw, db], self.batch_size)

    def predict(self, X):
        self.validate_predict(X)
        return tf.matmul(X, self.w) + self.b
