import tensorflow as tf
from comp.models import linreg
from comp.functions import split_func
from metrics.class_accuracy import ClassAccuracy
from utils.compress import *
from .tf_model import TFModel


class SequentialModel(TFModel):
    def __init__(self, layers=None):
        super().__init__()
        self.layers = layers

    def __str__(self):
        s = "Sequential Model\n"
        if self.model_fit:
            s += "Currently fit\n"
            s += f"epochs: {self.num_epochs}/{self.curr_epoch}\n"
            s += f"loss: {self.loss}\n"
            s += f"opt: {self.opt}\n"
            s += f"batch_size: {self.batch_size}\n"
        else:
            s += "Currently not fit\n"
        for i in range(len(self.layers)):
            s += f"\nLayer {i}: {self.layers[i].__str__()}"

        return s

    def __repr__(self):
        if self.model_fit:
            s = f"MLPModel(model_fit={True}, loss={self.loss}, opt={self.opt}, batch_size={self.batch_size}, num_epochs={self.num_epochs}, curr_epoch={self.curr_epoch}, mean={self.mean}, stddev={self.stddev})"
            for i in range(len(self.layers)):
                s += f"\nLayer {i}: {self.layers[i].__repr__()}"
        else:
            s = f"MLPModel(model_fit={False})"
        for i in range(len(self.layers)):
            s += f"\nLayer {i}: {self.layers[i].__repr__()}"
        return s

    def save(self, path):
        super().save()
        data = {
            "loss": self.loss,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "curr_epoch": self.curr_epoch,
            "opt": self.opt,
        }
        compress_files(path, data, None, self.layers)

    def load(self, path):
        data, arrays = uncompress_files(path)

        self.layers = data["layers"]
        self.loss = data["loss"]
        self.opt = data["opt"]
        self.num_epochs = data["num_epochs"]
        self.curr_epoch = data["curr_epoch"]
        self.batch_size = data["batch_size"]    
        self.model = self.create_net()    

    def validate_model(self):
        # model structure validation goes here
        pass

    def validate_fit(self):
        pass

    def validate_predict(self):
        pass

    def create_net(self):
        def net(X, model):
            for layer in model:
                X = layer.op(X)
            return X
        return net

    def build_model(self, inp):
        # change to 2D+
        prev = []
        curr = inp
        for layer in self.layers:
            prev = curr
            layer.set_dims(prev)
            curr = layer.out
            if layer.weighted:
                layer.init_weights()
        self.model = self.create_net()

    def fit(self, X, y, loss, opt, batch_size=16, num_epochs=32):
        # data casts
        X = tf.cast(tf.constant(X), tf.float32)

        # setting instance variables
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.opt = opt
        self.loss = loss

        # build, validate, fit
        self.build_model(X.shape[1:])
        # self.validate_fit(X, y, act)
        super().fit(X, y)

    def train_epoch(self, X, y):
        for X_batch, y_batch in self.data_iter(X, y):
            self.train_step(X_batch, y_batch)
        train_l = self.loss.compare(y, self.model(X, self.layers))
        print(f"epoch {self.curr_epoch}, loss {float(tf.reduce_mean(train_l)):f}")

    def gather_weights(self):
        weights = []
        for layer in self.layers:
            if layer.weighted:
                weights.extend(layer.weights())
        return weights

    def gather_loss(self):
        loss = 0
        for layer in self.layers:
            if layer.weighted:
                loss += layer.loss()
        return loss

    def train_step(self, X, y):
        with tf.GradientTape() as g:
            l = self.loss.compare(y, self.model(X, self.layers))
            l += self.gather_loss()
        grads = g.gradient(l, self.gather_weights())
        self.opt.update_model(self.layers, grads, self.batch_size)

    def predict(self, X):
        return self.model(X, self.layers)
