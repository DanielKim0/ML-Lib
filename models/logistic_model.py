from .tf_model import TFModel
import tensorflow as tf

def logreg(X, w, b):
    return softmax(tf.matmul(X, w) + b)

def softmax(o):
    o_exp = tf.math.exp(o)
    sums = tf.math.reduce_sum(o_exp, 1, keepdims=True)
    return o_exp / sums

def sgd(params, grads, lr, batch_size):
    for param, grad in zip(params, grads):
        param.assign_sub(lr * grad / batch_size)
    
def accuracy(true, pred):
    maxes = tf.math.argmax(pred, axis=1)
    cmp = (tf.cast(maxes, true.dtype) == true)
    return float(tf.reduce_sum(tf.cast(cmp, true.dtype))) / len(pred)

class LogisticModel(TFModel):
    def __init__(self):
        super().__init__()

    def save(self):
        super().save()

    def load(self):
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def validate_fit(self):
        pass

    def validate_predict(self):
        pass

    def build_model(self, w_size, w_classes, w_mean, w_stddev):
        self.w = tf.Variable(tf.random.normal(shape=(w_size, w_classes), mean=w_mean, stddev=w_stddev), trainable=True)
        self.b = tf.Variable(tf.zeros(w_classes), trainable=True)
        self.model = logreg
        self.update = sgd

    def fit(self, X, y, classes, loss, lr=0.03, batch_size=16, num_epochs=32, mean=0, stddev=0.01):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.loss = loss.compare
        self.lr = lr
        self.X = X
        self.y = y
        self.build_model(X.shape[1], classes, mean, stddev)
        # self.validate_fit(self.X, self.y)
        super().fit()

    def train_epoch(self):
        for X, y in self.data_iter(self.X, self.y):
            self.train_step(X, y)
        train_l = self.loss(self.y, self.model(self.X, self.w, self.b))
        print(f"epoch {self.curr_epoch}, loss {float(tf.reduce_mean(train_l)):f}, accuracy {float(accuracy(self.y, self.model(self.X, self.w, self.b)))}")

    def train_step(self, X, y):
        with tf.GradientTape() as g:
            l = self.loss(y, self.model(X, self.w, self.b))
        dw, db = g.gradient(l, [self.w, self.b])
        self.update([self.w, self.b], [dw, db], self.lr, self.batch_size)

    def predict(self):
        pass
