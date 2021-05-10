from .base_opt import BaseOpt


class SGD(BaseOpt):
    def __init__(self, lr):
        self.lr = lr

    def __str__(self):
        return f"Stochastic Gradient Descent Optimizer: lr={self.lr}"

    def __repr__(self):
        return f"SGD(lr={self.lr})"

    def update(self, params, grads, batch_size):
        for param, grad in zip(params, grads):
            param.assign_sub(self.lr * grad / batch_size)

    def update_model(self, layers, grads, batch_size):
        i = 0
        for layer in layers:
            if layer.weighted:
                self.update([layer.w, layer.b], grads[i*2:i*2+2], batch_size)
                i += 1
