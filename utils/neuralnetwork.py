import tensorflow as tf
import numpy as np
import tensorflow_graphics.math.interpolation as inter

from custom_lbfgs import lbfgs, Struct

class BSplineLayer(tf.keras.layers.Layer):
    def __init__(self, layer_sizes, num_elem, degree):
        super(BSplineLayer, self).__init__()
        self.layer_sizes = layer_sizes
        self.num_elem = num_elem
        self.degree = degree

    def get_spline(self, num_elem, x):
        pos_x = x * num_elem
        pos_spl = tf.cast(tf.math.floor(pos_x), dtype=tf.int32) + tf.constant(1) # let's get which element we're in...
        pos_spl = tf.cond(pos_spl > num_elem, lambda: num_elem, lambda: pos_spl) # ...and not to get out of range (could be done better to avoid cond)
        splines = inter.bspline._quadratic(pos_x%1.0) # comptue spline values (this impl support only [0,1] range)
        return splines, pos_spl

    def eval_point_2d(self, u, x, y, num_elem, degree): # compute `u` from 2D splines
        splines_x, pos_x = self.get_spline(num_elem, x)
        splines_y, pos_y = self.get_spline(num_elem, y)
        X, Y = tf.meshgrid(splines_y, splines_x)
        W = tf.slice(u, [pos_x-1, pos_y-1], [degree,degree]) # getting only those splines that are non-zero in this element
        return tf.math.reduce_sum(W * X * Y) # Equivalent to: Sum_i_j { u_i_j * spline_x_i * spline_y_j }

    def build(self, input_shape):
        self.layers = [tf.keras.layers.Lambda(lambda X: X)] # Redundant I guess
        for size in self.layer_sizes[1:-1]:
            self.layers.append(tf.keras.layers.Dense(size, activation=tf.nn.tanh, kernel_initializer="glorot_normal"))
        self.layers.append(tf.keras.layers.Dense(1, activation=tf.nn.tanh, kernel_initializer="glorot_normal"))
        # BSpline indices can be done either by simple (i,j) encoding or one_hot_encoding
        # Current implementation is one_hot_encoding, line below is for the reference of (i,j) encoding
        # self.cofs = [[tf.math.floor(i / 7.0), i % 7.0] for i in range(49)]
        input_shape = self.layer_sizes[0]
        self.matrix_size = tf.math.sqrt(float(input_shape))
        one_hot_encoded = []
        for i in range(input_shape):
            row = np.zeros((input_shape,))
            row[i] = 1
            one_hot_encoded.append(row)
        self.cofs = tf.convert_to_tensor(one_hot_encoded, dtype=tf.float32) # cofs ->> coefficients' indices

    def call(self, input):
        output = self.cofs # first we need to ask our NN for coefficients for each 2D BSpline...
        for layer in self.layers:
            output = layer(output)
        u = tf.reshape(output, [self.matrix_size,self.matrix_size]) * 3.0 # ...and reshape it. IMPORTANT! I think we need to multiply it to scale bc otherwise all coefficients will be almost 0 and with BSplines values result ends up as 0
        return tf.map_fn(lambda xy : self.eval_point_2d(u, (xy[0] + 1.0) / 2.0, xy[1], self.num_elem, self.degree), input) # for each learning example we need to compute `u` from BSpline combination (standard FEM procedure after solving coefficients)

class NeuralNetwork(object):
    def __init__(self, hp, logger, ub, lb):

        layers = hp["layers"]

        # Setting up the optimizers with the hyper-parameters
        self.nt_config = Struct()
        self.nt_config.learningRate = hp["nt_lr"]
        self.nt_config.maxIter = hp["nt_epochs"]
        self.nt_config.nCorrection = hp["nt_ncorr"]
        self.nt_config.tolFun = 1.0 * np.finfo(float).eps
        self.tf_epochs = hp["tf_epochs"]
        self.tf_optimizer = tf.keras.optimizers.Adam(
            learning_rate=hp["tf_lr"],
            beta_1=hp["tf_b1"],
            epsilon=hp["tf_eps"])
        self.dtype = "float32"

        # Computing the sizes of weights/biases for future decomposition
        self.sizes_w = []
        self.sizes_b = []
        for i, width in enumerate(layers):
            if i != 1:
                self.sizes_w.append(int(width * layers[1]))
                self.sizes_b.append(int(width if i != 0 else layers[1]))

        self.logger = logger

        bspline_model = tf.keras.Sequential()
        bspline_model.add(tf.keras.layers.InputLayer(input_shape=(2,)))
        self.bspline_layer = BSplineLayer(layers, num_elem=5, degree=3) # this layer contains other layers. IMPORTANT! for current vesion `degree` can't be changed witout changing base functions choice. degree=3 is `_quadratic` in `get_spline`
        bspline_model.add(self.bspline_layer)
        self.bspline_model = bspline_model

    # Defining custom loss
    @tf.function
    def loss(self, u, u_pred):
        return tf.reduce_mean(tf.square(u - u_pred))

    @tf.function
    def grad(self, X, u):
        with tf.GradientTape() as tape:
            loss_value = self.loss(u, self.bspline_model(X))
        grads = tape.gradient(loss_value, self.wrap_training_variables())
        return loss_value, grads

    def wrap_training_variables(self):
        var = self.bspline_model.trainable_variables
        return var

    def get_params(self, numpy=False):
        return []

    def get_weights(self, convert_to_tensor=True):
        w = []
        for layer in self.bspline_layer.layers[1:]:
            weights_biases = layer.get_weights()
            weights = weights_biases[0].flatten()
            biases = weights_biases[1]
            w.extend(weights)
            w.extend(biases)
        if convert_to_tensor:
            w = self.tensor(w)
        return w

    def set_weights(self, w):
        for i, layer in enumerate(self.bspline_layer.layers[1:]):
            start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
            end_weights = sum(self.sizes_w[:i+1]) + sum(self.sizes_b[:i])
            weights = w[start_weights:end_weights]
            w_div = int(self.sizes_w[i] / self.sizes_b[i])
            weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
            biases = w[end_weights:end_weights + self.sizes_b[i]]
            weights_biases = [weights, biases]
            layer.set_weights(weights_biases)

    def get_loss_and_flat_grad(self, X, u):
        def loss_and_flat_grad(w):
            with tf.GradientTape() as tape:
                self.set_weights(w)
                loss_value = self.loss(u, self.bspline_model(X))
            grad = tape.gradient(loss_value, self.wrap_training_variables())
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat = tf.concat(grad_flat, 0)
            return loss_value, grad_flat

        return loss_and_flat_grad

    def tf_optimization(self, X_u, u):
        self.logger.log_train_opt("Adam")
        for epoch in range(self.tf_epochs):
            loss_value = self.tf_optimization_step(X_u, u)
            self.logger.log_train_epoch(epoch, loss_value)

    @tf.function
    def tf_optimization_step(self, X_u, u):
        loss_value, grads = self.grad(X_u, u)
        self.tf_optimizer.apply_gradients(
                zip(grads, self.wrap_training_variables()))
        return loss_value

    def nt_optimization(self, X_u, u):
        self.logger.log_train_opt("LBFGS")
        loss_and_flat_grad = self.get_loss_and_flat_grad(X_u, u)
        self.nt_optimization_steps(loss_and_flat_grad)

    def nt_optimization_steps(self, loss_and_flat_grad):
        lbfgs(loss_and_flat_grad,
              self.get_weights(),
              self.nt_config, Struct(), True,
              lambda epoch, loss, is_iter:
              self.logger.log_train_epoch(epoch, loss, "", is_iter))

    def fit(self, X_u, u):
        self.logger.log_train_start(self)

        # Creating the tensors
        X_u = self.tensor(X_u)
        u = self.tensor(u)

        # Optimizing
        self.tf_optimization(X_u, u)
        self.nt_optimization(X_u, u)

        self.logger.log_train_end(self.tf_epochs + self.nt_config.maxIter)

    def predict(self, X_star):
        u_pred = self.bspline_model(X_star)
        return u_pred.numpy()

    def summary(self):
        return self.bspline_model.summary()

    def tensor(self, X):
        return tf.convert_to_tensor(X, dtype=self.dtype)

