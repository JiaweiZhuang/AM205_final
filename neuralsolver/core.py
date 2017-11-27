"""
Reference:
Lagaris, Isaac E., Aristidis Likas, and Dimitrios I. Fotiadis.
"Artificial neural networks for solving ordinary and partial
differential equations." IEEE Transactions on Neural Networks
 9.5 (1998): 987-1000.
"""


# autograd core
import autograd.numpy as np
from autograd import grad
from autograd import elementwise_grad as egrad

# autograd utils
import autograd.numpy.random as npr
from autograd.misc.flatten import flatten

# scipy optimizer
from scipy.optimize import minimize

# A global counter for printing scipy training progress
# https://stackoverflow.com/questions/16739065/how-to-display-progress-of-scipy-optimize-function
# TODO: Can we improve this dirty hack?
count = 0


def init_weights(n_in=1, n_hidden=10, n_out=1):
    '''
    Initialize NN weights
    '''
    # TODO: more layers?
    W1 = npr.randn(n_in, n_hidden)
    b1 = np.zeros(n_hidden)
    W2 = npr.randn(n_hidden, n_out)
    b2 = np.zeros(n_out)
    params = [W1, b1, W2, b2]
    return params


def predict(params, t, y0, act=np.tanh):
    '''
    Make NN prediction
    '''
    W1, b1, W2, b2 = params

    a = act(np.dot(t, W1) + b1)
    out = np.dot(a, W2) + b2  # standard NN output

    y = y0 + t*out  # apply Lagaris et al. (1998)

    return y


predict_dt = egrad(predict, argnum=1)  # element-wise grad w.r.t t


class NNSolver(object):
    def __init__(self, f, t, y0_list, n_hidden=10):
        '''
        Neural Network Solver Class

        Parameters
        ----------
        f : callable
            Right-hand side of the ODE system dy/dt = f(t, y).
            Similar to the input for scipy.integrate.solve_ivp()

            Important notes:
            - Must use autograd's numpy inside f (import autograd.numpy as np)
            - For a single ODE, should return a list of one element.

        t : column vector, i.e. numpy array of shape (n, 1)
            Training points

        y0_list : a list of floating point numbers
            Initial condition.
            For a single ODE, should be a list of one element.

        n_hidden : integer, optional
            Number of hidden units of the NN
        '''

        Nvar = len(y0_list)
        assert len(f(t[0], y0_list)) == Nvar, \
            'f and y0_list should have same size'

        assert t.shape == (t.size, 1), 't must be a column vector'

        self.Nvar = Nvar
        self.f = f
        self.t = t
        self.y0_list = y0_list
        self.n_hidden = n_hidden

        self.reset_weights()

    def __str__(self):
        return ('Neural ODE Solver \n'
                'Number of equations:       {} \n'
                'Initial condition y0:      {} \n'
                'Numnber of hidden units:   {} \n'
                'Number of training points: {} '
                .format(self.Nvar, self.y0_list, self.n_hidden, self.t.size)
                )

    def __repr__(self):
        return self.__str__()

    def reset_weights(self):
        '''reinitialize NN weights (randomly)'''
        self.params_list = [init_weights(n_hidden=self.n_hidden)
                            for _ in range(self.Nvar)]

        flattened_params, unflat_func = flatten(self.params_list)
        self.flattened_params = flattened_params
        self.unflat_func = unflat_func

    def loss_func(self, params_list):
        '''Compute loss function'''
        # params_list should be an explicit input, not from self

        # some shortcut
        y0_list = self.y0_list
        t = self.t
        f = self.f

        y_pred_list = []
        dydt_pred_list = []
        for params, y0 in zip(params_list, y0_list):
            y_pred = predict(params, t, y0)
            dydt_pred = predict_dt(params, t, y0)

            y_pred_list.append(y_pred)
            dydt_pred_list.append(dydt_pred)

        f_pred_list = f(t, y_pred_list)

        loss_total = 0.0
        for f_pred, dydt_pred in zip(f_pred_list, dydt_pred_list):
            loss = np.sum((dydt_pred-f_pred)**2)
            loss_total += loss

        return loss_total

    def loss_wrap(self, flattened_params):
        '''Loss function that takes flattened parameters, for scipy optimizer
        '''
        # flattened_params should be an explicit input, not from self
        params_list = self.unflat_func(flattened_params)

        return self.loss_func(params_list)

    def train(self, method='BFGS', maxiter=2000, iprint=200):
        '''
        Train the neural net

        Parameters
        ----------
        method : string, optional
            Optimization method for scipy.optimize.minimize()
            'BFGS' should be the most robust one

        maxiter : integer, optional
            Maximum number of iterations

        maxiter : integer, optional
            Print loss per iprint step
        '''

        global count
        count = 0  # reset counter for next training

        def print_loss(x):
            global count
            if count % iprint == 0:
                print("iteration:", count, "loss: ", self.loss_wrap(x))
            count += 1

        opt = minimize(self.loss_wrap, x0=self.flattened_params,
                       jac=grad(self.loss_wrap), method=method,
                       callback=print_loss,
                       options={'disp': True, 'maxiter': maxiter})

        # update parameters
        self.flattened_params = opt.x
        self.params_list = self.unflat_func(opt.x)

    def predict(self, t=None):
        '''
        Make new predicts

        Parameters
        ----------
        t : 1D numpy array, optional
            use training points by default

        '''
        if t is None:
            t = self.t

        y_pred_list = []
        dydt_pred_list = []
        for params, y0 in zip(self.params_list, self.y0_list):
            y_pred = predict(params, t, y0)
            dydt_pred = predict_dt(params, t, y0)

            y_pred_list.append(y_pred.squeeze())
            dydt_pred_list.append(dydt_pred.squeeze())

        return y_pred_list, dydt_pred_list
