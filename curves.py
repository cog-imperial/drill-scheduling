#!/usr/bin/env
"""curves.py: models for fitting curves to data."""
import GPy
import rogp
import numbers
import numpy as np


class Curve():
    """
    Generic parent class for fitting curves.

    """

    def __init__(self, data):
        raise NotImplementedError

    def __call__(self, x, *args, **kwargs):
        # if x < self.bounds[0] or x > self.bounds[1]:
        #     print("Warning: extrapolating.")
        return self.calc(x, *args, **kwargs)


class Linear(Curve):
    """
    Fits a line through some data points: y = a*x + b.

    Args:
        data: numpy array of data points

    """
    def __init__(self, data):
        self.coef = np.polyfit(data[:, 0], data[:, 1], 1).tolist()
        self.bounds = (data[:, 0].min(), data[:, 0].max())

    def calc(self, x):
        """ Predict y at x. """
        return self.coef[0]*x + self.coef[1]


class Quadratic(Curve):
    """
    Fits a quadratic through some data points: y = a*x^2 + b*x + c.

    Args:
        data: numpy array of data points

    """
    def __init__(self, data):
        self.coef = np.polyfit(data[:, 0], data[:, 1], 2).tolist()
        self.bounds = (data[:, 0].min(), data[:, 0].max())

    def calc(self, x):
        """ Predict y at x. """
        return self.coef[0]*x**2 + self.coef[1]*x + self.coef[2]


class LogQuadratic(Quadratic):
    """
    Fits a log10-quadratic to some data points: log(y) = a*x^2 + b*x + c.

    Args:
        data: numpy array of data points

    """
    def __init__(self, data):
        self.bounds = (data[:, 0].min(), data[:, 0].max())
        data[:, 1] = np.log10(data[:, 1])
        super().__init__(data)

    def calc(self, x):
        """ Predict y at x. """
        return np.power(10, super().calc(x))


class GP(Curve):
    """
    Fits a (warped) GP to some data points.

    Args:
        data: list of two numpy arrays with data points: [x, y]

    """

    def __init__(self, X, Y, X_norm=None, Y_norm=None, kernel=None):
        self.bounds = (X.min(), X.max())
        if X_norm is None:
            self.X_norm = GPy.util.normalizer.Standardize()
        if Y_norm is None:
            self.Y_norm = GPy.util.normalizer.Standardize()
        if kernel is None:
            self.kernel = GPy.kern.RBF(input_dim=X.shape[1], variance=1.,
                                       lengthscale=1.)

        self.train(X, Y)

    def normalize(self, X, Y):
        norm = rogp.util.Normalizer()
        norm.scale_by(X, Y)
        X, Y = norm.normalize(X, Y)
        return X, Y, norm

    def _train(self, X, Y):
        return GPy.models.GPRegression(X, Y, self.kernel)

    def train(self, X, Y, kernel=None):
        X, Y, norm = self.normalize(X, Y)
        gp = self._train(X, Y)
        gp.optimize(messages=True)
        self.rogp = rogp.from_gpy(gp, norm=norm)

    def calc(self, x):
        x, scalar = self._x_to_array(x)
        y = self.rogp.predict_mu(x)
        if scalar:
            y = y[0, 0]
        return y

    def calc_var(self, x1, x2=None):
        x1, scalar1 = self._x_to_array(x1)
        if x2 is None:
            y = self.rogp.predict_cov(x1)
            if scalar1:
                return y[0, 0]
            return y
        else:
            x2, scalar2 = self._x_to_array(x2)
            assert scalar1 and scalar2
            y = self.rogp.predict_cov(np.concatenate((x1, x2), axis=0))[0, 1]
            return y

    def _x_to_array(self, x):
        if isinstance(x, np.ndarray):
            return x, False
        elif isinstance(x, numbers.Number):
            X = np.array([[x]])
        else:
            X = np.empty((1, 1), dtype=object)
            X[0, 0] = x
        return X, True


class WarpedGP(GP):
    def __init__(self, X, Y, warping_terms=1, **kwargs):
        self.warping_terms = warping_terms
        super().__init__(X, Y, **kwargs)

    def _train(self, X, Y):
        return GPy.models.WarpedGP(X, Y, kernel=self.kernel,
                                   warping_terms=self.warping_terms)

    def calc(self, x, y, cons):
        x, scalar = self._x_to_array(x)
        y, scalar_z = self._x_to_array(y)
        y = self.rogp.predict_mu(x, y, cons=cons)
        if scalar:
            y = y[0, 0]
        return y


class Reciprocal(Curve):
    def __init__(self, curve):
        self.curve = curve
        self.bounds = curve.bounds

    def calc(self, x):
        return 1/self.curve(x)
