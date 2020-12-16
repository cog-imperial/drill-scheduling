#!/usr/bin/env
"""tools.py: models for drilling tools."""


class Bit():
    """ Stores bit parameters. """
    def __init__(self, a, rho):
        self.a = a
        self.rho = rho


class PDM():
    """
    Models a PDM including degradation characteristics.

    Args:
        rpm: object which predicts PDM RPM: rpm(diff. press.)
        torque: object which predicts PDM torque: torque(diff. press.)
        failure: object which predicts PDM lifetime: failure(diff. press.)
        r0: initial degradation (between 0 and 1)
        c_var: variance coef. (PDM deg. variance sigma^2 = (dt*dp*c_var)^2)
    """
    def __init__(self, rpm, torque, failure, r0, c_var):
        self.rpm = rpm
        self.torque = torque
        self.failure = failure
        self.r0 = r0
        self.c_var = c_var

    def degradation(self, dt, dp):
        """ Predict degradation rate mean. """
        # return dt / self.failure(dp)
        return dt * self.failure(dp)

    def variance(self, dt, dp):
        """ Predict degradation rate variance. """
        return (dt * dp * self.c_var + 0.00001)**2

    def sig(self, dt, dp):
        """ Predict degradation rate standard deviation. """
        return (dt * dp * self.c_var + 0.00001)

    def k(self, dt, dp):
        # return (1 + 1/(self.failure(dp) * dp * self.c_var + 0.00001))
        return (1 + self.failure(dp) / (dp * self.c_var + 0.00001))


class DrillString():
    """ Model drill string (PDM + bit currently). """
    def __init__(self, pdm, bit):
        self.pdm = pdm
        self.bit = bit
