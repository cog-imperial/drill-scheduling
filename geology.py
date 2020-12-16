#!/usr/bin/env
"""geology.py: tools for modeling geology."""
import bisect


class Rock():
    """ Stores rock parameters for Detournay bit-rock interaction model. """
    def __init__(self, Sstar, wstar, xieps, mugam, betaw, xi):
        self.Sstar = Sstar
        self.wstar = wstar
        self.xieps = xieps
        self.mugam = mugam
        self.betaw = betaw
        self.xi = xi


class Geology():
    """ Stores geology as a list of rock types. """
    def __init__(self, data):
        self.data = data
        self.transitions = list(data.keys())
        self.transitions.sort()

    def lookup(self, x):
        """ Look up rock parameters at depth/length x. """
        key = max(key for key in self.data.keys() if x - key >= 0)
        return self.data[key]

    def __call__(self, x):
        return self.lookup(x)

    def segment(self, x):
        return bisect.bisect_left(self.transitions, x) - 1

    def midpoint(self, i, xfin):
        trans = self.transitions + [xfin]
        return (trans[i+1] + trans[i])/2

    def start(self, i):
        return self.transitions[i]
