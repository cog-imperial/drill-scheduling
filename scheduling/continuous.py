#!/usr/bin/env
import bisect
import curves
import utils
import pyomo.environ as p
import pandas as pd
import scipy as sp
import numpy as np
from ropo import ROPO
from rogp.util.numpy import _to_np_obj_array, _pyomo_to_np


eps = 0.0001


class Model():
    def __init__(self):
        pass

    def solve(self, solver='Ipopt', options={}):
        return utils.solve(self, solver, options)

    def multisolve(self, solver='Ipopt', options={}, N=30, Nstop=5):
        return utils.multisolve(self, solver, options, N, Nstop)

    def initialize(self):
        raise NotImplementedError


class Deterministic(Model):
    def __init__(self, geo, drillstring, cost_maint, xfin, xstart=0, Xm=[],
                 Rmax=1, piecewise=False):
        self.geology = geo
        self.drillstring = drillstring
        self.cost_maint = cost_maint  # TODO: should be part of DrillString?
        self.xfin = xfin
        self.build(xstart=xstart, Xm=Xm, Rmax=Rmax, piecewise=piecewise)

    def build(self, xstart=0, Xm=[], Rmax=1, piecewise=False):
        self.xstart = xstart
        self.X = self.get_segments(xstart, self.xfin)
        self.Xm = Xm.copy()
        self.m = p.ConcreteModel()
        self.m.cons = p.ConstraintList()
        self.add_vars(Rmax=Rmax)
        self.add_rop(piecewise)
        self.add_deg()
        self.add_obj()
        self.initialize(self.Xm)

    def add_vars(self, Rmax=1):
        self.m.xm = p.Var(self.Xm, within=p.NonNegativeReals,
                          bounds=(self.xstart, self.xfin))
        self.Xvar = self.add_maintenance_segments(self.Xm)
        self.m.R = p.Var(self.X, within=p.NonNegativeReals, bounds=(0, Rmax))
        self.m.r = p.Var(self.X, within=p.NonNegativeReals, bounds=(0, 1))

    def add_rop(self, piecewise):
        """ Add ROPO model for each segment. """
        # Add one ROPO block per segment
        self.m.rop = p.Block(self.X)
        self.blocks = [ROPO(self.geology(x),
                       self.drillstring,
                       self.m.rop[x],
                       piecewise=piecewise) for x in self.X[0:-1]]

    def get_segments(self, xstart, xfin):
        """ Return list of geological segments between xstart and xfin."""
        # Make sure start and end depth are reasonable
        assert xstart < xfin
        # Start with all geology transitions between xstart and xfin
        X = self.geology.transitions.copy()
        X = [x for x in X if x > xstart and x < xfin]
        # Make sure they are sorted
        X.sort()
        # Insert xstart and xfin into the list
        X.insert(0, xstart)
        X.append(xfin)

        return X

    def add_maintenance_segments(self, Xm):
        Xvar = self.X.copy()
        # Make sure maintenance falls into valid segment
        # Add maintenance depths to transitions
        for xm in reversed(Xm):
            assert xm < self.xfin and xm > self.xstart
            i = bisect.bisect(self.X, xm)
            Xvar.insert(i, self.m.xm[xm])
            self.X.insert(i, xm)
        # Set bounds for maintenance depths
        for i, x in enumerate(Xvar):
            if not isinstance(x, (float, int)):
                if isinstance(Xvar[i-1], (float, int)):
                    x.setlb(Xvar[i-1])
                else:
                    self.m.cons.add(Xvar[i-1] <= x)
                if isinstance(Xvar[i+1], (float, int)):
                    x.setub(Xvar[i+1])
                else:
                    self.m.cons.add(x <= Xvar[i+1])
        return Xvar

    def add_deg(self):
        """ Add degradation constraints. """
        R = 0
        X = self.X
        Xvar = self.Xvar
        delta = {X[i]: Xvar[i+1] - Xvar[i] for i in range(len(X) - 1)}
        fc = self.drillstring.pdm.failure
        for i, x in enumerate(X[0:-1]):
            # Reset r to zero after maintenance
            if not isinstance(Xvar[i], (float, int)):
                R = 0
            # Add degradation in current segment
            dt = delta[x] / (self.m.rop[x].V + eps)
            dp = self.m.rop[x].deltap
            # If GP is warped also pass var and cons list for implicit def
            if isinstance(fc, curves.WarpedGP):
                r = fc(dp, self.m.r[x], self.m.cons)
            # Explicit definition
            else:
                r = fc(dp)
                self.m.cons.add(self.m.r[x] == r)
            R += dt*self.m.r[x]
            self.m.cons.add(self.m.R[x] == R)

    def add_obj(self):
        """ Add objective to model. """
        # Cost of/time spent drilling
        self.m.cost_drill = p.Var()
        X = self.X
        Xvar = self.Xvar
        delta = {X[i]: Xvar[i+1] - Xvar[i] for i in range(len(X) - 1)}
        self.delta = delta
        cost_drilling = sum([delta[x]/(self.m.rop[x].V + eps)
                             for x in self.X[0:-1]])
        self.m.cons.add(self.m.cost_drill == cost_drilling)
        # Cost of/time spent on maintenance
        self.m.cost_maint = p.Var()
        cost_maint = sum([self.cost_maint(self.m.xm[xm]) for xm in self.Xm])
        self.m.cons.add(self.m.cost_maint == cost_maint)
        # Total cost/time to completion
        cost = cost_drilling + cost_maint
        # Add objective to model
        self.m.Obj = p.Objective(expr=cost, sense=p.minimize)

    def initialize(self, X=[], random=False):
        val = self.X[0]
        for i, x in enumerate(self.Xm):
            if random:
                bounds = list(self.m.xm[x].bounds)
                bounds[0] = max(val, bounds[0])
                val = np.random.uniform(low=bounds[0], high=bounds[1])
            else:
                val = X[i]
            self.m.xm[x].value = val

    # move this to utils
    def print_schedule(self):
        """ Print schedule. """
        for i, x in enumerate(self.X[0:-1]):
            xvar = x
            if not isinstance(self.Xvar[i], (float, int)):
                xvar = self.Xvar[i].value
            print('x: {0}, ROP: {1}, R: {2},'.format(xvar, self.m.rop[x].V(),
                                                     self.m.R[x]()),
                  'dp: {0}, t: {1}, w: {2},'.format(self.m.rop[x].deltap(),
                                                    self.m.rop[x].t(),
                                                    self.m.rop[x].w()))

    def get_schedule(self):
        """ Return schedule as pandas DataFrame. """
        res = []
        for i, x in enumerate(self.X[0:-1]):
            xvar = x
            if not isinstance(self.Xvar[i], (float, int)):
                xvar = self.Xvar[i].value
            res.append({'x': x, 'xvar': xvar,
                        'ROP': self.m.rop[x].V(),
                        'R': self.m.R[x](),
                        'r': self.m.r[x](),
                        # 'u': self.m.u[x](),
                        'dp': self.m.rop[x].deltap(),
                        't': self.m.rop[x].t(),
                        'w': self.m.rop[x].w()})
        return pd.DataFrame(res)

    def calc_avg_V(self):
        rhs = 0
        for i, x in enumerate(self.X[:-1]):
            dx = self.Xvar[i+1] - self.Xvar[i]
            V = self.m.rop[x].V()
            rhs += dx/V
        Vavg = (self.X[-1] - self.X[0])/rhs
        if not isinstance(Vavg, (float, int)):
            Vavg = Vavg()
        return Vavg


class Wolfe(Deterministic):
    def __init__(self, geo, drillstring, cost_maint, xfin, xstart=0, Xm=[],
                 Rmax=1, alpha=0.5, piecewise=False):
        self.alpha = alpha
        super().__init__(geo, drillstring, cost_maint, xfin, xstart=xstart,
                         Xm=Xm, Rmax=Rmax, piecewise=piecewise)

    def add_vars(self, Rmax=1):
        super().add_vars(Rmax=Rmax)
        self.m.u = p.Var(self.X, bounds=(-1000, 0))  # Bounds?

    def add_deg(self):
        k = 0
        for i, x in enumerate(self.Xvar):
            if not isinstance(x, (float, int)):
                self._add_deg_block(k, i)
                k = i
        self._add_deg_block(k, len(self.X) - 1)

    def _add_deg_block(self, k, i):
        # Shorthands
        fc = self.drillstring.pdm.failure
        m = self.m
        # Initialize parameters
        alpha = 1 - (1 - self.alpha)/(len(self.Xm) + 1)
        F = sp.stats.norm.ppf(alpha)
        self.F = F
        X = self.X[k:i]
        Xvar = self.Xvar
        delta = {self.X[j]: Xvar[j+1] - Xvar[j] for j in range(k, i)}
        dp = [[m.rop[x].deltap] for x in X]
        dp = _to_np_obj_array(dp)
        dt = [[delta[x]/(m.rop[x].V + eps)] for x in X]
        dt = _to_np_obj_array(dt)
        r = _pyomo_to_np(m.r, ind=X)

        # Calculate matrices
        Sig = fc.rogp.predict_cov_latent(dp)
        dHinv = 1/fc.rogp.warp_deriv(r)
        dHinv = np.diag(dHinv[:, 0])
        hz = fc.rogp.warp(r)
        mu = fc.rogp.predict_mu_latent(dp)
        u = _pyomo_to_np(m.u, ind=X[-1:])

        LHS = np.matmul(Sig, dHinv)
        LHS = np.matmul(LHS, dt)
        RHS = LHS
        LHS = LHS + 2*u*(hz - mu)
        # Add stationarity condition
        for lhs in np.nditer(LHS, ['refs_ok']):
            m.cons.add(lhs.item() == 0)
        RHS = np.matmul(dHinv, RHS)
        rhs = np.matmul(dt.T, RHS)[0, 0]
        lhs = 4*u[0, 0]**2*F
        # Dual variable constraint
        m.cons.add(lhs == rhs)
        # Primal constraint
        m.cons.add(np.matmul(dt.T, r).item() == m.R[X[-1]])
