#!/usr/bin/env
import math
import bisect
import curves
import pyomo.environ as p
import pandas as pd
import scipy as sp
from ropo import ROPO


class Deterministic():
    """
    Drill-scheduling model.

    Args:
        geology: Geology object
        drillstring: DrillString object
        cost_maint: Curve predicting cost of maintenance
        xfin: final depth/length
        deltax (200): segment length
        xstart (0): initial depth/length

    """
    def __init__(self, geo, drillstring, cost_maint,
                 xfin, deltax=200, xstart=0, method='det', alpha=0.5,
                 mip=True, penalty=False):
        self.geology = geo
        self.drillstring = drillstring
        self.cost_maint = cost_maint  # TODO: should be part of DrillString?
        self.xstart = xstart
        self.xfin = xfin
        self.method = method
        self.alpha = alpha
        self.mip = mip
        self.penalty = penalty
        self.eps = 0.0001
        self.get_segments(xstart, xfin, deltax)
        self.m = p.ConcreteModel()
        self.m.cons = p.ConstraintList()
        self.add_vars(mip)
        self.add_rop(mip)
        self.add_deg()
        self.add_obj()

    def get_segments(self, xstart, xfin, deltax):
        """ Split length into segments under consideration of geology. """
        n_seg = math.ceil((xfin - xstart)/deltax)
        # Split into segments of length deltax
        X = [xstart + i*deltax for i in range(0, n_seg)] + [xfin]
        # Split where rock types change
        for x in self.geology.transitions:
            if x not in X:
                bisect.insort(X, x)
        self.X = set(X[:-1])
        self.Xlist = X[:-1]
        self.N = len(self.X)
        self.delta = {X[i]: X[i+1] - X[i] for i in range(self.N)}

    def add_vars(self, mip):
        self.m.y = p.Var(self.X, within=p.NonNegativeReals, bounds=(0, 1))
        self.m.R = p.Var(self.X, within=p.NonNegativeReals, bounds=(0, 1))
        self.m.r = p.Var(self.X, within=p.NonNegativeReals, bounds=(0, 1))
        if mip:
            self.m.z = p.Var(self.X, within=p.Binary)

    def add_rop(self, mip):
        """ Add ROPO model for each segment. """
        self.m.rop = p.Block(self.X)
        self.blocks = [ROPO(self.geology(x),
                       self.drillstring,
                       self.m.rop[x], piecewise=mip) for x in self.X]

    def calc_dt(self, x):
        self.delta[x]/(self.m.rop[x].V + self.eps)

    def add_deg(self):
        """ Add degradation constraints. """
        R = 0
        fc = self.drillstring.pdm.failure
        for x in self.Xlist:
            dt = self.delta[x] / (self.m.rop[x].V + 0.001)
            dp = self.m.rop[x].deltap
            if isinstance(fc, curves.WarpedGP):
                r = fc(dp, self.m.r[x], self.m.cons)
            # Explicit definition
            else:
                r = fc(dp)
                self.m.cons.add(self.m.r[x] == r)
            R += dt*self.m.r[x]
            R -= self.m.y[x]
            self.m.cons.add(self.m.R[x] == R)
            # z_x >= y_x
            if self.mip:
                self.m.cons.add(self.m.z[x] >= self.m.y[x])

    def add_obj(self):
        """ Add objective to model. """
        # Cost of/time spent drilling
        self.m.cost_drill = p.Var()
        cost_drilling = sum([self.delta[x]/(self.m.rop[x].V + 0.001)
                             for x in self.X])
        self.m.cons.add(self.m.cost_drill == cost_drilling)
        # Cost of/time spent on maintenance
        self.m.cost_maint = p.Var()
        if self.mip:
            z = self.m.z
        else:
            z = self.m.y
        cost_maint = sum([z[x]*self.cost_maint(x + self.delta[x])
                          for x in self.X])
        self.m.cons.add(self.m.cost_maint == cost_maint)
        # Total cost/time to completion
        cost = cost_drilling + cost_maint
        # Add penalty
        if self.penalty:
            cost += sum([self.m.y[x]**2 for x in self.X])
        self.m.Obj = p.Objective(expr=cost, sense=p.minimize)

    def add_cons(self):
        pass

    def solve(self, solver='Bonmin', options={}):
        """"
        Solve model.

        Args:
            solver: gams solver to use
            options: dict of solver/gams options

        """
        if solver == 'Ipopt':
            self.solver = p.SolverFactory('ipopt', solver_io='nl')
            results = self.solver.solve(self.m,
                                        tee=True,
                                        logfile='drilling.log',
                                        symbolic_solver_labels=True)
        else:
            self.solver = p.SolverFactory('gams', solver_io='shell')
            opt = {'optcr': 0.001, 'resLim': 60}
            opt.update(options)
            opt_list = ['option {0}={1};'.format(key, val)
                        for key, val in opt.items()]
            results = self.solver.solve(self.m,
                                        tee=True,
                                        logfile='drilling.log',
                                        io_options={'solver': solver},
                                        symbolic_solver_labels=True,
                                        add_options=opt_list)
        results.write()
        self.time = results['Solver'][0]['User time']
        self.lb = results['Problem'][0]['Lower bound']

        return results.solver.status, results.solver.termination_condition

    def print_schedule(self):
        """ Print schedule. """
        for x in self.Xlist:
            print('x: {0}, ROP: {1}, R: {2},'.format(x, self.m.rop[x].V(),
                                                     self.m.R[x]()),
                  'dp: {0}, t: {1}, w: {2},'.format(self.m.rop[x].deltap(),
                                                    self.m.rop[x].t(),
                                                    self.m.rop[x].w()),
                  'z: {0}, y: {1}'.format(self.m.z[x](),
                                          self.m.y[x]()))

    def get_schedule(self):
        """ Return schedule as pandas DataFrame. """
        res = []
        for x in self.Xlist:
            res.append({'x': x, 'ROP': self.m.rop[x].V(), 'R': self.m.R[x](),
                        'dp': self.m.rop[x].deltap(), 't': self.m.rop[x].t(),
                        'w': self.m.rop[x].w(), 'z': self.m.z[x](),
                        'y': self.m.y[x]()})
        return pd.DataFrame(res)

    def calc_avg_V(self):
        """ Calculate and return average ROP. """
        return sum([self.m.rop[x].V() for x in self.X])/self.N


class Wolfe(Deterministic):
    def __init__(self, geology, drillstring, cost_maint,
                 xfin, deltax=200, xstart=0, method='det', alpha=0.5,
                 mip=True, penalty=False):
        super().__init__(geology, drillstring, cost_maint,
                         xfin, deltax=deltax, xstart=xstart,
                         method=method, alpha=alpha,
                         mip=mip, penalty=penalty)

    def add_deg(self):
        """ Add degradation constraints. """
        r = 0
        pad = 0
        for x in self.Xlist:
            dt = self.delta[x] / (self.m.rop[x].V + 0.001)
            dp = self.m.rop[x].deltap
            r += self.drillstring.pdm.degradation(dt, dp)
            r -= self.m.y[x]
            F = sp.stats.norm.ppf(self.alpha)
            pad += dt*self.drillstring.pdm.failure.calc_var(dp)*dt
            for xp in [xi for xi in self.Xlist if xi < x]:
                dtp = self.delta[xp] / (self.m.rop[xp].V + 0.001)
                dpp = self.m.rop[xp].deltap
                sig = self.drillstring.pdm.failure.calc_var(dp, dpp)
                pad += 2*dt*sig*dtp
            self.m.cons.add(r + F*p.sqrt(pad + 0.0001) <= 1)
            # self.m.cons.add(r + F*p.sqrt(0.0001) <= 1)
            # 0 <= R_x <= 1 for all x
            self.m.cons.add(self.m.R[x] == r)
            # z_x >= y_x
            if self.mip:
                self.m.cons.add(self.m.z[x] >= self.m.y[x])
        # self.m.cons.add(sum([self.m.z[x] for x in self.X]) <= 2)


class Chance(Deterministic):
    """
    Drill scheduling model with chance constraint for Gaussian uncertainty

    Args:
        geology: Geology object
        drillstring: DrillString object
        cost_maint: Curve predicting cost of maintenance
        xfin: final depth/length
        deltax (200): segment length
        xstart (0): initial depth/length

    """
    def __init__(self, geology, drillstring, cost_maint,
                 xfin, eps, deltax=200, xstart=0):
        self.eps = eps
        self.F = sp.stats.norm.ppf(1 - eps)
        super().__init__(geology, drillstring, cost_maint, xfin,
                         deltax=deltax, xstart=xstart)

    def add_deg(self):
        self.m.n = p.Var(within=p.NonNegativeIntegers)
        # self.m.z = p.Var(self.X, within=p.Binary)
        self.m.z = p.Var(self.X, within=p.NonNegativeReals, bounds=(0, 1))
        self.m.y = p.Var(self.X, within=p.NonNegativeReals, bounds=(0, 1))
        self.m.R = p.Var(self.X, within=p.NonNegativeReals, bounds=(0, 1))
        self.m.sig = p.Var(self.X, within=p.NonNegativeReals)

        r = 0
        var = 0
        for x in self.Xlist:
            dt = self.delta[x]/(self.m.rop[x].V + 0.001)
            dp = self.m.rop[x].deltap
            r += self.drillstring.pdm.degradation(dt, dp)
            var += self.drillstring.pdm.variance(dt, dp)
            self.m.cons.add(self.m.sig[x] == p.sqrt(var))
            r -= self.m.y[x]
            self.m.cons.add(self.m.R[x] == r + p.sqrt(var)*self.F)
            self.m.cons.add(self.m.z[x] >= self.m.y[x])
        self.m.cons.add(sum([self.m.z[x] for x in self.X]) == self.m.n)


class Gamma(Deterministic):
    """
    Drill-scheduling model using Chernoff bounds and Gamma uncertainty.

    NOTE: Very experimental
    """
    def __init__(self, geology, drillstring, cost_maint,
                 xfin, eps, deltax=200, xstart=0):
        self.eps = eps
        super().__init__(geology, drillstring, cost_maint, xfin,
                         deltax=deltax, xstart=xstart)

    def add_deg(self):
        self.m.n = p.Var(within=p.NonNegativeIntegers)
        self.m.z = p.Var(self.X, within=p.Binary)
        self.m.y = p.Var(self.X, within=p.NonNegativeReals, bounds=(0, 1))
        self.m.R = p.Var(self.X, within=p.NonNegativeReals, bounds=(0, 1))
        self.m.s = p.Var(self.X, within=p.NonNegativeReals,
                         bounds=(0.001, 100000))
        self.m.sig = p.Var(self.X, within=p.NonNegativeReals)

        R = 0
        for x in self.Xlist:
            r = -self.m.s[x]
            rhs = 0
            lhs = 1
            for x2 in [xi for xi in self.Xlist if xi <= x]:
                dt = self.delta[x2]/(self.m.rop[x2].V + 0.001)
                dp = self.m.rop[x2].deltap
                mu = self.drillstring.pdm.degradation(dt, dp)
                sig = self.drillstring.pdm.sig(dt, dp)
                k = self.drillstring.pdm.k(dt, dp)
                r -= p.log(1 - sig*self.m.s[x]) * k
                r -= self.m.s[x] * self.m.y[x2]
                self.m.cons.add(self.m.s[x]*sig <= 0.99)
                lhs += self.m.y[x2]
                rhs += k*sig/(1 - sig*self.m.s[x])
            self.m.cons.add(r <= math.log(self.eps))
            # self.m.cons.add(lhs == rhs)
            R += mu - self.m.y[x]
            self.m.cons.add(self.m.R[x] == R)
            self.m.cons.add(self.m.z[x] >= self.m.y[x])
        self.m.cons.add(sum([self.m.z[x] for x in self.X]) == self.m.n)
