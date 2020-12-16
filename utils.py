#!/usr/bin/env
import copy
import time
import numpy as np
import pyomo.environ as p
from pyomo.opt import SolverStatus, TerminationCondition


# TODO: Generalize to higher dimension
def _to_np_obj_array(x):
    """ Convert nested list to numpy array with dtype=object """
    X = np.empty((len(x), len(x[0])), dtype=object)
    X[:] = x
    return X


def _eval(x, eval):
    if eval:
        return x()
    else:
        return x


def _pyomo_to_np(X, ind=None, eval=False):
    if ind is None:
        XX = [[_eval(x, eval)] for _, x in X.items()]
    else:
        XX = [[_eval(X[i], eval)] for i in ind]
    return _to_np_obj_array(XX)


def _eval_p(X, ind=None):
    if ind is None:
        return [x() for _, x in X.items()]
    else:
        return [X[i]() for i in ind]


def solve(m, solver='Ipopt', options={}):
    """"
    Solve model.

    Args:
        solver: gams solver to use
        options: dict of solver/gams options

    """
    if solver == 'Ipopt':
        m.solver = p.SolverFactory('ipopt', solver_io='nl')
        results = m.solver.solve(m.m,
                                 tee=False,
                                 logfile='drilling.log',
                                 symbolic_solver_labels=True)
    else:
        m.solver = p.SolverFactory('gams', solver_io='shell')
        opt = {'optcr': 0.001, 'resLim': 60}
        opt.update(options)
        opt_list = ['option {0}={1};'.format(key, val)
                    for key, val in opt.items()]
        results = m.solver.solve(m.m,
                                 tee=True,
                                 logfile='drilling.log',
                                 io_options={'solver': solver},
                                 symbolic_solver_labels=True,
                                 add_options=opt_list)
    results.write()
    m.time = results['Solver'][0]['User time']
    m.lb = results['Problem'][0]['Lower bound']

    return results.solver.status, results.solver.termination_condition


def multisolve(self, solver='Ipopt', options={}, N=30, Nstop=5):
        self.m_best = None
        self.obj_list = []
        self.opt = 1
        N_errors = 0
        Nopt = 0
        Ninf = 0
        t_start = time.perf_counter()
        for i in range(N):
            self.initialize(random=True)
            try:
                status, condition = self.solve(solver=solver,
                                               options=options)
                obj = self.m.Obj()
                if (status is SolverStatus.ok and condition
                        is TerminationCondition.optimal):
                    if self.m_best is None or obj <= self.m_best.Obj():
                        self.m_best = copy.deepcopy(self.m)
                    self.obj_list.append(obj)
                elif condition is TerminationCondition.infeasible:
                    Ninf += 1
            except:
                N_errors += 1
            if len(self.obj_list) > 0:
                Nopt = sum(np.abs(np.array(self.obj_list)
                                  - min(self.obj_list)) <= 1e-5)
            if Nopt >= Nstop:
                # if len(self.obj_list) >= Nstop:
                print('Found enough optimal solutions.')
                self.opt = 2
                break
            if Ninf >= Nstop:
                print('Problem seems to be infeasible')
                self.opt = -1
                break
        self.t_solve = time.perf_counter() - t_start
        if len(self.obj_list) == 0:
            print('WARNING: no optimal solution found')
            self.opt = 0
        else:
            self.m = self.m_best
        print('Fraction of times Ipopt failed: {}'.format(N_errors/N))


def sample_y_beta(fc, x, N):
    mu = fc(x)
    var = 0.000005*(np.exp(x/250) - 1)
    k = 7.5
    theta = np.sqrt(var/k)
    mode = (k - 1)*theta
    y = mu - mode + np.random.gamma(k, scale=theta, size=(x.shape[0], N))
    return y


def sample_y_normal(fc, x, N):
    mu = fc(x)
    var = 0.000005*(np.exp(x/250) - 1)
    y = np.random.normal(mu, np.sqrt(var), size=(x.shape[0], N))
    return y


def sample_y(fc, x, N=10000, dist='beta'):
    if dist == 'beta':
        return sample_y_beta(fc, x, N)
    if dist == 'normal':
        return sample_y_normal(fc, x, N)


def estimate_feasibility(s, fc, N=100000, dist='beta', joint=False):
    k = 0
    feas_list = np.zeros((1, N)) < 1
    feas = 1.0
    for i, x in enumerate(s.Xvar):
        if not isinstance(x, (float, int)):
            feas_block = estimate_feasibility_block(s, fc, k, i, N, dist)
            import ipdb; ipdb.set_trace()
            feas_list = feas_list & feas_block
            feas = min(feas, np.sum(feas_block)/N)
            k = i
    feas_block = estimate_feasibility_block(s, fc, k, len(s.X) - 1, N, dist)
    feas_list = feas_list & feas_block
    feas = min(feas, np.sum(feas_block)/N)
    if joint:
        feas = np.sum(feas_list)/N
    return feas


def estimate_feasibility_block(s, fc, k, i, N, dist):
    X = s.X[k:i]
    Xvar = s.Xvar
    delta = {s.X[j]: Xvar[j+1] - Xvar[j] for j in range(k, i)}
    dp = [[s.m.rop[x].deltap()] for x in X]
    dp = np.array(dp)
    dt = [[delta[x]/(s.m.rop[x].V + 0.001)] for x in X]
    dt = [[x[0]()] for x in dt]
    dt = _to_np_obj_array(dt)
    r = sample_y(fc, dp, N=N, dist=dist)
    obj = np.matmul(dt.T, r)
    return obj <= 1
