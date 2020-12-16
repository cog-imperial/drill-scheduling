#!/usr/bin/env
import utils
import rogp
import numpy as np
import scipy as sp
import pyomo.environ as p
from rogp.util.numpy import _to_np_obj_array, _pyomo_to_np


class Sep():
    def __init__(self, X):
        m = p.ConcreteModel()
        m.cons = p.ConstraintList()
        m.r = p.Var(X, within=p.NonNegativeReals, bounds=(0, 1))
        self.m = m

def check_feasibility(s, bb=False):
    k = 0
    feas = True
    if bb:
        check_block = check_deg_block_bb
    else:
        check_block = check_deg_block
    for i, x in enumerate(s.Xvar):
        if not isinstance(x, (float, int)):
            if not check_block(s, k, i):
                feas = False
                break
            k = i
    if feas:
        return check_block(s, k, len(s.X) - 1)

def check_deg_block(s, k, i):
    fc = s.drillstring.pdm.failure
    fc.rogp.set_tanh(False)
    # Initialize parameters
    alpha = 1 - (1 - s.alpha)/(len(s.Xm) + 1)
    F = sp.stats.norm.ppf(alpha)
    X = s.X[k:i]
    Xvar = s.Xvar
    delta = {s.X[j]: Xvar[j+1] - Xvar[j] for j in range(k, i)}
    dp = [[s.m.rop[x].deltap()] for x in X]
    dp = _to_np_obj_array(dp)
    # TODO: make eps = 0.001 a parameter
    dt = [[delta[x]/(s.m.rop[x].V + 0.001)] for x in X]
    dt = [[x[0]()] for x in dt]
    dt = _to_np_obj_array(dt)
    sep = Sep(X)
    r = _pyomo_to_np(sep.m.r, ind=X)

    # Calculate matrices
    Sig = fc.rogp.predict_cov_latent(dp).astype('float')
    inv = np.linalg.inv(Sig)
    hz = fc.rogp.warp(r)
    mu = fc.rogp.predict_mu_latent(dp)
    diff = hz - mu

    obj = np.matmul(dt.T, r)[0, 0]
    sep.m.Obj = p.Objective(expr=obj, sense=p.maximize)
    c = np.matmul(np.matmul(diff.T, inv), diff)[0, 0]
    sep.m.cons.add(c <= F)
    utils.solve(sep, solver='Baron')

    if obj() - 1.0 > 10e-5:
        return False

    return True


def get_deg_block(s, k, i):
    fc = s.drillstring.pdm.failure
    fc.rogp.set_tanh(False)
    # Initialize parameters
    alpha = 1 - (1 - s.alpha)/(len(s.Xm) + 1)
    F = sp.stats.norm.ppf(alpha)
    X = s.X[k:i]
    Xvar = s.Xvar
    delta = {s.X[j]: Xvar[j+1] - Xvar[j] for j in range(k, i)}
    dp = [[s.m.rop[x].deltap()] for x in X]
    dp = _to_np_obj_array(dp)
    # TODO: make eps = 0.001 a parameter
    dt = [[delta[x]/(s.m.rop[x].V + 0.001)] for x in X]
    dt = [[x[0]()] for x in dt]
    dt = _to_np_obj_array(dt)

    # Calculate matrices
    cov = fc.rogp.predict_cov_latent(dp).astype('float')*F
    mu = fc.rogp.predict_mu_latent(dp).astype('float')
    c = dt.astype('float')

    return mu, cov, c.flatten()


def check_deg_block_bb(s, k, i):
    print(k, i)
    mu, cov, c = get_deg_block(s, k, i)
    warping = s.drillstring.pdm.failure.rogp
    bb = rogp.util.sep.BoxTree(mu, cov, warping, c)
    lb, ub, node, n_iter, tt = bb.solve(max_iter=1000000, eps=0.001)
    if ub - 1 <= 0.001:
        return True
    else:
        return False


def get_extrema(s, k, i):
    fc = s.drillstring.pdm.failure
    mu, cov, c = get_deg_block(s, k, i)
    inv = np.linalg.inv(cov)
    rad = np.sqrt(np.diag(cov)[:, None])
    X = s.X[k:i]
    sep = Sep(X)
    m = sep.m
    xub = fc.rogp.warp_inv(mu + rad)
    xlb = fc.rogp.warp_inv(mu - rad)
    r = _pyomo_to_np(m.r, ind=X)
    hz = fc.rogp.warp(r)
    diff = hz - mu
    c = np.matmul(np.matmul(diff.T, inv), diff)[0, 0]
    obj = (c - 1)**2
    m.Obj = p.Objective(expr=obj, sense=p.minimize)
    extrema = []
    for i in range(mu.shape[0]):
        m.r[X[i]].value = xlb[i]
        m.r[X[i]].fixed = True
        utils.solve(sep, solver='Baron')
        r = _pyomo_to_np(m.r, ind=X, evaluate=True)
        hz = fc.rogp.warp(r)
        extrema.append(hz)
        m.r[X[i]].fixed = False
    return extrema

