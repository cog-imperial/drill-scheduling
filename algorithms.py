#!/usr/bin/env
import copy
import time
import numpy as np


def no_degradation_heuristic(s, Rmax=10):
    s.build(Rmax=Rmax)
    s.multisolve()
    s = reconstruct_r(s)
    return s


def no_degradation_start_heuristic(s):
    t_start = time.time()
    s = no_degradation_heuristic(s)
    Xmax = get_maintenance_locations(s)
    s.build(Xm=Xmax)
    s.multisolve()
    s.time = time.time() - t_start
    return s, s.m.Obj()


def boundary_heuristic(s):
    t_start = time.time()
    s = no_degradation_heuristic(s)
    Xm = get_maintenance_locations(s)
    best_obj = float('Inf')
    best_s = None
    n_iter = 0
    done = False
    # Iterate until all # of maint have been explored
    while not done:
        # Iterate while optimal maint. is at segment bound
        at_bound = True
        while at_bound:
            s.build(Xm=Xm)
            s.multisolve()
            print(Xm)
            n_iter += 1
            if s.m.Obj() < best_obj:
                best_s = copy.copy(s)
                best_obj = s.m.Obj()
            Xm, at_bound = check_if_at_bound(s)
        if len(Xm) > 0:
            Xm.pop(0)
        else:
            done = True
    best_s.n_iter = n_iter
    best_s.time = time.time() - t_start

    return best_s, best_obj


def midpoints(xstart, xfin, n):
    eps = (np.array(range(n)) + 1)/(n + 1)
    x = xstart + (xfin - xstart)*eps
    return x.tolist()


def Xm_from_Nm(X, Nm):
    Xm = []
    for n in set(Nm):
        Xm += midpoints(X[n], X[n+1], Nm.count(n))
    Xm.sort()
    return Xm


def enum(s):
    t_start = time.time()
    s = no_degradation_heuristic(s)
    geo = s.geology
    X = geo.transitions + [s.xfin]
    Xmax = get_maintenance_locations(s)
    Nmax = [geo.segment(x) for x in Xmax]
    s.build(Xm=Xmax)
    s.multisolve()
    best_s = copy.copy(s)
    best_obj = s.m.Obj()
    n_iter = 0
    for m in reversed(range(len(Xmax))):
        Nm = [0] * (m + 1)
        Nmax = Nmax[-(m + 1):]
        i = m
        while not Nm == Nmax:
            if Nm[i] < Nmax[i]:
                Nm[i:] = [Nm[i] + 1] * (m + 1 - i)
                i = m
                Xm = Xm_from_Nm(X, Nm)
                print(Xm, Nm, Nmax)
                s.build(Xm=Xm)
                s.multisolve()
                n_iter += 1
                if s.m.Obj() <= best_obj and s.opt > 0:
                    best_s = copy.copy(s)
                    best_obj = s.m.Obj()
            else:
                i -= 1
    s.build(Xm=[])
    s.multisolve()
    if s.m.Obj() <= best_obj and s.opt > 0:
        best_s = copy.copy(s)
        best_obj = s.m.Obj()
    best_s.n_iter = n_iter
    best_s.time = time.time() - t_start

    return best_s, best_obj


def reconstruct_r(s):
    R = s.m.R[s.X[-2]]()
    xfin = s.X[-1]
    for x in reversed(s.X[:-1]):
        s.m.R[x].value = R
        r = s.m.r[x]()
        rop = s.m.rop[x].V()
        dx = xfin - x
        R = R - dx/rop*r
        xfin = x
    return s


def get_maintenance_locations(m):
    df = m.get_schedule()
    R = df['R'].max()
    RR = df['R'].tolist()
    RR.insert(0, 0)
    X = df['x'].tolist()
    X.append(m.xfin)
    Xm = []
    while R > 1:
        R -= 1
        i = df[df['R'] >= R].index[0]
        dx = X[i+1] - X[i]
        dR = RR[i+1] - RR[i]
        Xm.append(X[i] + dx * (R - RR[i])/dR)
    Xm.sort()
    return Xm


def to_number(x):
    if not isinstance(x, (int, float)):
        x = x.value
    return x


def check_if_at_bound(m):
    Xm = []
    X = [to_number(x) for x in m.Xvar]
    at_bound = False
    tol = 10e-3
    for i, x in enumerate(m.Xvar):
        if not isinstance(x, (int, float)):
            # Drop maint if it is at xstart or xfin
            if X[i] - X[0] < tol or X[-1] - X[i] < tol:
                at_bound = True
            # Move to previous segment if at min bound
            elif X[i] - X[i-1] < tol:
                Xm.append((X[i-1] + X[i-2])/2)
                at_bound = True
            # Move to next segment if at max bound
            elif X[i+1] - X[i] < tol:
                Xm.append((X[i+1] + X[i+2])/2)
                at_bound = True
            # Keep it where it is if not at bound
            else:
                Xm.append(X[i])
    Xm.sort()
    return Xm, at_bound
