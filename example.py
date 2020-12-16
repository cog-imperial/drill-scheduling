import curves
import tools
import geology
import algorithms
import sep
import scheduling.continuous as sc
import numpy as np


if __name__ == '__main__':

    # Tools
    # Bit
    bit = tools.Bit(100, 0.0)
    # Motor curves
    rpmcurve = curves.Quadratic(np.array([[0., 100.],
                                          [1400., 60.],
                                          [2500., 0.]]))
    torquecurve = curves.Quadratic(np.array([[0., 0.],
                                             [1400., 3.5],
                                             [2500., 5.0]]))
    # Motor degradation curve
    data = np.load('gp_data.npy')
    x, y = data[:,0,None], data[:,1,None]
    failurecurve = curves.WarpedGP(x, y, warping_terms=2)
    initial_degradation = 1
    # PDM
    pdm = tools.PDM(rpmcurve,
                    torquecurve,
                    failurecurve,
                    initial_degradation,
                    1/1500/25)
    # Drill string
    drillstring = tools.DrillString(pdm, bit)
    # Cost function maintenance
    cost_maint = curves.Linear(np.array([[0., 4.],
                                         [4000., 20.]]))

    # Geology
    rock1 = geology.Rock(315, 68.6, 50, 0.93, 33, 0.65)
    rock2 = geology.Rock(278, 330, 125, 0.48, 157, 0.98)
    geo = geology.Geology({0: rock1,
                           800: rock2})

    # Enumeration
    algo = algorithms.enum

    # Boundary heuristic
    # algo = algorithms.boundary_heuristic

    # No-degradation heuristic
    # algo = algorithms.no_degradation_start_heuristic

    # Model
    scheduler = sc.Deterministic(geo, drillstring, cost_maint, xfin=3500)
    # scheduler = sc.Wolfe(geo, drillstring, cost_maint, xfin=3500, alpha=0.95)
    scheduler.build()
    scheduler, obj = algo(scheduler)
    scheduler.print_schedule()
