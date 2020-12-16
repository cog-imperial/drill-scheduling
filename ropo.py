#!/usr/bin/env
import time
import pyomo.environ as p


class ROPO():
    """
    Connect ROP to weight-on-bit and drill string rotational speed.

    Args:
        rock: Rock object
        drillstring: DrillString object
        m: pyomo model/block to use (for use within scheduling model)

    """
    def __init__(self, rock, drillstring, m=None, piecewise=False):
        if m is None:
            # Initialize pyomo model if none was provided
            self.m = p.ConcreteModel()
        else:
            self.m = m
        self.bit = drillstring.bit
        self.pdm = drillstring.pdm
        self.rock = rock
        self.m.cons = p.ConstraintList()
        # Add variables, constraints and objective to model
        self.add_vars(piecewise=piecewise)
        self.add_cons(piecewise=piecewise)
        if m is None:
            self.add_obj()

    # TODO: this seems to be obsolete?
    def build_block(self, piecewise=False):
        b = p.block()
        self.add_vars(b, piecewise=piecewise)
        self.add_cons(b, piecewise=piecewise)

    def add_vars(self, piecewise=False):
        """ Add pyomo variables to model. """
        m = self.m
        m.d = p.Var(within=p.NonNegativeReals)       # depth of cut per revol.
        m.Ntop = p.Var(within=p.NonNegativeReals,
                       bounds=(0, 200))              # RPM at top/drillstring
        m.Npdm = p.Var(within=p.NonNegativeReals)    # PDM RPM (relative)
        m.deltap = p.Var(within=p.NonNegativeReals)  # Differential pressure
        wstar = self.rock.wstar
        if piecewise:
            wmin = 0
        else:
            wmin = wstar
        m.w = p.Var(within=p.NonNegativeReals,
                    bounds=(wstar, 2*wstar))  # Reduced weight-on-bit
        # m.W = p.Var(within=p.NonNegativeReals)     # Weight-on-bit
        m.t = p.Var(within=p.NonNegativeReals)       # Reduced torque
        m.Tcap = p.Var(within=p.NonNegativeReals,
                       bounds=(0, 100000))           # Torque
        m.V = p.Var(within=p.NonNegativeReals,
                    bounds=(0.0, 1000))              # Rate of penetration

    def add_obj(self):
        m = self.m
        # Maximize ROP V = d*(N_top + N_pdm)
        V = m.d * (m.Ntop + m.Npdm)
        m.Obj = p.Objective(expr=V, sense=p.maximize)

    def add_cons(self, piecewise=False):
        """ Add Detournay and powercurve constraints to model. """
        m = self.m
        bit = self.bit
        rock = self.rock

        # Detournay rock-bit interaction model
        m.cons.add(m.t == 2*m.Tcap*1000/(bit.a**2*(1 - bit.rho**2)))
        if piecewise:
            x = [0, rock.wstar, 2*rock.wstar]
            y = [0, rock.wstar/rock.Sstar,
                 rock.wstar/rock.Sstar + rock.wstar/rock.xieps]
            # TODO: Pyomo BIGM_BIN is deprecated
            m.dvsw = p.Piecewise(m.d, m.w, pw_pts=x,
                                 pw_constr_type='EQ',
                                 pw_repn='BIGM_BIN',
                                 f_rule=y)
            transition = rock.betaw/(1 - rock.mugam*rock.xi)

            x = [0, transition, max(2*transition, 2*rock.wstar)]
            y = [0, rock.mugam*x[1], 1/rock.xi*(x[2] - rock.betaw)]
            m.tvsw = p.Piecewise(m.t, m.w, pw_pts=x,
                                 pw_constr_type='EQ',
                                 pw_repn='BIGM_BIN',
                                 f_rule=y)
        else:
            m.cons.add(m.d == rock.wstar/rock.Sstar
                       + (m.w - rock.wstar)/rock.xieps)
            m.cons.add(m.t == 1/rock.xi*(m.w - rock.betaw))
        # Powercurve relationships
        m.cons.add(m.Npdm == self.pdm.rpm(m.deltap))
        m.cons.add(m.Tcap == self.pdm.torque(m.deltap)*1355.82)

        m.cons.add(m.V == m.d/1000 * (m.Ntop + m.Npdm)*60)

    def solve(self, solver='Bonmin'):
        """
        Solve model.

        Args:
            solver: gams solver to use

        """
        tstart = time.perf_counter()
        self.solver = p.SolverFactory('gams', solver_io='shell')
        results = self.solver.solve(self.m,
                                    logfile='drilling.log',
                                    io_options={'solver': solver},
                                    symbolic_solver_labels=True)
        results.write()
        self.time = time.perf_counter() - tstart
        return results.solver.status, results.solver.termination_condition
