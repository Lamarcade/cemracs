"""IFRS 9 Stage 2 extension: lifetime ECL triggered by a significant increase in credit risk.

The main engine one level up is Stage 1 throughout -- the impairment is the 12-month
ECL, so ``Psi_g(p) = ell*p + 12.5*R_star*K(p)`` is a pointwise function of the current
PD. This package adds the Stage 2 switch described in appendix A of the underlying
note: past a degradation threshold, the impairment jumps to the lifetime ECL, which
*anticipates* the loss instead of spreading it.

That timing is not neutral for a first-passage breach. Anticipation digs an earlier,
deeper trough than the smoothed 12-month path ever reaches, even when terminal capital
coincides -- which is precisely why it matters to a reverse stress test.

Scope, deliberately narrow (see the module docstrings for the reasoning):

- SICR is ``p_scenario / p_baseline > Thresh``, re-tested at every date.
- Sector granularity: a bucket is one ``(sector, region)`` pair.
- Lifetime is a fixed 2.5-year window, the F-IRB maturity ``M`` that ``MA(p)`` already uses.
- **The critical PD is out of scope.** It is what would force an approximation, because
  inverting a discontinuous ``Psi`` needs an assumption about the future path. Scenario
  evaluation needs no such thing: the multiplier is read off the projected trajectory,
  so everything here is exact.

Everything else is reused from the parent package unchanged --
``regulatory.capital_charge``, ``breach.cushion``, ``breach.calibrate_cet1_for_ratio``,
``portfolio``, ``scenarios``, ``config``, ``report``.
"""
