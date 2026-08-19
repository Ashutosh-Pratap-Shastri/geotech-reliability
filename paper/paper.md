---
title: 'geotech-reliability: A generic Monte Carlo reliability engine with a validated library of geotechnical limit states'
tags:
  - Python
  - geotechnical engineering
  - reliability analysis
  - Monte Carlo
  - slope stability
  - rock mechanics
authors:
  - name: Ashutosh Pratap Shastri
    orcid: TODO
    affiliation: 1
affiliations:
  - name: TODO (department, institution)
    index: 1
date: TODO
bibliography: paper.bib
---

# Summary

`geotech-reliability` separates two things that are usually bundled
together in geotechnical software: the *reliability method* (how you
sample uncertain inputs and estimate a probability of failure) and the
*limit-state function* (the specific physics of a specific failure
mode). The package provides a small, generic Monte Carlo reliability
engine that operates on any object implementing a two-method
`LimitState` interface, plus a library of limit-state functions
implementing two geotechnical problems: slope stability by Bishop's
Simplified method of slices [@bishop1955], and the stability of a lined
rock cavern under internal gas pressure, evaluated against wall-crushing
(via the closed-form Kirsch stress solution [@kirsch1898] and a
simplified Hoek-Brown rock mass strength estimate [@hoek2002]) and
hydraulic-jacking (minimum-principal-stress cover) limit states. New
limit states — new failure modes, new problem domains — can be added by
implementing `evaluate()` and a dictionary of input `RandomVariable`s,
without modifying the reliability engine itself.

# Statement of need

Reliability-based design is increasingly used in geotechnical
engineering, particularly for high-consequence infrastructure such as
underground hydrogen and compressed-air energy storage, where
deterministic factor-of-safety checks alone are considered insufficient
to characterize risk. In practice, however, probabilistic analysis is
usually implemented either (a) inside commercial, closed-source
FEM/FDM-coupled software, which is expensive and not extensible by
researchers, or (b) ad hoc in spreadsheets coupling a single
deterministic calculation to a Monte Carlo add-in, with no reusable
separation between the sampling/reliability logic and the underlying
physics. `geotech-reliability` addresses this gap with a small, tested,
openly licensed Python package in which the reliability engine and the
limit-state physics are decoupled, so that researchers can validate a
new limit-state function independently and immediately reuse the
existing, tested Monte Carlo (and, in an upcoming release, FORM)
machinery, including system (union) reliability across multiple
correlated failure modes — a common need in cavern and slope design
where more than one failure mechanism competes (e.g. wall crushing vs.
hydraulic jacking under the same internal pressure).

# Comparison to existing software

TODO before submission: expand with explicit comparison to PySlope
[@pyslope] (deterministic slope stability only, no reliability layer),
GeoStudio/Slide2/GEO5 (commercial, closed-source, include reliability
modules but not extensible), and any open FORM/Monte Carlo reliability
packages from other engineering domains (e.g. structural reliability
toolkits) that are not geotechnics-specific.

# Validation

The Bishop's Simplified implementation is validated against the
closed-form infinite-slope factor-of-safety solution under a slip
geometry that approximates an infinite slope (see
`tests/test_slope_bishop.py`), matching to within 1e-6 relative error.
The Monte Carlo engine is validated against the exact closed-form
probability of failure for a linear limit state with normally
distributed capacity and demand (see `tests/test_monte_carlo.py`).
TODO before submission: add validation of the cavern limit states
against an independent numerical model or published case study, per the
package's own stated limitation (see README "Validation status").

# Acknowledgements

TODO.

# References
