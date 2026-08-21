\---

title: 'geotech-reliability: A generic Monte Carlo/FORM reliability engine with a validated library of geotechnical limit states'

tags:

&#x20; - Python

&#x20; - geotechnical engineering

&#x20; - reliability analysis

&#x20; - Monte Carlo

&#x20; - FORM

&#x20; - slope stability

&#x20; - rock mechanics

authors:

&#x20; - name: Ashutosh Pratap Shastri

&#x20;   orcid: 0009-0007-9117-1591

&#x20;   affiliation: 1

&#x20; - name: Satyabrata Behera

&#x20;   orcid: null

&#x20;   affiliation: 1

affiliations:

&#x20; - name: Department of Mining Engineering, Indian Institute of Technology (BHU), Varanasi, Uttar Pradesh, India

&#x20;   index: 1

date: 19 August 2026

bibliography: paper.bib

\---



\# Summary



`geotech-reliability` separates two things that are usually bundled

together in geotechnical software: the \*reliability method\* (how you

sample uncertain inputs and estimate a probability of failure) and the

\*limit-state function\* (the specific physics of a specific failure

mode). The package provides a small, generic Monte Carlo and First-Order

Reliability Method (FORM) engine that operates on any object

implementing a two-method `LimitState` interface, plus a library of limit-state functions

implementing two geotechnical problems: slope stability by Bishop's

Simplified method of slices \[@bishop1955], and the stability of a lined

rock cavern under internal gas pressure, evaluated against wall-crushing

(via the closed-form Kirsch stress solution \[@kirsch1898] and a

simplified Hoek-Brown rock mass strength estimate \[@hoek2002]) and

hydraulic-jacking (minimum-principal-stress cover) limit states. New

limit states — new failure modes, new problem domains — can be added by

implementing `evaluate()` and a dictionary of input `RandomVariable`s,

without modifying the reliability engine itself.



\# Statement of need



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

existing, tested Monte Carlo and FORM machinery, including system

(union) reliability across multiple

correlated failure modes — a common need in cavern and slope design

where more than one failure mechanism competes (e.g. wall crushing vs.

hydraulic jacking under the same internal pressure).



\# Comparison to existing software



Several open-source Python packages already implement deterministic

slope-stability limit-equilibrium analysis, including `PySlope`

\[@pyslope], which implements Bishop's Simplified and Janbu's Simplified

methods with an object-oriented interface, and `pyCSS` \[@pycss], which

implements Fellenius and Bishop's methods for circular slip surfaces.

Both are useful, focused tools, but neither provides a reliability

layer: they return a deterministic factor of safety rather than a

probability of failure, and neither separates the reliability \*method\*

from the slope-specific \*physics\* in a way that would let a user reuse

the same sampling/reliability-index machinery for an unrelated

geotechnical problem (e.g. cavern stability). Commercial packages such

as Slide2, GeoStudio, and RS2 do provide probabilistic/reliability

modules alongside deterministic analysis, but are closed-source,

licensed, and not extensible or inspectable by researchers.

`geotech-reliability` differs from both groups by treating the

reliability engine (Monte Carlo, FORM) as the reusable core and the

physics (slope stability, cavern stability, or any future limit state)

as a thin, pluggable layer implementing a two-method interface — so

adding reliability analysis to a new geotechnical problem does not

require re-implementing sampling, FORM's HL-RF search, or system

(union) reliability across correlated failure modes.



\# Validation



The Bishop's Simplified implementation is validated against the

closed-form infinite-slope factor-of-safety solution under a slip

geometry that approximates an infinite slope (see

`tests/test\_slope\_bishop.py`), matching to within 1e-6 relative error.

The Monte Carlo engine is validated against the exact closed-form

probability of failure for a linear limit state with normally

distributed capacity and demand (see `tests/test\_monte\_carlo.py`). FORM

is validated against the same closed-form linear case, for which FORM

is mathematically exact rather than approximate, and is cross-checked

against Monte Carlo on the nonlinear slope and cavern limit states,

where the two methods agree in sign and lie within a generous tolerance

of one another (see `tests/test\_form.py`); the residual disagreement is

expected, since FORM linearizes an inherently nonlinear limit-state

surface. The cavern wall-crushing and hydraulic-jacking limit states

are included primarily to demonstrate that the reliability engine

generalizes beyond slope stability to a structurally different

problem (stress-based rather than limit-equilibrium); their individual

formulas (Kirsch stress solution, simplified Hoek-Brown strength) are

checked against their published closed forms, but the resulting

system-level reliability estimates have not yet been benchmarked

against an independent numerical model or a published case study. This

is stated explicitly as a limitation in the package documentation, and

addressing it is intended as near-term future work rather than a

precondition for the reliability \*engine\* itself, which is validated

independently of any particular limit state.



\# Acknowledgements



The authors thank the Department of Mining Engineering, Indian

Institute of Technology (BHU) Varanasi, for institutional support

during the development of this software.



\# References

