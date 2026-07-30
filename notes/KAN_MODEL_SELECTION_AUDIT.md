# KAN Model-Selection Audit

## Citation

Yuntian Hou, Tianrui Ji, Di Zhang, and Angelos Stefanidis,
*Kolmogorov-Arnold Networks: A Critical Assessment of Claims, Performance,
and Practical Viability*, arXiv:2407.11075v8 (2025),
[https://arxiv.org/abs/2407.11075](https://arxiv.org/abs/2407.11075).

## Relevant finding

The assessment reports that Kolmogorov-Arnold Networks (KANs) are most
competitive on symbolic-regression tasks. It finds no general empirical
advantage over parameter- and compute-matched multilayer perceptrons on the
broader benchmark classes it surveys, attributes much of the observed effect
to spline activations, and records substantial computational overhead.

## Decision for this repository

KANs are not used to replace exact combinatorics, projective identities,
topological certificates, or the directly measured Hopf-to-Hex crossing
curves. In the current experiment:

- the Hopf map is available in closed form;
- common-`U(1)` invariance is tested directly;
- Hex crossing is evaluated exactly on every sampled completed board; and
- the threshold response is a one-dimensional monotone empirical curve.

A learned surrogate would add approximation error without improving those
checks. If a later experiment asks for a compact symbolic law relating several
geometric controls to crossing probability, a KAN may enter only as a
candidate symbolic-regression model. It must then be compared against at
least a monotone spline, logistic/generalized-linear model, and matched MLP
using held-out error, parameter count, training time, inference time, and
stability across seeds.

## Evidential boundary

Good predictive fit would not prove a Four Color, knot, gauge, or continuum
statement. A fitted expression would be a conjecture generator whose proposed
identity or inequality still requires an independent proof or exact
certificate.
