# Draft note for Daniel P. Sanders

Subject: A bounded smoothing-state diagnostic related to Four Color

Dear Professor Sanders,

I have been studying the 2026 Penrose-Kauffman reformulation of the Four
Color Theorem alongside the Robertson-Sanders-Seymour-Thomas
reducibility/unavoidability framework. I built an exhaustive finite checker
for the Boolean smoothing landscape described in Theorem 5.3 of
Kauffman-Silver-Williams.

For the odd wheel `W_(2k+2)`, the computation through `k=6` finds one
non-3-colorable maximum (the zero state) and exactly

```text
(4^k - 1) / 3
```

nonzero maxima, each with component graph `K3`. There is a natural proof
sketch by fixing the hub color, coloring the odd rim with the other three
colors, and quotienting by `S3`.

The pattern is emphatically not universal. Among 27 small 3-connected,
4-chromatic planar graphs from the NetworkX atlas, 25 have nonzero strict
maxima whose component graphs are still not 3-colorable. Thus the experiment
does not supply either reducibility or unavoidability and is not being
presented as a new Four Color proof.

I would value your judgment on two narrow questions:

1. Is the odd-wheel smoothing decomposition or its closed-form count already
   standard in the Penrose-polynomial literature?
2. Is there a useful way to compare these local-maxima component graphs with
   the boundary-coloring data used in reducibility tests, or are they largely
   orthogonal to the 633-configuration machinery?

The code records every state convention, exact CPU reconstruction check,
CUDA audit, and the small counterexamples. I would be glad to send the short
report and repository link.

Sincerely,

Ravi Bajaj
