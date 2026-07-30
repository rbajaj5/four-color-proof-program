# Jaffe-Liu Picture-Language Source Audit

## Citation

Arthur Jaffe and Zhengwei Liu, *A Mathematical Picture Language Program*,
*Proceedings of the National Academy of Sciences* **115** (2018), 81-86,
[doi:10.1073/pnas.1710707114](https://doi.org/10.1073/pnas.1710707114).
The user supplied the authors'
[PDF](https://arthurjaffe.com/Assets/pdf/PictureLanguage.pdf).

## Repository use

The paper motivates a precise algebraic gate rather than a visual analogy:

1. pictures have declared composition and reflection operations;
2. gluing compatible boundaries represents an algebraic pairing;
3. reflected doubles can define positive quadratic forms; and
4. a picture proof must be backed by a simulation map into established
   mathematics.

The repository implements two finite instances:

- noncrossing boundary pairings with loop-weight evaluation, including an
  explicit color-indicator Gram factorization; and
- Penrose half-pictures whose boundary amplitudes are exact contractions of
  the three-color Levi-Civita tensor.

## What the finite gate establishes

At an integer loop weight `d`, the pairing kernel

```text
G(P,Q) = d ^ loops(P glued to Q)
```

is a Gram matrix: map each pairing to the indicator of boundary colorings
that are constant on every pair. Its inner product with a second indicator
counts exactly `d ^ loops(P glued to Q)`. This gives a constructive
sum-of-squares explanation of positivity for the tested smoothing language.

For a Penrose half-picture with boundary-amplitude vector `A`, gluing it to
its reflected copy evaluates to `sum_c A(c)^2`. The exact tripod fixture gives
`6`, and a two-vertex channel gives `12`.

## Obstruction and evidential boundary

Mixed gluing is an inner product, not a norm square. Reversing the cyclic
orientation of the tripod changes its amplitude from `A` to `-A`, so the
mixed evaluation is `-6` even though both reflected doubles equal `6`.

Therefore reflection positivity for doubles does not imply positivity of
arbitrary Penrose contractions, and it does not prove that every target link
diagram has a colorable smoothing. A successful Four Color application still
needs a proved reduction that places the relevant diagram evaluation inside
a positive reflected-double construction without discarding the existence
predicate.

