My attempt at implementing [Can Neural Networks Understand Logical Entailment?](https://arxiv.org/abs/1802.08535)

## Details of my implementation (so far)

This implemetation parallelises computation over each batch. It bundles together;
- the n-ary operations over each depth first traversal in a batch.
- the different worlds.

It would be possible to do more bundling;
- over levels of a tree rather than depth first traversal.
but I didnt come up with nice to do it without refactoring most of my code...

## Baseline

In [notes](/notes) I included a baseline that the paper didnt. Possible worlds (not learned...).
Randomly pick a set of worlds (assignments of `True`, `False` to the variables) and evaluate entailment.

## Lessons learned

- The key to fast dynamic computation is bundling. Grouping together operations so they can be exectued within a single op (that already knows how to efficiently parallelise them).
- Bundling your compute means unintuitive code (at least the way I did it)... Want to nice syntactic sugar.
- Instantiating a variable from a tensor removes the tenors prior dependencies.
- Always check the gradients!

## Wait, but why?

Why do we care about entailment?
Still not sure... TODO

## Tests

TODO... `python -m unittest`
