My attempt at implementing [Can Neural Networks Understand Logical Entailment?](https://arxiv.org/abs/1802.08535)

## Details of my implementation

> This architecture is slow... maybe bc of the lack of parallelism.

This implemetation parallelises computation over each batch. It bundles together the n-ary operations over each depth first traversal in a batch.
It would be possible to do more bundling (over levels of a tree rather than depth first traversal, and over worlds) but I didnt come up with nice to do it without a refactoring of my code so far...

> tf.Eager doesnt give gradients for most of my variables...

Not sure what this is about. Currently I am not recieving gradients for any variable used within the encoder (ie worlds, treenn and sat3), only for the dense layer in PWN.
Current hypothesis, the `scatter_add` doesnt work as intended. You cant instantiate new variables like that...

## Baseline

In [notes](/notes) I included a baseline that the paper didnt. Possible worlds (not learned...).
Randomly pick a set of worlds (assignments of `True`, `False` to the variables) and evaluate entailment. This approach achieves similar to the best baselines provided.

## Lessons learned

- The key to fast dynamic compution is bundling. Grouping together operations so they can be exectued as a single op.
- Bundling your compute means unintuitive code (at least the way I did it)... Want to nice syntactic sugar.
-
