My attempt at implementing [Can Neural Networks Understand Logical Entailment?](https://arxiv.org/abs/1802.08535)

## Details of my implementation (so far)

This implemetation parallelises computation over each batch. It bundles together;
- the n-ary operations over each depth first traversal in a batch.
- the different worlds.

It would be possible to do more bundling;
- over levels of a tree rather than depth first traversal.
but I didnt come up with nice to do it without refactoring most of my code...

I attempted to train the network, but my computer is slow, google cloud is more expensive than I thougt, and this architecture is slow (maybe I have done something wrong...).
The typical result I observed was for the loss to decrease, and eventually flatten at 0.69 (aka 50:50). The losses variance also reduces.

## Baseline

In [notes](/notes) I included a baseline that the paper didnt. Possible worlds (not learned...).
Randomly pick a set of worlds (assignments of `True`, `False` to the variables) and evaluate entailment.

## Lessons learned

- The key to fast dynamic computation is bundling. Grouping together operations so they can be exectued within a single op (that already knows how to efficiently parallelise them).
- Bundling your compute means unintuitive code (at least the way I did it)... Want to nice syntactic sugar.
- Instantiating a variable from a tensor removes the tenors prior dependencies.
- Always check the gradients!
- TPUs are more expensive than I thought...

## Wait, but why?

Why do we care?

> Which architectures are best at inferring, encoding, and relating features in a purely structural sequence-based problem?”. In answering these questions, we aim to better understand the inductive biases of popular architectures with regard to structure and abstraction in sequence data. Such understanding would help pave the road to agents and classifiers that reason structurally, in addition to reasoning on the basis of essentially semantic representations.

Ok, but that doesnt really say anything about why we should care about entailment. Oh, so this is less about the specific logical problem but rather we are exploring how a NN can learn to do logic?

Want architectures that can generalise to;
- different symbols, (as the actualy symbols are meaningess, only their relationships are important)
- sentences with variable complexity/length, (once you know the rules of how a language works, the only constraint on parsing complex sentences should be computational -- yes, but how does the complexity scale, it must be efficient, linear?)

## Train your own

You will need to get clone the [data repo](https://github.com/deepmind/logical-entailment-dataset) and put somewhere convinient.

```
python train_possible_worlds_net.py
tensorboard --logdir=/tmp/pwn
```

## Tests

Yea, tensorflow eager doesnt work very nicely here (state is carried across tests...). Not sure how to fix, but didnt spend much time investigating. Just run each test independently.

## Future work

Not sure when/if ill get around to it.

- Learned parser
- Visualise the embeddings
- Comparse against recurrent net with `Sat3cell`s
