import tensorflow as tf

class TreeNN():
  def __init__(self, cell, parser, batch_size):
    self.cell = cell
    self.parser = parser
    self.batch_size = batch_size

    self.variables = self.cell.variables

  def __call__(self, w, trees):
    """
    Applies `self.cell` according to the tree(s) generated by `self.parser`.

    Args:
      w: a world
      trees (list): stacked outputs of self.parser

    Returns: (batch_size x cell.output_size)
    """
    # a stack for keeping track of computed nodes
    states = []

    # TODO could be smarter here. bundle across levels of the tree,
    # rather than a traversal. this would allow larger bundles.
    # would end up being a trade off for batch size?

    # depth first traversal across all the trees in the batch
    lens = [len(ops) for ops, _ in trees]
    for i in range(max(lens)):  # traverse the trees
        locs_n_ops_n_args = []
        for loc, opsnargs in enumerate(trees):  # for each element of our batch
            ops, args = opsnargs
            if i<len(ops):  # make sure we dont try to index out of range
                arg = [i+a for a in args[i]]  # relative indexing to absolute
                locs_n_ops_n_args.append((loc, ops[i], arg))

        state = self.cell(states, locs_n_ops_n_args, w)
        states.append(state)

    return states[-1]
