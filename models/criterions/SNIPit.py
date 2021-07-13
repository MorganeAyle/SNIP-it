from models.criterions.SNIP import SNIP


class SNIPit(SNIP):

    """
    Original creation from our paper:  https://arxiv.org/abs/2006.00896
    Implements SNIP-it (before training)
    """

    def __init__(self, *args, limit=0.0, steps=5, lower_limit=0.5, **kwargs):
        self.limit = limit
        super(SNIPit, self).__init__(*args, **kwargs)
        if limit > 0.5:
            lower_limit = 0.5
        else:
            lower_limit = 0.2
        # always smaller than limit, steps+1 elements (including limit)
        self.steps = [limit - (limit - lower_limit) * (0.5 ** i) for i in range(steps + 1)] + [limit]

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage=0.0, *args, **kwargs):
        while len(self.steps) > 0:

            # determine k_i
            percentage = self.steps.pop(0)

            # prune
            super().prune(percentage=percentage, *args, **kwargs)
