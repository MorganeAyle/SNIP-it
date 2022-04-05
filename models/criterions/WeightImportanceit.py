from models.criterions.WeightImportance import WeightImportance
from copy import deepcopy


class WeightImportanceit(WeightImportance):

    """
    Original creation from our paper:  https://arxiv.org/abs/2006.00896
    Implements SNIP-it (before training)
    """

    def __init__(self, *args, limit=0.0, steps=5, lower_limit=0.5, orig_scores=False, **kwargs):
        self.limit = limit
        super(WeightImportanceit, self).__init__(*args, **kwargs)
        if limit > 0.5:
            lower_limit = 0.5
        else:
            lower_limit = 0.2
        # always smaller than limit, steps+1 elements (including limit)
        # self.steps = [limit - (limit - lower_limit) * (0.5 ** i) for i in range(steps - 1)] + [limit]
        self.steps = [0.5, 0.5]
        self.orig_scores = orig_scores

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage=0.0, *args, **kwargs):
        self.grads_abs = None
        while len(self.steps) > 0:

            # determine k_i
            percentage = self.steps.pop(0)

            criterion = WeightImportance(model=self.model, orig_scores=self.orig_scores)

            # prune
            criterion.prune(percentage=percentage, *args, **kwargs)

            # if self.grads_abs is None:
            #     self.grads_abs = criterion.grads_abs
            # else:
            #     for name, grads in criterion.grads_abs.items():
            #         self.grads_abs[name] -= (grads != 0).float() * self.grads_abs[name]
            #         self.grads_abs[name] += grads

        self.grads_abs = criterion.grads_abs
        if self.orig_scores:
            self.scores_mean = criterion.scores_mean
            self.scores_std = criterion.scores_std
