from glow.John import John


class Johnit(John):
    """
    Original creation from our paper:  https://arxiv.org/abs/2006.00896
    Implements SNIP-it (before training)
    """

    def __init__(self, *args, limit=0.0, steps=5, lower_limit=0.25, **kwargs):
        self.limit = limit
        if self.limit > 0.5:
            self.lower_limit = 0.5
        else:
            self.lower_limit = 0.25
        super(Johnit, self).__init__(*args, **kwargs)
        self.steps = [limit - (limit - self.lower_limit) * (0.5 ** i) for i in range(steps + 1)] + [limit]
        # self.steps = [limit]

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage=0.0, *args, **kwargs):
        while len(self.steps) > 0:
            print(self.steps)
            # determine k_i
            percentage = self.steps.pop(0)

            # prune
            super().prune(percentage=percentage, *args, **kwargs)
