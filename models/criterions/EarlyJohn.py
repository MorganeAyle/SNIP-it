from models.criterions.John import John


class EarlyJohn(John):
    """
    Original creation from our paper:  https://arxiv.org/abs/2006.00896
    Implements SNIP-it (before training)
    """

    def __init__(self, *args, limit=0.0, lower_limit=0.5, steps=100, **kwargs):
        self.limit = limit
        super(EarlyJohn, self).__init__(*args, **kwargs)
        if limit > 0.5:
            lower_limit = 0.5
        else:
            lower_limit = 0.2
        self.steps = [limit - (limit - lower_limit) * (0.5 ** i) for i in range(steps + 1)] + [limit]

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage=0.0, local=False, *args, **kwargs):
        while len(self.steps) > 0:

            # # determine k_i
            # percentage = self.steps.pop(0)
            #
            # # prune
            # super().prune(percentage=percentage, *args, **kwargs)
            # if len(self.steps) != 0:
            #     while self.model.pruned_percentage > self.steps[0]:
            #         self.steps.pop(0)

            if len(self.steps) != 0:
                while self.model.pruned_percentage > self.steps[0]:
                    self.steps.pop(0)
                    if len(self.steps) == 0:
                        break

                if len(self.steps) == 0:
                    break

                # determine k_i
                percentage = self.steps.pop(0)
                print(percentage)

                # prune
                super().prune(percentage=percentage, local=local, *args, **kwargs)
