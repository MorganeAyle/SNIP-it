from models.criterions.NEW import NEW
from models.criterions.NEWit import NEWit


class NEWitDuring(NEWit):

    """
    Original creation from our paper:  https://arxiv.org/abs/2006.00896
    Implements SNIP-it (during training)
    """

    def __init__(self, *args, **kwargs):
        super(NEWitDuring, self).__init__(*args, **kwargs)

    def prune(self, percentage=0.0, *args, **kwargs):
        if len(self.steps) == 0:
            print("finished all pruning events already")
            return

        if len(self.steps) > 0:
            # determine k_i
            percentage = self.steps.pop(0)
            kwargs["percentage"] = percentage

            # prune
            NEW.prune(self, **kwargs)
