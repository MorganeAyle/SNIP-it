from models.criterions.SNIPAdv import SNIPAdv
from models.criterions.SNIPitAdv import SNIPitAdv


class SNIPitDuringAdv(SNIPitAdv):

    """
    Original creation from our paper:  https://arxiv.org/abs/2006.00896
    Implements SNIP-it (during training)
    """

    def __init__(self, *args, **kwargs):
        super(SNIPitDuringAdv, self).__init__(*args, **kwargs)

    def prune(self, percentage=0.0, *args, **kwargs):
        if len(self.steps) == 0:
            print("finished all pruning events already")
            return

        if len(self.steps) > 0:
            # determine k_i
            percentage = self.steps.pop(0)
            kwargs["percentage"] = percentage

            # prune
            SNIPAdv.prune(self, **kwargs)
