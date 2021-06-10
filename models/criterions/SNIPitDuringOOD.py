from models.criterions.SNIPOOD import SNIPOOD
from models.criterions.SNIPitOOD import SNIPitOOD


class SNIPitDuringOOD(SNIPitOOD):

    """
    Original creation from our paper:  https://arxiv.org/abs/2006.00896
    Implements SNIP-it (during training)
    """

    def __init__(self, *args, **kwargs):
        super(SNIPitDuringOOD, self).__init__(*args, **kwargs)

    def prune(self, percentage=0.0, *args, **kwargs):
        if len(self.steps) == 0:
            print("finished all pruning events already")
            return

        if len(self.steps) > 0:
            # determine k_i
            percentage = self.steps.pop(0)
            kwargs["percentage"] = percentage

            # prune
            SNIPOOD.prune(self, **kwargs)
