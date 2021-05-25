import torch

from utils.system_utils import *
from models.trainers.DefaultTrainer import DefaultTrainer
from models.trainers.AdversarialTrainer import AdversarialTrainer


class MixedTrainer(AdversarialTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rewind_happened = False

    def _handle_pruning(self, epoch):
        if self._is_pruning_time(epoch):
            if self._is_not_finished_pruning():
                self.out("\nPRUNING...\n")
                self._criterion.prune(
                    percentage=self._arguments['pruning_rate'],
                    train_loader=self._train_loader,
                    manager=DATA_MANAGER,
                    ood_loader=self._ood_loader,
                )
        if not self._is_not_finished_pruning():
            if self._model.is_rewindable and not self.rewind_happened:
                self.out("rewinding weights to checkpoint...\n")
                self._model.do_rewind()
                self.rewind_happened = True

        if self._is_checkpoint_time(epoch):
            self.out(f"\nCreating weights checkpoint at epoch {epoch}\n")
            self._model.save_rewind_weights()

    def _batch_iteration(self,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         train: bool = True,
                         targeted=False,
                         exclude_wrong_predictions=False,):
        """ one iteration of forward-backward """

        if not self.rewind_happened:
            return super()._batch_iteration(x, y, train, targeted, exclude_wrong_predictions)
        else:
            return super(AdversarialTrainer, self)._batch_iteration(x, y, train)
