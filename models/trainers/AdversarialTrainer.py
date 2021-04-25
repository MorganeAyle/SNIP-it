import torch
import torch.nn.functional as F
import foolbox as fb

from models.trainers.DefaultTrainer import DefaultTrainer
from utils.attacks_utils import construct_adversarial_examples

class AdversarialTrainer(DefaultTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack_method = self._arguments['attack']
        self.epsilons = self._arguments['epsilons']

    def _batch_iteration(self,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         train: bool = True,
                         targeted=False,
                         exclude_wrong_predictions=False,):
        """ one iteration of forward-backward """

        # attack stuff
        if train:
            adv_batch_size = int(self._arguments['batch_size'] / 2)
            x_to_adv = x[:adv_batch_size]
            y_to_adv = y[:adv_batch_size]
            self._model.eval()
            if self._model.is_maskable:
                self._model.apply_weight_mask()
            adv_results, _ = construct_adversarial_examples(x_to_adv, y_to_adv, self.attack_method, self._model, self._device, exclude_wrong_predictions, targeted, self.epsilons)
            _, advs, _ = adv_results
            x = torch.cat((torch.cat(advs).cpu(), x[adv_batch_size:]))

            self._model.train()

        # unpack
        x, y = x.to(self._device).float(), y.to(self._device)

        # update metrics
        self._metrics.update_batch(train)

        # record time
        if "cuda" in str(self._device):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        # forward pass
        accuracy, loss, out = self._forward_pass(x, y, train=train)

        # compute entropy
        prob = F.softmax(out, dim=1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-8)) / x.shape[0]

        # backward pass
        if train:
            self._backward_pass(loss)

        # record time
        if "cuda" in str(self._device):
            end.record()
            torch.cuda.synchronize(self._device)
            time = start.elapsed_time(end)
        else:
            time = 0

        # free memory
        for tens in [out, y, x, loss, entropy]:
            tens.detach()

        return accuracy, loss.item(), time, entropy.detach().cpu()