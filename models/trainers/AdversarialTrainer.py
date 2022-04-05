import torch
import torch.nn.functional as F
import foolbox as fb
from torch.distributions import Categorical

from models.trainers.DefaultTrainer import DefaultTrainer
from utils.attacks_utils import construct_adversarial_examples

from utils.config_utils import *
from utils.model_utils import *
from utils.system_utils import *

from torchvision import transforms


class AdversarialTrainer(DefaultTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack_method = self._arguments['attack']
        self.epsilon = self._arguments['epsilon']
        # load data
        (self.un_train_loader, self.un_test_loader), mean, std = find_right_model(
            DATASETS, self._arguments['data_set'] + '_unnormalized',
            arguments=self._arguments
        )
        self.transform = transforms.Normalize(mean, std)

    def _batch_iteration(self,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         train: bool = True,
                         targeted=False,
                         exclude_wrong_predictions=False,
                         return_acc: bool = True,
                         **kwargs):
        """ one iteration of forward-backward """

        # attack stuff
        if train:
            adv_batch_size = int(self._arguments['batch_size'] / 2)
            x_to_adv = kwargs['xun'][:adv_batch_size]
            y_to_adv = y[:adv_batch_size]
            self._model.eval()
            if self._model.is_maskable:
                self._model.apply_weight_mask()
            adv_results, _ = construct_adversarial_examples(x_to_adv, y_to_adv, self.attack_method, self._model, self._device, self.epsilon, exclude_wrong_predictions, targeted)
            _, advs, _ = adv_results

            advs = advs.cpu()
            new_advs = []
            for image in advs:
                image = self.transform(image.squeeze()).unsqueeze(0)
                new_advs.append(image)
            advs = torch.cat(new_advs)

            x = torch.cat((advs, x[adv_batch_size:]))

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
        if return_acc:
            accuracy, loss, out = self._forward_pass(x, y, train=train, return_acc=return_acc)
        else:
            loss, out = self._forward_pass(x, y, train=train, return_acc=return_acc)

        # backward pass
        if train:
            self._backward_pass(loss)

        # compute entropy
        probs = F.softmax(out, dim=-1)
        entropy = Categorical(probs).entropy().squeeze().mean()

        # get max predicted prob
        preds, _ = torch.max(probs, dim=-1)

        # record time
        if "cuda" in str(self._device):
            end.record()
            torch.cuda.synchronize(self._device)
            time = start.elapsed_time(end)
        else:
            time = 0

        # free memory
        for tens in [out, y, x, loss, entropy, preds]:
            tens.detach()

        if return_acc:
            return accuracy, loss.item(), time, entropy.detach().cpu(), preds.cpu()
        else:
            return loss.item(), time, entropy.detach().cpu(), preds.cpu()

    def _epoch_iteration(self):
        """ implementation of an epoch """

        self.out("\n")

        self._acc_buffer, self._loss_buffer = self._metrics.update_epoch()
        self._entropy_buffer = []

        for batch_num, batch, un_batch in enumerate(zip(self._train_loader, self.un_train_loader)):
            self.out(f"\rTraining... {batch_num}/{len(self._train_loader)}", end='')

            if self._model.is_tracking_weights:
                self._model.save_prev_weights()

            unx, uny = un_batch

            acc, loss, elapsed, entropy, preds = self._batch_iteration(*batch, self._model.training, unx=unx, uny=uny)

            if self._model.is_tracking_weights:
                self._model.update_tracked_weights(self._metrics.batch_train)

            self._acc_buffer.append(acc)
            self._loss_buffer.append(loss)
            self._elapsed_buffer.append(elapsed)
            self._entropy_buffer.append(entropy)

            self._log(batch_num)

            self._check_exit_conditions_epoch_iteration()

        self.out("\n")
