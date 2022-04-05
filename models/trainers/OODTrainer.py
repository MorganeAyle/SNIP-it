import torch
import foolbox as fb
from torch.distributions import Categorical

from models.trainers.DefaultTrainer import DefaultTrainer
from utils.attacks_utils import construct_adversarial_examples
from utils.model_utils import find_right_model
from utils.constants import LOSS_DIR

import argparse
import pickle
import sys
import time

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.distributions import Categorical

from models import GeneralModel
from models.statistics import Metrics
from models.statistics.Flops import FLOPCounter
from models.statistics.Saliency import Saliency
from utils.model_utils import find_right_model
from utils.system_utils import *
from utils.attacks_utils import construct_adversarial_examples
from utils.metrics import calculate_aupr, calculate_auroc

from utils.cka_utils import cka_batch


class OODTrainer(DefaultTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._ood_loss_function = find_right_model(
            LOSS_DIR, 'CrossEntropyKL',
            device=self._arguments['device'],
            l1_reg=self._arguments['l1_reg'],
            lp_reg=self._arguments['lp_reg'],
            l0_reg=self._arguments['l0_reg'],
            hoyer_reg=self._arguments['hoyer_reg']
        )
        self.add_kl = False

    def _batch_iteration_ood(self,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         ood_x: torch.Tensor,
                         ood_y: torch.Tensor,
                         train: bool = True,
                         return_acc: bool = True):
        """ one iteration of forward-backward """

        # unpack
        x, y = x.to(self._device).float(), y.to(self._device)
        ood_x, ood_y = ood_x.to(self._device).float(), ood_y.to(self._device)

        # update metrics
        self._metrics.update_batch(train)

        # record time
        if "cuda" in str(self._device):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        # forward pass
        if return_acc:
            accuracy, loss, out = self._forward_pass_ood(x, y, ood_x, ood_y, train=train, return_acc=return_acc)
        else:
            loss, out = self._forward_pass_ood(x, y, ood_x, ood_y, train=train, return_acc=return_acc)

        if self._arguments['prune_criterion'] == 'RigL':
            self._handle_pruning(self._metrics._epoch)

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

    def _forward_pass_ood(self,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      ood_x: torch.Tensor,
                      ood_y: torch.Tensor,
                      train: bool = True,
                      return_acc: bool = True):
        """ implementation of a forward pass """

        if train:
            self._optimizer.zero_grad()
            if self._model.is_maskable:
                self._model.apply_weight_mask()

        out = self._model(x).squeeze()
        ood_out = self._model(ood_x).squeeze()
        loss = self._ood_loss_function(
            output=out,
            target=y,
            weight_generator=self._model.parameters(),
            model=self._model,
            criterion=self._criterion,
            ood_output=ood_out,
            add_kl=self.add_kl
        )
        if return_acc:
            accuracy = self._get_accuracy(out, y)
            return accuracy, loss, out
        else:
            return loss, out

    def _epoch_iteration(self):
        """ implementation of an epoch """

        self.out("\n")

        self._acc_buffer, self._loss_buffer = self._metrics.update_epoch()
        self._entropy_buffer = []

        for batch_num, (batch, ood_batch) in enumerate(zip(self._train_loader, self._ood_prune_loader)):
            self.out(f"\rTraining... {batch_num}/{len(self._train_loader)}", end='')

            if self._model.is_tracking_weights:
                self._model.save_prev_weights()

            acc, loss, elapsed, entropy, preds = self._batch_iteration_ood(*batch, *ood_batch, self._model.training)

            if self._model.is_tracking_weights:
                self._model.update_tracked_weights(self._metrics.batch_train)

            self._acc_buffer.append(acc)
            self._loss_buffer.append(loss)
            self._elapsed_buffer.append(elapsed)
            self._entropy_buffer.append(entropy)

            self._log(batch_num)

            self._check_exit_conditions_epoch_iteration()

        self.out("\n")

    def validate(self):
        """ validates the model on test set """

        self.out("\n")

        # init test mode
        self._model.eval()
        cum_acc, cum_loss, cum_elapsed, cum_entropy, ood_cum_entropy, success_rates = [], [], [], [], [], []

        ood_true = np.zeros(0)
        ood_preds = np.zeros(0)

        with torch.no_grad():
            for batch_num, batch in enumerate(self._test_loader):
                acc, loss, elapsed, entropy, preds = self._batch_iteration(*batch, self._model.training)
                cum_acc.append(acc)
                cum_loss.append(loss),
                cum_elapsed.append(elapsed)
                cum_entropy.append(entropy)
                ood_true = np.concatenate((ood_true, np.ones(len(preds))))
                ood_preds = np.concatenate((ood_preds, preds.reshape((-1))))
                self.out(f"\rEvaluating... {batch_num}/{len(self._test_loader)}", end='')
        self.out("\n")

        # validate on OOD data
        with torch.no_grad():
            for batch_num, batch in enumerate(self._ood_loader):
                _, _, entropy, preds = self._batch_iteration(*batch, self._model.training, return_acc=False)
                ood_cum_entropy.append(entropy)
                ood_true = np.concatenate((ood_true, np.zeros(len(preds))))
                ood_preds = np.concatenate((ood_preds, preds.reshape((-1))))

        # validate on adversarial attacks
        for batch_num, batch in enumerate(self._test_loader):
            im, crit = batch
            adv_results, predictions = construct_adversarial_examples(im, crit, self._arguments['attack'], self._model,
                                                                      self._device, self._arguments['epsilon'])
            _, advs, success = adv_results
            attack_success = (success.float().sum() / len(success)).item()
            success_rates.append(attack_success)

        # put back into train mode
        self._model.train()

        return float(np.mean(cum_acc)), float(np.mean(cum_loss)), cum_elapsed, float(np.mean(cum_entropy)), float(
            np.mean(ood_cum_entropy)), float(np.mean(success_rates)), ood_preds, ood_true

    def train(self):
        """ main training function """
        from utils.constants import RESULTS_DIR

        # setup data output directories:
        setup_directories()
        save_codebase_of_run(self._arguments)
        DATA_MANAGER.write_to_file(
            os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, OUTPUT_DIR, "calling_command.txt"), str(" ".join(sys.argv)))

        # data gathering
        epoch = self._metrics._epoch

        self._model.train()

        try:

            self.out(
                f"{PRINTCOLOR_BOLD}Started training{PRINTCOLOR_END}"
            )

            if "Early" in self._arguments['prune_criterion']:
                # for i in range(10):
                # self._metrics.handle_weight_plotting(epoch, trainer_ns=self)
                while self._stable == False:
                    self.out("Network has not reached stable state")
                    self.out(f"\n\n{PRINTCOLOR_BOLD}EPOCH {epoch} {PRINTCOLOR_END} \n\n")
                    # do epoch
                    self._epoch_iteration()

                    # if calculate_aupr(self.ood_true, self.ood_preds) >= 0.78:
                    #     self._stable = True

                    if epoch == self._arguments['prune_to']:
                        self._stable = True
                        self.add_kl = False
                    if epoch == 60:
                        self.add_kl = True
                    epoch += 1

            if self._arguments['skip_first_plot']:
                self._metrics.handle_weight_plotting(0, trainer_ns=self)

            # if prune before training
            if self._arguments['prune_criterion'] in SINGLE_SHOT:
                self._criterion.prune(self._arguments['pruning_limit'],
                                      train_loader=self._train_loader,
                                      ood_loader=self._ood_prune_loader,
                                      local=self._arguments['local_pruning'],
                                      manager=DATA_MANAGER,)
                if self._model.is_rewindable:
                    self.out("rewinding weights to checkpoint...\n")
                    self._model.do_rewind()
                # If structured, probably needs to re-initialize optimizer with new architecture
                if self._arguments['prune_criterion'] in STRUCTURED_SINGLE_SHOT:
                    self._optimizer = find_right_model(OPTIMS, self._arguments['optimizer'],
                                                       params=self._model.parameters(),
                                                       lr=self._arguments['learning_rate'],
                                                       weight_decay=self._arguments['l2_reg'])
                    self._metrics.model_to_tensorboard(self._model, timestep=epoch)

            if self._arguments['prune_criterion'] == 'RigL':
                # TODO do random pruning
                pass

            # do training
            for epoch in range(epoch, self._arguments['epochs'] + epoch):
                self.out(f"\n\n{PRINTCOLOR_BOLD}EPOCH {epoch} {PRINTCOLOR_END} \n\n")

                # do epoch
                self._epoch_iteration()

                # plotting
                if (epoch % self._arguments['plot_weights_freq']) == 0 and self._arguments['plot_weights_freq'] > 0:
                    self._metrics.handle_weight_plotting(epoch, trainer_ns=self)

                # do all related to pruning
                self._handle_pruning(epoch)

                # save what needs to be saved
                self._handle_backing_up(epoch)

                if epoch == self._arguments['epochs'] - 1:
                    self._model.zero_grad()
                    self._model.eval()
                    import copy
                    self._test_model = copy.deepcopy(self._model)
                    self._test_model.add_hooks()
                    for batch_num, batch in enumerate(self._train_loader):
                        self._test_model(batch[0].to(self._device))
                        # break
                    activations1 = []
                    for value in self._test_model.hooks.values():
                        activations1.append(value)

                    self._test_model = copy.deepcopy(self._model)
                    self._test_model.add_hooks()
                    for batch_num, batch in enumerate(self._ood_loader):
                        self._test_model(batch[0].to(self._device))
                        # break
                    activations2 = []
                    for value in self._test_model.hooks.values():
                        activations2.append(value)

                    cka_distances = np.zeros(len(activations1))
                    import math
                    for j in range(len(activations1)):
                        cka_distances[j] = cka_batch(activations1[j], activations2[j])
                    for cka, layer_name in zip(cka_distances, self._model.mask.keys()):
                        self._metrics.add(cka, key="cka/layer" + '_' + layer_name)
                        print(layer_name, self._model.mask[layer_name].sum() / torch.numel(self._model.mask[layer_name]))
                    self._metrics.add(np.mean(cka_distances), key="criterion/cka")
                    self.cka_mean = np.mean(cka_distances)

                    self._model.train()

            if self._arguments['skip_first_plot']:
                self._metrics.handle_weight_plotting(epoch + 1, trainer_ns=self)

            # example last save
            save_models([self._model, self._metrics], "finished")

        except KeyboardInterrupt as e:
            self.out(f"Killed by user: {e} at {time.time()}")
            save_models([self._model, self._metrics], f"KILLED_at_epoch_{epoch}")
            sys.stdout.flush()
            DATA_MANAGER.write_to_file(
                os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, OUTPUT_DIR, "log.txt"), self._metrics.log)
            self._writer.close()
            exit(69)
        except Exception as e:
            self._writer.close()
            report_error(e, self._model, epoch, self._metrics)

        # flush prints
        sys.stdout.flush()
        DATA_MANAGER.write_to_file(
            os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, OUTPUT_DIR, "log.txt"), self._metrics.log)
        self._writer.close()