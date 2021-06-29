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


class DefaultTrainer:
    """
    Implements generalised computer vision classification with pruning
    """

    def __init__(self,
                 model: GeneralModel,
                 loss: GeneralModel,
                 optimizer: Optimizer,
                 device,
                 arguments,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 ood_loader: DataLoader,
                 ood_prune_loader: DataLoader,
                 metrics: Metrics,
                 criterion: GeneralModel,
                 run_name
                 ):

        self._test_loader = test_loader
        self._train_loader = train_loader
        self._ood_loader = ood_loader
        self._ood_prune_loader = ood_prune_loader
        self._loss_function = loss
        self._model = model
        self._arguments = arguments
        self._optimizer = optimizer
        self._device = device
        self._global_steps = 0
        self.out = metrics.log_line
        DATA_MANAGER.set_date_stamp(addition=run_name)
        from utils.constants import RESULTS_DIR
        self._writer = SummaryWriter(os.path.join(DATA_MANAGER.directory, RESULTS_DIR, DATA_MANAGER.stamp, SUMMARY_DIR))
        self._metrics: Metrics = metrics
        self._metrics.init_training(self._writer)
        self._acc_buffer = []
        self._loss_buffer = []
        self._entropy_buffer = []
        self._elapsed_buffer = []
        self._criterion = criterion

        self.ts = None

        ## John
        self._stable = False
        ##

        batch = next(iter(self._test_loader))
        self.saliency = Saliency(model, device, batch[0][:8])
        self._metrics.write_arguments(arguments)
        self._flopcounter = FLOPCounter(model, batch[0][:8], self._arguments['batch_size'], device=device)
        self._metrics.model_to_tensorboard(model, timestep=-1)

    def _batch_iteration(self,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         train: bool = True,
                         return_acc: bool = True):
        """ one iteration of forward-backward """

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

    def _forward_pass(self,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      train: bool = True,
                      return_acc: bool = True):
        """ implementation of a forward pass """

        if train:
            self._optimizer.zero_grad()
            if self._model.is_maskable:
                self._model.apply_weight_mask()

        out = self._model(x).squeeze()
        loss = self._loss_function(
            output=out,
            target=y,
            weight_generator=self._model.parameters(),
            model=self._model,
            criterion=self._criterion
        )
        if return_acc:
            accuracy = self._get_accuracy(out, y)
            return accuracy, loss, out
        else:
            return loss, out

    def _backward_pass(self, loss):
        """ implementation of a backward pass """

        loss.backward()
        # layer_count = 0
        # for name, layer in self._model.named_modules():
        #     if "Norm" in str(layer): continue
        #     if name + ".weight" in self._model.mask:
        #         self._writer.add_scalar("grad_l1_norm/" + "layer" + str(layer_count) + "_" + name, layer.weight.grad.norm(1), self._metrics.batch_train)
        #         self._writer.add_scalar("weight_l1_norm/" + "layer" + str(layer_count) + "_" + name, layer.weight.norm(1), self._metrics.batch_train)
        #         layer_count += 1
        self._model.insert_noise_for_gradient(self._arguments['grad_noise'])
        if self._arguments['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._arguments['grad_clip'])
        self._optimizer.step()
        if self._model.is_maskable:
            self._model.apply_weight_mask()

    def _epoch_iteration(self):
        """ implementation of an epoch """

        self.out("\n")

        self._acc_buffer, self._loss_buffer = self._metrics.update_epoch()
        self._entropy_buffer = []

        for batch_num, batch in enumerate(self._train_loader):
            self.out(f"\rTraining... {batch_num}/{len(self._train_loader)}", end='')

            if self._model.is_tracking_weights:
                self._model.save_prev_weights()

            acc, loss, elapsed, entropy, preds = self._batch_iteration(*batch, self._model.training)

            if self._model.is_tracking_weights:
                self._model.update_tracked_weights(self._metrics.batch_train)

            self._acc_buffer.append(acc)
            self._loss_buffer.append(loss)
            self._elapsed_buffer.append(elapsed)
            self._entropy_buffer.append(entropy)

            self._log(batch_num)

            self._check_exit_conditions_epoch_iteration()

        self.out("\n")

    def _log(self, batch_num: int):
        """ logs to terminal and tensorboard if the time is right"""

        if (batch_num % self._arguments['eval_freq']) == 0:
            # validate on test and train set
            self.train_acc, train_loss, self.train_entropy = np.mean(self._acc_buffer), np.mean(
                self._loss_buffer), np.mean(self._entropy_buffer)
            self.test_acc, test_loss, test_elapsed, self.test_entropy, self.ood_entropy, self.success_rate, ood_preds, ood_true = self.validate()
            self._elapsed_buffer += test_elapsed

            # log metrics
            self._add_metrics(self.test_acc, test_loss, self.train_acc, train_loss, self.train_entropy,
                              self.test_entropy, self.ood_entropy,
                              self.success_rate, ood_preds, ood_true)

            # reset for next log
            self._acc_buffer, self._loss_buffer, self._elapsed_buffer, self._entropy_buffer = [], [], [], []

            # print to terminal
            self.out(self._metrics.printable_last)

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

        self._model.eval()
        # validate on OOD data
        with torch.no_grad():
            for batch_num, batch in enumerate(self._ood_loader):
                _, _, entropy, preds = self._batch_iteration(*batch, self._model.training, return_acc=False)
                ood_cum_entropy.append(entropy)
                ood_true = np.concatenate((ood_true, np.zeros(len(preds))))
                ood_preds = np.concatenate((ood_preds, preds.reshape((-1))))

        self._model.eval()
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

    def _add_metrics(self, test_acc, test_loss, train_acc, train_loss, train_entropy, test_entropy, ood_entropy,
                     success_rate, ood_preds, ood_true):
        """
        save metrics
        """

        self.sparsity = self._model.pruned_percentage
        spasity_index = 2 * ((self.sparsity * test_acc) / (1e-8 + self.sparsity + test_acc))

        # L0 norm of weights
        l0_norm_weights = 0
        for name, layer in self._model.named_modules():
            if name + ".weight" in self._model.mask:
                large_weights = torch.abs(layer.weight) > 0.01
                l0_norm_weights += torch.nonzero(large_weights).shape[0]

        # AU-ROC
        self.auroc = calculate_auroc(ood_true, ood_preds)
        # AU-PR
        self.aupr = calculate_aupr(ood_true, ood_preds)

        flops_per_sample, total_seen = self._flopcounter.count_flops(self._metrics.batch_train)

        self._metrics.add(train_acc, key="acc/train")
        self._metrics.add(train_loss, key="loss/train")
        self._metrics.add(train_entropy, key="entropy/train")
        self._metrics.add(test_loss, key="loss/test")
        self._metrics.add(test_acc, key="acc/test")
        self._metrics.add(test_entropy, key="entropy/test")
        self._metrics.add(success_rate, key=f"adv_success_rate/test")
        self._metrics.add(ood_entropy, key="entropy/ood")
        self._metrics.add(self.auroc, key="entropy/auroc")
        self._metrics.add(self.aupr, key="entropy/aupr")
        self._metrics.add(l0_norm_weights, key="sparse/l0_norm")
        self._metrics.add(self.sparsity, key="sparse/weight")
        self._metrics.add(self._model.structural_sparsity, key="sparse/node")
        self._metrics.add(spasity_index, key="sparse/hm")
        self._metrics.add(np.log(self._model.compressed_size), key="sparse/log_disk_size")
        self._metrics.add(np.mean(self._elapsed_buffer), key="time/gpu_time")
        self._metrics.add(int(flops_per_sample), key="time/flops_per_sample")
        self._metrics.add(np.log10(total_seen), key="time/flops_log_cum")
        if torch.cuda.is_available():
            self._metrics.add(torch.cuda.memory_allocated(0), key="cuda/ram_footprint")
        self._metrics.timeit()

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
                    if epoch == self._arguments['prune_to']:
                        self._stable = True
                    epoch += 1

            if self._arguments['skip_first_plot']:
                self._metrics.handle_weight_plotting(0, trainer_ns=self)

            # if prune before training
            if self._arguments['prune_criterion'] in SINGLE_SHOT:
                self._criterion.prune(self._arguments['pruning_limit'],
                                      train_loader=self._train_loader,
                                      ood_loader=self._ood_prune_loader,
                                      local=self._arguments['local_pruning'],
                                      manager=DATA_MANAGER)
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

            # with open('/nfs/homedirs/ayle/model_conv6_0.5.pickle', 'wb') as f:
            #     pickle.dump(self._model, f)

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

                if self._arguments['get_hooks']:
                    if epoch % 5 == 0:
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

    def _handle_backing_up(self, epoch):
        from utils.constants import RESULTS_DIR
        if (epoch % self._arguments['save_freq']) == 0 and epoch > 0:
            self.out("\nSAVING...\n")
            save_models(
                [self._model, self._metrics],
                f"save_at_epoch_{epoch}"
            )
        sys.stdout.flush()
        DATA_MANAGER.write_to_file(
            os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, OUTPUT_DIR, "log.txt"),
            self._metrics.log
        )

    def _handle_pruning(self, epoch):
        if self._is_pruning_time(epoch, self._metrics.batch_train):
            self.out("\nIS PRUNING TIME...\n")
            if self._is_not_finished_pruning():
                self.out("\nPRUNING...\n")
                if self._arguments['prune_criterion'] == 'HYDRA':
                    self._criterion.prune(self._arguments['pruning_limit'])
                    self._log(1000)
                else:
                    self._criterion.prune(
                        # percentage overwritten in SNIPitDuring, IMP by the step, not used in HoyerSquare
                        percentage=self._arguments['pruning_rate'],  # I think GroupHoyerSquare uses it, EfficientConvNets
                        train_loader=self._train_loader,
                        manager=DATA_MANAGER,
                        ood_loader=self._ood_loader,
                    )
                    # DURING_TRAINING actually contains only structured ones, need to re-initialize optimizer
                    if self._arguments['prune_criterion'] in DURING_TRAINING:
                        self._optimizer = find_right_model(
                            OPTIMS, self._arguments['optimizer'],
                            params=self._model.parameters(),
                            lr=self._arguments['learning_rate'],
                            weight_decay=self._arguments['l2_reg']
                        )
                        self._metrics.model_to_tensorboard(self._model, timestep=epoch)
                    if self._model.is_rewindable:
                        self.out("rewinding weights to checkpoint...\n")
                        self._model.do_rewind()
            if self._model.is_growable:
                self.out("growing too...\n")
                self._criterion.grow(self._arguments['growing_rate'])

        if self._is_checkpoint_time(epoch):
            self.out(f"\nCreating weights checkpoint at epoch {epoch}\n")
            self._model.save_rewind_weights()

    def _is_not_finished_pruning(self):

        print(self._arguments['pruning_limit'] > (self._model.pruned_percentage + 0.01))

        return self._arguments['pruning_limit'] > (self._model.pruned_percentage + 0.01) \
               or \
               (
                       self._arguments['prune_criterion'] in DURING_TRAINING
                       and
                       self._arguments['pruning_limit'] > self._model.structural_sparsity
               ) or  \
               (
                       self._arguments['prune_criterion'] == 'RigL'
               )

    @staticmethod
    def _get_accuracy(output, y):
        predictions = output.argmax(dim=-1, keepdim=True).view_as(y)
        correct = y.eq(predictions).sum().item()
        return correct / output.shape[0]

    def _is_checkpoint_time(self, epoch: int):
        return epoch == self._arguments['rewind_to'] and self._model.is_rewindable

    def _is_pruning_time(self, epoch: int, iteration: int):
        if self._arguments['prune_criterion'] == "EmptyCrit":
            return False
        if self._arguments['prune_criterion'] == 'RigL':
            return (iteration % self._arguments['delta_t']) == 0 and \
                   iteration < self._arguments['t_end']

        epoch -= self._arguments['prune_delay']
        return (epoch % self._arguments['prune_freq']) == 0 and \
               epoch >= 0 and \
               self._model.is_maskable and \
               self._arguments['prune_criterion'] not in SINGLE_SHOT

    def _check_exit_conditions_epoch_iteration(self, patience=1):

        time_passed = datetime.now() - DATA_MANAGER.actual_date
        # check if runtime is expired
        if (time_passed.total_seconds() > (self._arguments['max_training_minutes'] * 60)) \
                and \
                self._arguments['max_training_minutes'] > 0:
            raise KeyboardInterrupt(
                f"Process killed because {self._arguments['max_training_minutes']} minutes passed "
                f"since {DATA_MANAGER.actual_date}. Time now is {datetime.now()}")
        if patience == 0:
            raise NotImplementedError("feature to implement",
                                      KeyboardInterrupt("Process killed because patience is zero"))
