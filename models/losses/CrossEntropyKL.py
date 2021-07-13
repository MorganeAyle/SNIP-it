import torch
from torch import nn
import torch.nn.functional as F

from models.GeneralModel import GeneralModel


class CrossEntropyKL(GeneralModel):

    def __init__(self, device, l1_reg=0, lp_reg=0, **kwargs):
        super(CrossEntropyKL, self).__init__(device, **kwargs)

        self.l1_reg = torch.tensor(l1_reg).to(device)
        self.lp_reg = torch.tensor(lp_reg).to(device)

        self.loss = nn.CrossEntropyLoss()
        self.ood_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, output=None, target=None, weight_generator=None, ood_output=None, **kwargs):
        regularisation = torch.zeros([1]).to(self.device)
        if self._do_l1 or self._do_lp:
            for param in weight_generator:
                if self._do_l1:
                    regularisation += self.l1_reg.__mul__(torch.norm((param + 1e-8), p=1))
                if self._do_lp:
                    regularisation += self.lp_reg.__mul__(torch.norm((param + 1e-8), p=0.2))

        return self.loss.forward(output, target) + 0.5*self.ood_loss.forward(torch.log(F.softmax(ood_output, -1)), torch.ones_like(ood_output).float()/10) + regularisation

    @property
    def _do_l1(self):
        return self.l1_reg > 0

    @property
    def _do_lp(self):
        return self.lp_reg > 0
