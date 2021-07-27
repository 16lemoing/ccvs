from abc import ABCMeta
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_gan_loss(opt):
    if opt.gan_loss == "original":
        return OriginalGANLoss
    elif opt.gan_loss == "hinge":
        return GANHingeLoss
    elif opt.gan_loss == "logistic":
        return GANLogisticLoss
    elif opt.gan_loss == "wgan":
        return ImprovedWGANLoss
    else:
        raise ValueError

class GANLoss(metaclass=ABCMeta):
    def __init__(self, discriminator):
        self.discriminator = discriminator

    @abstractmethod
    def forward(self, x_real, x_fake):
        pass

    @abstractmethod
    def generator_loss(self, x_fake):
        pass

    @abstractmethod
    def discriminator_loss(self, x_real, x_fake):
        pass

    @abstractmethod
    def _generator_loss_logits(self, d_fake):
        pass

    @abstractmethod
    def _discriminator_loss_logits(self, d_real, d_fake, x_real, x_fake, forward):
        pass

    def generator_loss_logits(self, d_fake):
        if isinstance(d_fake, list):
            return torch.stack([self._generator_loss_logits(d_f) for d_f in d_fake]).mean()
        return self._generator_loss_logits(d_fake)

    def discriminator_loss_logits(self, d_real, d_fake, x_real=None, x_fake=None, forward=None):
        if isinstance(d_real, list):
            return torch.stack([self._discriminator_loss_logits(d_r, d_f, x_real, x_fake, f) for d_r, d_f, f in zip(d_real, d_fake, forward)]).mean()
        return self._discriminator_loss_logits(d_real, d_fake, x_real, x_fake, forward)


class OriginalGANLoss(GANLoss):
    def __init__(self, discriminator):
        self.discriminator = discriminator
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x_real, x_fake):
        d_real = self.discriminator(x_real)["score"]
        d_fake = self.discriminator(x_fake)["score"]
        gen_loss = self.generator_loss_logits(d_fake)
        dis_loss = self.discriminator_loss_logits(d_real, d_fake)
        return gen_loss, dis_loss

    __call__ = forward

    def generator_loss(self, x_fake):
        d_fake = self.discriminator(x_fake)["score"]
        return self.generator_loss_logits(d_fake)

    def _generator_loss_logits(self, d_fake):
        ones = torch.ones_like(d_fake)
        return self.criterion(d_fake, ones)

    def discriminator_loss(self, x_real, x_fake):
        d_real = self.discriminator(x_real)["score"]
        d_fake = self.discriminator(x_fake)["score"]
        return self.discriminator_loss_logits(d_real, d_fake)

    def _discriminator_loss_logits(self, d_real, d_fake, x_real=None, x_fake=None, forward=None):
        ones = torch.ones_like(d_real)
        zeros = torch.zeros_like(d_fake)
        real_loss = self.criterion(d_real, ones)
        fake_loss = self.criterion(d_fake, zeros)
        return (real_loss + fake_loss) / 2


class GANHingeLoss(GANLoss):
    def __init__(self, discriminator):
        self.discriminator = discriminator

    def forward(self, x_real, x_fake):
        d_real = self.discriminator(x_real)["score"]
        d_fake = self.discriminator(x_fake)["score"]
        gen_loss = self.generator_loss_logits(d_fake)
        dis_loss = self.discriminator_loss_logits(d_real, d_fake)
        return gen_loss, dis_loss

    __call__ = forward

    def generator_loss(self, x_fake):
        d_fake = self.discriminator(x_fake)["score"]
        return self.generator_loss_logits(d_fake)

    def _generator_loss_logits(self, d_fake):
        return -d_fake.mean()

    def discriminator_loss(self, x_real, x_fake):
        d_real = self.discriminator(x_real)["score"]
        d_fake = self.discriminator(x_fake)["score"]
        return self.discriminator_loss_logits(d_real, d_fake)

    def _discriminator_loss_logits(self, d_real, d_fake, x_real=None, x_fake=None, forward=None):
        real_loss = F.relu(1 - d_real).mean()
        fake_loss = F.relu(1 + d_fake).mean()
        return (real_loss + fake_loss) / 2


class ImprovedWGANLoss(GANLoss):
    def __init__(self, discriminator, lambda_=10.0):
        self.discriminator = discriminator
        self.lambda_ = lambda_

    def gradient_penalty(self, x_real, x_fake, forward=None):
        if forward is None:
            forward = self.discriminator
        n = x_real.size(0)
        device = x_real.device

        alpha = torch.rand(n)
        alpha = alpha.to(device)
        if len(x_real.shape) == 4:
            alpha = alpha[:, None, None, None]
        else:
            alpha = alpha[:, None, None, None, None]

        interpolates = alpha * x_real.detach() + (1 - alpha) * x_fake.detach()
        interpolates.requires_grad = True
        dis_interpolates = forward(interpolates)

        grad_outputs = torch.ones_like(dis_interpolates).to(device)
        grad = torch.autograd.grad(outputs=dis_interpolates,
                                   inputs=interpolates,
                                   grad_outputs=grad_outputs,
                                   create_graph=True,
                                   retain_graph=True,
                                   only_inputs=True)[0]
        grad = grad.contiguous().view(grad.size(0), -1)

        penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
        return penalty

    def forward(self, x_real, x_fake):
        d_real = self.discriminator(x_real)["score"]
        d_fake = self.discriminator(x_fake)["score"]
        gen_loss = self.generator_loss_logits(d_fake)
        dis_loss = self.discriminator_loss_logits(x_real, x_fake, d_real, d_fake)
        return gen_loss, dis_loss

    def generator_loss(self, x_fake):
        d_fake = self.discriminator(x_fake)["score"]
        return self.generator_loss_logits(d_fake)

    def _generator_loss_logits(self, d_fake):
        return -d_fake.mean()

    def discriminator_loss(self, x_real, x_fake):
        d_real = self.discriminator(x_real)["score"]
        d_fake = self.discriminator(x_fake)["score"]
        return self.discriminator_loss_logits(x_real, x_fake, d_real, d_fake)

    def _discriminator_loss_logits(self, d_real, d_fake, x_real, x_fake, forward=None):
        grad_penalty = self.gradient_penalty(x_real, x_fake, forward=forward)
        return d_fake.mean() - d_real.mean() + self.lambda_ * grad_penalty

    __call__ = forward


class GANLogisticLoss(GANLoss):
    def __init__(self, discriminator):
        self.discriminator = discriminator

    def forward(self, x_real, x_fake):
        d_real = self.discriminator(x_real)["score"]
        d_fake = self.discriminator(x_fake)["score"]
        gen_loss = self.generator_loss_logits(d_fake)
        dis_loss = self.discriminator_loss_logits(d_real, d_fake)
        return gen_loss, dis_loss

    __call__ = forward

    def generator_loss(self, x_fake):
        d_fake = self.discriminator(x_fake)["score"]
        return self.generator_loss_logits(d_fake)

    def _generator_loss_logits(self, d_fake):
        return F.softplus(-d_fake).mean()

    def discriminator_loss(self, x_real, x_fake):
        d_real = self.discriminator(x_real)["score"]
        d_fake = self.discriminator(x_fake)["score"]
        return self.discriminator_loss_logits(d_real, d_fake)

    def _discriminator_loss_logits(self, d_real, d_fake, x_real=None, x_fake=None, forward=None):
        real_loss = F.softplus(-d_real)
        fake_loss = F.softplus(d_fake)
        return real_loss.mean() + fake_loss.mean()

    def discriminator_loss_logits_fake(self, d_fake):
        return F.softplus(d_fake).mean()

    def discriminator_loss_logits_real(self, d_real):
        return F.softplus(-d_real).mean()

    def generator_loss_logits_real(self, d_real):
        return F.softplus(d_real).mean()




