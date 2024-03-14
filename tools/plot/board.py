import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import subprocess
import time
from .. import FormatConverter


class VisualBoard:
    def __init__(self, optimizer=None, name=None, start_iter=0, new_process=False):
        if new_process:
            subprocess.Popen(['tensorboard', '--logdir=runs'])
        time_str = time.strftime("%m-%d-%H%M%S")
        if name is not None:
            self.writer = SummaryWriter(f'runs/{name}')
        else:
            self.writer = SummaryWriter(f'runs/{time_str}')

        self.iter = start_iter
        self.optimizer = optimizer

        self.init_loss()

    def __call__(self, epoch, iter):
        self.iter = iter
        self.writer.add_scalar('misc/epoch', epoch, self.iter)
        if self.optimizer:
            self.writer.add_scalar('misc/learning_rate', self.optimizer.param_groups[0]["lr"], self.iter)

    def write_scalar(self, scalar, name):
        try:
            scalar = scalar.detach().cpu().numpy()
        except:
            scalar = scalar
        self.writer.add_scalar(name, scalar, self.iter)

    def write_tensor(self, im, name):
        self.writer.add_image('attack/'+name, im.detach().cpu(), self.iter)

    def write_cv2(self, im, name):
        im = FormatConverter.bgr_numpy2tensor(im)[0]
        self.writer.add_image(f'attack/{name}', im, self.iter)
    
    def write_numpy(self, im, name):
        im = torch.from_numpy(im)/255.0
        self.writer.add_image(f'attack/{name}', im, self.iter)

    def write_ep_loss(self, ep_loss):
        for k in self.loss.keys():
            self.writer.add_scalar('loss/{}_det_loss'.format(k), np.array(self.loss[k]).mean(), self.iter)
        for k in self.conf_max.keys():
            self.writer.add_scalar('loss/{}_conf_max'.format(k), np.array(self.conf_max[k]).mean(), self.iter)
        for k in self.conf_min.keys():
            self.writer.add_scalar('loss/{}_conf_min'.format(k), np.array(self.conf_min[k]).mean(), self.iter)
        self.init_loss()

    def init_loss(self):
        self.loss = {}
        self.conf_max = {}
        self.conf_min = {}

    def note_loss(self, loss, conf_max,conf_min, name):
        if name not in self.loss.keys():
            self.loss[name] = []
        if name not in self.conf_max.keys():
            self.conf_max[name] = []
        if name not in self.conf_min.keys():
            self.conf_min[name] = []
        self.loss[name].append(loss.detach().cpu().numpy())
        self.conf_max[name].append(conf_max.detach().cpu().numpy())
        self.conf_min[name].append(conf_min.detach().cpu().numpy())

