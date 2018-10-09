import visdom
import torch


class Monitor(object):

    def __init__(self, spec=''):
        self.vis = visdom.Visdom()
        self.spec = spec
        self.loss_window = self.vis.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1)).cpu(),
            opts=dict(xlabel='episode',
                      ylabel='mle loss',
                      title='Training Loss' + spec,
                      legend=['Loss']))

    def update(self, eps, loss=None):
        self.vis.line(
            X=torch.Tensor([eps]).cpu(),
            Y=torch.Tensor([loss]).cpu(),
            win=self.loss_window,
            update='append')
