import visdom
import torch

class Monitor(object):	

	def __init__(self):
		self.vis = visdom.Visdom()
		self.loss_window = self.vis.line(
								X=torch.zeros((1,)).cpu(),
							   	Y=torch.zeros((1)).cpu(),
							   	opts = dict(xlabel='episode',
											ylabel='mle loss',
											title='Training Loss',
											legend=['Loss']))

		self.value_window = self.vis.line(
								X=torch.zeros((1,)).cpu(),
							   	Y=torch.zeros((1, 3)).cpu(),
							   	opts = dict(xlabel='episode',
											ylabel='scalarized Q value',
											title='Value Dynamics',
											legend=['Total Reward', 'Q_max', 'Q_min']))

	
	def update(self, eps, tot_reward, Q_max, Q_min, loss):
		self.vis.line(
				X=torch.Tensor([eps]).cpu(),
				Y=torch.Tensor([loss]).unsqueeze(0).cpu(),
				win=self.loss_window,
				update='append')

		self.vis.line(
				X=torch.Tensor([eps]).cpu(),
				Y=torch.Tensor([tot_reward, Q_max, Q_min]).unsqueeze(0).cpu(),
				win=self.value_window,
				update='append')
		
