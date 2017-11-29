import visdom
import torch

class Monitor(object):	

	def __init__(self, train=False, spec=''):
		self.vis   = visdom.Visdom()
		self.train = train
		self.spec  = spec
		if self.train:
			self.loss_window = self.vis.line(
								X=torch.zeros((1,)).cpu(),
							   	Y=torch.zeros((1)).cpu(),
							   	opts = dict(xlabel='episode',
											ylabel='mle loss',
											title='Training Loss'+spec,
											legend=['Loss']))

		self.value_window = None 		
		self.text_window = None

	
	def update(self, eps, tot_reward, Q_max, Q_min, loss=None):
		if self.train:
			self.vis.line(
				X=torch.Tensor([eps]).cpu(),
				Y=torch.Tensor([loss]).cpu(),
				win=self.loss_window,
				update='append')

		if self.value_window == None:
			self.value_window = self.vis.line(  X=torch.Tensor([eps]).cpu(),
						Y=torch.Tensor([tot_reward, Q_max, Q_min]).unsqueeze(0).cpu(),
						opts = dict(xlabel='episode',
							ylabel='scalarized Q value',
							title='Value Dynamics'+self.spec,
							legend=['Total Reward', 'Q_max', 'Q_min']))
		else:
			self.vis.line(
				X=torch.Tensor([eps]).cpu(),
				Y=torch.Tensor([tot_reward, Q_max, Q_min]).unsqueeze(0).cpu(),
				win=self.value_window,
				update='append')


	def text(self, tt):
		if self.text_window == None:
			self.text_window = self.vis.text("QPath"+self.spec)
		self.vis.text(
			tt,
			win = self.text_window,
			append=True)
		
