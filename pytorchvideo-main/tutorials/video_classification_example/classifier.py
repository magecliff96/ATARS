import torch
import torch.nn as nn
import time
class Head(nn.Module):
	def __init__(self, in_channel, num_actor_classes):
		super(Head, self).__init__()
		self.fc_actor = nn.Sequential(
			nn.ReLU(inplace=False),
			nn.Linear(in_channel, num_actor_classes)
			)
		
	def forward(self, x):
		y_actor = self.fc_actor(x)
		return y_actor

class Instance_Head(nn.Module):
	def __init__(self, in_channel, num_actor_classes):
		super(Instance_Head, self).__init__()
		self.num_actor_classes = num_actor_classes
		self.fc_actor = nn.ModuleList()
		for i in range(num_actor_classes):
			self.fc_actor.append(nn.Sequential(
	                nn.ReLU(inplace=False),
	                nn.Linear(in_channel, 1),
	                )
				)

	def forward(self, x):
		b, n, _ = x.shape
		y_actor = []
		for i in range(self.num_actor_classes):
			y_actor.append(self.fc_actor[i](x[:, i, :]))
		y_actor = torch.stack(y_actor, dim=0)
		y_actor = y_actor.permute((1, 0, 2))
		y_actor = torch.reshape(y_actor, (b, n))
		# x = torch.reshape(x, (b, n))
			
		return y_actor


class Allocated_Head(nn.Module):
	def __init__(self, in_channel, num_actor_classes):
		super(Allocated_Head, self).__init__()
		self.num_actor_classes = num_actor_classes
		self.fc_actor = nn.ModuleList()
		for i in range(num_actor_classes):
			self.fc_actor.append(nn.Sequential(
	                nn.ReLU(inplace=False),
	                nn.Linear(in_channel, 1),
	                )
				)

	def forward(self, x):
		b, n, _ = x.shape
		y_actor = []
		for i in range(self.num_actor_classes):
			y_actor.append(self.fc_actor[i](x[:, i, :]))
		y_actor = torch.stack(y_actor, dim=0)
		y_actor = y_actor.permute((1, 0, 2))
		y_actor = torch.reshape(y_actor, (b, n))
		# x = torch.reshape(x, (b, n))
			
		return y_actor
