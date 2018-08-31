#coding=utf-8

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
	'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexNet(nn.Module):

	def __init__(self, num_classes=1000):
		super(AlexNet, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 96, 11, stride=4),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(3, stride=2),
			nn.LocalResponseNorm(5),
			nn.Conv2d(96, 256, 5, stride=1, padding=2, groups=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(3, stride=2),
			nn.LocalResponseNorm(5),
		)