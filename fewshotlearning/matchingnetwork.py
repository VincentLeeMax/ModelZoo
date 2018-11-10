#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

class encoder(nn.Module):
	def __init__(self, layer_size, num_channel, keep_rate=1.0):
		"""
		encoder for the feature embedding
		:param layer_size:
		:param num_channel:
		:param keep_rate:
		"""
		super(encoder, self).__init__()
		self.layer_size = layer_size
		self.num_channel = num_channel
		self.keep_rate = keep_rate
		self.layer1 = self._conv(self.num_channel, self.layer_size, 3, self.keep_rate)
		self.layer2 = self._conv(self.num_channel, self.layer_size, 3, self.keep_rate)
		self.layer3 = self._conv(self.num_channel, self.layer_size, 3, self.keep_rate)
		self.layer4 = self._conv(self.num_channel, self.layer_size, 2, self.keep_rate)

		self.weights_init(self.layer1)
		self.weights_init(self.layer2)
		self.weights_init(self.layer3)
		self.weights_init(self.layer4)

	def weights_init(self, module):
		for m in module.modules():
			if isinstance(m, nn.Conv2d):
				init.xavier_uniform(m.weight, gain=np.sqrt(2))
				init.constant(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()


	def forward(self, image_input):
		x = self.layer1(image_input)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.view(x.size(0), -1)

		return x

	def _conv(self, in_channels, out_channels, kernel_size, keep_rate=1.0):
		nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, stride=1, bias=True)
					  , nn.ReLU(inplace=True)
					  , nn.BatchNorm2d(out_channels)
					  , nn.MaxPool2d(kernel_size=2, stride=2)
					  , nn.Dropout2d(p=keep_rate))


