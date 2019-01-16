#coding=utf-8
import torch.nn as nn
import torch.utils.model_zoo
from torch.autograd import Variable
from tools.modeldownloader import download_model
import math
import os
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s')
this_dir = os.path.dirname(os.path.abspath(__file__))
model_urls = {
	'VGG11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
	'VGG13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
	'VGG16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
	'VGG19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
	'VGG11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
	'VGG13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
	'VGG16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
	'VGG19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

cfgs = {
	'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
	def __init__(self, vgg_name='VGG11', num_classes=1000, in_channel=3, dropout=0.0, init=True):
		"""
		create a vgg network structure...
		:param vgg_name:
		"""
		super(VGG, self).__init__()
		assert vgg_name in cfgs
		self.num_classes = num_classes
		self.in_channel = in_channel
		self.dropout = dropout
		self.features = self._make_network(vgg_name)
		self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
										nn.ReLU(inplace=True),
										nn.Dropout(dropout), nn.Linear(4096, 4096),
										nn.ReLU(inplace=True),
										nn.Dropout(dropout))
		self.fc_ = nn.Linear(4096, num_classes)
		if init:
			self._init_weights()

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		x = self.fc_(x)

		return x

	def _make_network(self, vgg_name):
		cfg = cfgs[vgg_name]

		layers = []
		in_channel = self.in_channel
		for param in cfg:
			if param == 'M':
				layers += [nn.MaxPool2d(2, stride=2)]
			else:
				layers += [nn.Conv2d(in_channel, param, 3, padding=1)]
				if vgg_name.find('bn') != -1:
					layers += [nn.BatchNorm2d(param)]
				layers += [nn.ReLU(inplace=True)]
				in_channel = param

		return nn.Sequential(*layers)

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

def vgg_11(num_classes, pretrain=False, in_channel=3, dropout=0.0):
	model = VGG(vgg_name='VGG11', num_classes=num_classes, in_channel=in_channel, dropout=dropout, init=not pretrain)
	if pretrain:
		pretrained_dict = download_model(model_urls['VGG11'])
		model_dict = model.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		# 更新现有的model_dictcc
		model_dict.update(pretrained_dict)
		# 加载我们真正需要的state_dict
		model.load_state_dict(model_dict)
		logging.info("Loaded pretrain model..")


	return model

if __name__ == '__main__':
	vgg_11_model = vgg_11(100)
	image = torch.ones(1, 3, 224, 224)
	tensor = Variable(image)
	result = vgg_11_model.forward(tensor)

	print result
