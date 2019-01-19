#coding=utf-8
import os
import torch
import torchvision
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader

class ImageTxt(data.Dataset):
	def __init__(self, lst_path, label_path, base_path='/', transform=None, size=None, loader=default_loader):
		self.transform = transform
		self.loader = loader
		self.size = size
		self.imgList = []
		self.labelList = []
		self.classes = {}
		with open(label_path, 'rb') as f:
			for line in f.readlines():
				line_infos = line.replace('\r', '').replace('\n', '').split('\t')
				self.classes[line_infos[1]] = line_infos[0]
		with open(lst_path, 'rb') as f:
			for line in f.readlines():
				line_infos = line.replace('\r', '').replace('\n', '').split('\t')
				self.imgList.append(os.path.join(base_path, line_infos[0]))
				self.labelList.append(int(line_infos[1]))

	def __getitem__(self, index):
		imgPath = self.imgList[index]
		img = self.loader(imgPath)
		if self.transform is not None:
			img = self.transform(img)
		label = self.labelList[index]

		return img, label

	def __len__(self):
		return len(self.imgList)

class ClassifyDataset():
	def __init__(self, batch_size, transform=None, label_transform=None, shuffle=True):
		"""
		Construct Classify dataset
		:param batch_size:  Experiment batch_size
		:param shuffle: Integer indicating the number of classes per set
		:param transform: Integer indicating samples per class
		:param label_transform: seed for random function
		"""
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.transform = transform
		self.label_transform = label_transform
		self.dataset = {'train': None, 'val': None, 'test': None}

	def from_dir(self, train_path=None, val_path=None, test_path=None):
		if train_path is not None:
			x_train = torchvision.datasets.ImageFolder(train_path, transform=self.transform)
			self.dataset['train'] = DataLoader(dataset=x_train, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4)
			self.classes = {int(idx): class_ for idx, class_ in enumerate(x_train.classes)}
		if val_path is not None:
			x_val = torchvision.datasets.ImageFolder(val_path, transform=self.transform)
			self.dataset['val'] = DataLoader(dataset=x_val, batch_size=self.batch_size / 4, shuffle=self.shuffle, num_workers=2)
			self.classes = {int(idx): class_ for idx, class_ in enumerate(x_val.classes)}
		if test_path is not None:
			x_test = torchvision.datasets.ImageFolder(test_path, transform=self.transform)
			self.dataset['test'] = DataLoader(dataset=x_test, batch_size=self.batch_size / 4, shuffle=self.shuffle, num_workers=2)
			self.classes = {int(idx): class_ for idx, class_ in enumerate(x_test.classes)}

	def from_txt(self, base_path, label_path, train_path=None, val_path=None, test_path=None):
		if train_path is not None:
			x_train = ImageTxt(train_path, label_path, base_path=base_path, transform=self.transform)
			self.dataset['train'] = DataLoader(dataset=x_train, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4)
			self.classes = x_train.classes
		if val_path is not None:
			x_val = ImageTxt(val_path, label_path, base_path=base_path, transform=self.transform)
			self.dataset['val'] = DataLoader(dataset=x_val, batch_size=self.batch_size / 4, shuffle=self.shuffle, num_workers=2)
			self.classes = x_val.classes
		if test_path is not None:
			x_test = ImageTxt(test_path, label_path, base_path=base_path, transform=self.transform)
			self.dataset['test'] = DataLoader(dataset=x_test, batch_size=self.batch_size / 4, shuffle=self.shuffle, num_workers=2)
			self.classes = x_test.classes

	def get_dataset(self):
		return self.dataset['train'], self.dataset['val'], self.dataset['test']

	def get_class(self):
		return self.classes
