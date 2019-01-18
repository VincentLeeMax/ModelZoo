#coding=utf-8
import torchvision
from torch.utils.data import DataLoader

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
			self.dataset['train'] = DataLoader(dataset=x_train, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=1)
			self.classes = x_train.classes
		if val_path is not None:
			x_val = torchvision.datasets.ImageFolder(val_path, transform=self.transform)
			self.dataset['val'] = DataLoader(dataset=x_val, batch_size=1, shuffle=self.shuffle, num_workers=1)
			self.classes = x_train.classes
		if test_path is not None:
			x_test = torchvision.datasets.ImageFolder(test_path, transform=self.transform)
			self.dataset['test'] = DataLoader(dataset=x_test, batch_size=1, shuffle=self.shuffle, num_workers=1)
			self.classes = x_train.classes

	def get_dataset(self):
		return self.dataset['train'], self.dataset['val'], self.dataset['test']

	def get_class(self):
		return self.classes
