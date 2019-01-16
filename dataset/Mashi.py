#coding=utf-8
import os
import torch
import numpy as np
import cv2
import torchvision
from PIL import Image
from torch.utils.data import DataLoader

class MashiClassifyDataset():
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

		x_train = torchvision.datasets.ImageFolder('/2T/workspace/data/MS_dataset/train'
																	, transform=transform)
		x_val = torchvision.datasets.ImageFolder('/2T/workspace/data/MS_dataset/val'
												   , transform=transform)
		x_test = torchvision.datasets.ImageFolder('/2T/workspace/data/MS_dataset/special_test'
												 , transform=transform)
		self.dataset = {'train': DataLoader(dataset=x_train, batch_size=self.batch_size, shuffle=self.shuffle),
						'val': DataLoader(dataset=x_val, batch_size=1, shuffle=self.shuffle),
						'test': DataLoader(dataset=x_test, batch_size=1, shuffle=self.shuffle)}
		self.classes = x_train.classes

	def get_dataset(self):
		return self.dataset['train'], self.dataset['val'], self.dataset['test']

	def get_class(self):
		return self.classes
