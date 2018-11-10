#coding=utf-8
import os
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import DataLoader

this_dir = os.path.dirname(os.path.abspath(__file__))

"""
Omniglot dataset contains 1623 different handwritten characters from 50 different alphabets.
Each of the 1623 characters was drawn online via Amazon's Mechanical Turk by 20 different people.
"""
class OmniglotNShotDataset():
	def __init__(self, batch_size, classes_per_set=20, samples_per_class=1, seed=2017, shuffle=True, use_cache=True):
		"""
		Construct N-shot dataset
		:param batch_size:  Experiment batch_size
		:param classes_per_set: Integer indicating the number of classes per set
		:param samples_per_class: Integer indicating samples per class
		:param seed: seed for random function
		:param shuffle: if shuffle the dataset
		:param use_cache: if true,cache dataset to memory.It can speedup the train but require larger memory
		"""
		np.random.seed(seed)
		self.batch_size = batch_size
		self.classes_per_set = classes_per_set
		self.samples_per_class = samples_per_class
		self.shuffle = shuffle
		self.use_cache = True
		x = np.load(os.path.join(this_dir, '..', '/datasrc/Omniglot.npy'))
		if self.shuffle:
			np.random.shuffle(x)
		x = np.reshape(x, newshape=(x.shape[0], x.shape[1], 28, 28, 1))
		x_train, x_val, x_test = x[:1200], x[1200:1411], x[1411:]
		x_train = self.preprocess(x_train)
		x_val = self.preprocess(x_val)
		x_test = self.preprocess(x_test)
		self.dataset = {'train': x_train, 'val': x_val, 'test': x_test}
		self.use_cache = use_cache
		if self.use_cache:
			self.cache_index = {'train': 0, 'val': 0, 'test': 0}
			self.cache_data = {'train': self._load_batch_data(x_train), 'val': self._load_batch_data(x_val)
				, 'test': self._load_batch_data(x_test)}

	def get_train_batch(self):
		return self._get_a_batch('train')

	def get_val_batch(self):
		return self._get_a_batch('val')

	def get_test_batch(self):
		return self._get_a_batch('test')

	def _get_a_batch(self, dataset_name):
		"""
		return a dataset
		:param dataset_name:
		:return:
		"""
		if self.use_cache:
			data_list = self.cache_data[dataset_name]
			data_index = self.cache_index[dataset_name]
			if data_index < len(data_list):
				data_index = 0
			data = data_list[data_index]
		else:
			data = self._sample_a_batch(self.dataset[dataset_name])

		return data

	def preprocess(self, data):
		"""
		normailze the data
		:param data:
		:return:
		"""
		mean = np.mean(data)
		std = np.std(data)

		return (data - mean) / std

	def _sample_a_batch(self, data_pack):
		"""
		create a batch
		:param data_pack:
		:return:
		"""
		support_x = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class, data_pack.shape[2]
							  , data_pack.shape[3], data_pack.shape[4]), np.float32)
		support_y = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class), np.int32)
		target_x = np.zeros((self.batch_size, data_pack.shape[2], data_pack.shape[3], data_pack.shape[4]), np.float32)
		target_y = np.zeros((self.batch_size, 1), np.int32)
		for i in range(self.batch_size):
			label_idx = np.arange(data_pack.shape[0])
			sample_idx = np.arange(data_pack.shape[1])
			chosen_label_idx = np.random.choice(label_idx, size=self.classes_per_set, replace=False)
			chosen_sample_idx = np.random.choice(sample_idx, size=self.samples_per_class + 1, replace=False)
			chosen_target_label = np.random.choice(self.classes_per_set, size=1, replace=False)

			chosen_label = data_pack[chosen_label_idx][:, chosen_sample_idx]
			support_x[i] = chosen_label[:, :-1]
			support_y[i] = np.expand_dims(np.arange(self.classes_per_set), axis=1)
			target_x[i] = chosen_label[chosen_target_label, -1]
			target_y[i] = np.arange(self.classes_per_set)[chosen_target_label]

		return support_x, support_y, target_x, target_y

	def _load_batch_data(self, data_pack):
		"""
		prepare cache dataset
		:param data_pack:
		:return:
		"""
		dataset = []
		for _ in range(1000):
			support_x, support_y, target_x, target_y = self._sample_a_batch(data_pack)
			dataset.append([support_x, support_y, target_x, target_y])

		return dataset

class OmniglotClassifyDataset():
	def __init__(self, batch_size, shuffle=True, transform=None, label_transform=None):
		"""
		Construct N-shot dataset
		:param batch_size:  Experiment batch_size
		:param classes_per_set: Integer indicating the number of classes per set
		:param samples_per_class: Integer indicating samples per class
		:param seed: seed for random function
		:param shuffle: if shuffle the dataset
		:param use_cache: if true,cache dataset to memory.It can speedup the train but require larger memory
		"""
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.transform = transform
		self.label_transform = label_transform
		x = np.load(os.path.join(this_dir, '..', 'datasrc/Omniglot.npy'))
		if self.shuffle:
			np.random.shuffle(x)
		x = np.reshape(x, newshape=(x.shape[0], x.shape[1], 28, 28))
		print x.shape

		x_train, x_val, x_test = OmniglotDataSet(x[:, :14], transform=transform, label_transform=label_transform), \
								 OmniglotDataSet(x[:, 14:17], transform=transform, label_transform=label_transform), \
								 OmniglotDataSet(x[:, 17:], transform=transform, label_transform=label_transform)
		self.dataset = {'train': DataLoader(dataset=x_train, batch_size=self.batch_size, shuffle=self.shuffle),
						'val': DataLoader(dataset=x_val, batch_size=self.batch_size, shuffle=self.shuffle),
						'test': DataLoader(dataset=x_test, batch_size=self.batch_size, shuffle=self.shuffle)}

	def get_dataset(self):
		return self.dataset['train'], self.dataset['val'], self.dataset['test']

class OmniglotDataSet(torch.utils.data.Dataset):
	def __init__(self, x, transform=None, label_transform=None):
		self.x = x
		self.transform = transform
		self.label_transform = label_transform

	def __getitem__(self, index):
		row = index / self.x.shape[0]
		col = index % (self.x.shape[1] - 1)
		img = self.x[row, col]
		img_ = Image.fromarray(np.uint8(img))
		if self.transform is not None:
			img_ = self.transform(img_)
		if self.label_transform is not None:
			row = self.label_transform(row)

		return img_, row

	def __len__(self):
		return self.x.shape[0] * self.x.shape[1]


if __name__ == '__main__':
	x = np.load(os.path.join(this_dir, '..', 'datasrc/Omniglot.npy'))
	x = np.reshape(x, newshape=(x.shape[0], x.shape[1], 28, 28))
	print x.shape

	output_ = '/media/yishi/e8a4df38-fa5b-4ad9-bc59-99d2c868f77e/label/test'
	for idx in range(x.shape[0]):
		output_dir = os.path.join(output_, str(idx))
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		for idx2 in range(x.shape[1]):
			output_file = os.path.join(output_dir, str(idx2) + '.jpg')
			cv2.imwrite(output_file, x[idx][idx2])
