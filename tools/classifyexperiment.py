#coding=utf-8
import os
import sys
import logging
import torch
from torch.autograd import Variable
from visdom import Visdom

logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s')

class classifyexperiment():
	def __init__(self, home_dir, net, max_epoch, criterion, optimizer, scheduler, snapshot=0, use_cuda=True, visdomName=None):
		self.home_dir = home_dir
		self.net = net
		self.max_epoch = max_epoch
		self.criterion = criterion
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.snapshot = snapshot
		self.use_cuda = use_cuda
		self.init_visdom(visdomName)

	def init_visdom(self, visdomName):
		if visdomName is None:
			self.viz = Visdom(env=os.path.basename(self.home_dir))
		else:
			self.viz = Visdom(env=visdomName)
		if self.viz.check_connection():
			self.viz.line(X=torch.FloatTensor([0]), Y=torch.FloatTensor([0]), win='train_loss_batch', opts={'title': 'train_loss_batch'})
			self.viz.line(X=torch.FloatTensor([0]), Y=torch.FloatTensor([0]), win='train_loss_epoch', opts={'title': 'train_loss_epoch'})
			self.viz.line(X=torch.FloatTensor([0]), Y=torch.FloatTensor([0]), win='train_acc_batch', opts={'title': 'train_acc_batch'})
			self.viz.line(X=torch.FloatTensor([0]), Y=torch.FloatTensor([0]), win='train_acc_epoch', opts={'title': 'train_acc_epoch'})
			self.viz.line(X=torch.FloatTensor([0]), Y=torch.FloatTensor([0]), win='val_acc_epoch', opts={'title': 'val_acc_epoch'})

	def set_dataloader(self, train_dataloader, val_dataloader):
		self.train_dataloader = train_dataloader
		self.val_dataloader = val_dataloader

	def get_net(self):
		return self.net

	def run(self):
		# 运行模型
		for epoch in range(0, self.max_epoch):
			self.scheduler.step()
			self.train(epoch)
			# 清除部分无用变量
			torch.cuda.empty_cache()
			self.val(epoch)
			# 清除部分无用变量
			torch.cuda.empty_cache()
			if self.snapshot != 0 and epoch % self.snapshot == 0:
				self.save(epoch)

		if self.snapshot == 0 or self.max_epoch % self.snapshot != 0:
			self.save(self.max_epoch)

	def train(self, epoch):
		self.net.train(True)
		train_loss = 0.0
		correct = 0.0
		total = 0

		# batch
		batch_len = len(self.train_dataloader)
		for batch_idx, (inputs, labels) in enumerate(self.train_dataloader):
			if self.use_cuda:
				inputs, labels = inputs.cuda(), labels.cuda()

			# zero optimizer grad
			self.optimizer.zero_grad()
			# Variable
			inputs, labels = Variable(inputs), Variable(labels)
			outputs = self.net(inputs)
			loss = self.criterion(outputs, labels)
			# loss backward
			loss.backward()
			# update
			self.optimizer.step()
			train_loss += loss.data
			# statist
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += predicted.eq(labels.data).cpu().sum()
			batch_acc = 100. * predicted.eq(labels.data).cpu().sum() / labels.size(0)
			sys.stdout.write('\rTrain || Batch_id: %d/%d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
							 % (batch_idx, batch_len, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
			if self.viz.check_connection():
				self.viz.line(X=torch.FloatTensor([epoch + 1. * batch_idx / batch_len])
							  , Y=torch.FloatTensor([batch_acc]), win='train_acc_batch', update='append')
				self.viz.line(X=torch.FloatTensor([epoch + 1. * batch_idx / batch_len])
							  , Y=torch.FloatTensor([loss.data]), win='train_loss_batch', update='append')
		sys.stdout.write('\n')
		sys.stdout.flush()
		logging.info('Epoch: %d || Train | lr: %f | Loss: %.3f | Acc: %.3f%% (%d/%d)'
				 % (epoch, self.scheduler.get_lr()[0], train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
		if self.viz.check_connection():
			self.viz.line(X=torch.FloatTensor([epoch])
						  , Y=torch.FloatTensor([100. * correct / total]), win='train_acc_epoch', update='append')
			self.viz.line(X=torch.FloatTensor([epoch])
						  , Y=torch.FloatTensor([train_loss / (batch_idx + 1)]), win='train_loss_epoch', update='append')

	def val(self, epoch):
		self.net.train(False)
		test_loss = 0.0
		correct = 0.0
		total = 0

		# batch
		batch_len = len(self.val_dataloader)
		for batch_idx, (inputs, labels) in enumerate(self.val_dataloader):
			if self.use_cuda:
				inputs, labels = inputs.cuda(), labels.cuda()

			# Variable
			inputs, labels = Variable(inputs), Variable(labels)
			outputs = self.net(inputs)
			loss = self.criterion(outputs, labels)
			test_loss += loss.data
			# statist
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += predicted.eq(labels.data).cpu().sum()
			sys.stdout.write('\rTest || Batch_id: %d/%d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
							 % (batch_idx, batch_len, test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
		sys.stdout.write('\n')
		sys.stdout.flush()
		logging.info('Epoch: %d || Test | Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (epoch, test_loss/(batch_idx+1), 100. * correct / total, correct, total))
		if self.viz.check_connection():
			self.viz.line(X=torch.FloatTensor([epoch])
						  , Y=torch.FloatTensor([100. * correct / total]), win='val_acc_epoch', update='append')

	def save(self, epoch):
		if not os.path.exists(self.home_dir):
			os.makedirs(self.home_dir)
		state = {'net': self.net.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
		torch.save(state, os.path.join(self.home_dir, '{}.pkl'.format(epoch)))
