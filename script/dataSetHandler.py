#coding=utf-8
import os
import os.path as osp
import random
import shutil
import math

def getClassification():
	ratios = (0.9, 0.1, 0)
	balacne = 200
	input_ = '/media/yishi/d63bdc64-d33f-4981-9633-4f8ba01423ee/workspace/data/1107/traindata_1107'
	output_ = '/media/yishi/d63bdc64-d33f-4981-9633-4f8ba01423ee/workspace/data/1107/MS_dataset'

	files_output = osp.join(output_, 'file')
	if not osp.exists(files_output):
		os.makedirs(files_output)
	train_file_w = open(osp.join(files_output, 'train.txt'), 'wb')
	val_file_w = open(osp.join(files_output, 'val.txt'), 'wb')
	test_file_w = open(osp.join(files_output, 'test.txt'), 'wb')
	train_output = osp.join(output_, 'train')
	val_output = osp.join(output_, 'val')
	test_output = osp.join(output_, 'test')

	labels_index = {}
	files_index = {}
	idx = 0
	for class_ in os.listdir(input_):
		labels_index[str(idx)] = class_
		files_index.setdefault(str(idx), [])
		for root, dirs, files in os.walk(osp.join(input_, class_)):
			for file_ in files:
				if 'txt' not in file_:
					files_index[str(idx)].append(osp.join(root, file_))
		idx += 1

	for idx in files_index:
		label = labels_index[idx]
		files = files_index[idx]
		num = len(files)
		random.shuffle(files)
		train_files = files[0: int(round(num * ratios[0]))]
		val_files = files[int(round(num * ratios[0])): int(round(num * sum(ratios[:2])))]
		test_files = files[int(round(num * sum(ratios))): ]

		train_output_class = osp.join(train_output, label)
		val_output_class = osp.join(val_output, label)
		test_output_class = osp.join(test_output, label)
		if not osp.exists(train_output_class):
			os.makedirs(train_output_class)
		if not osp.exists(val_output_class):
			os.makedirs(val_output_class)
		if not osp.exists(test_output_class):
			os.makedirs(test_output_class)

		output_num = len(train_files)
		if balacne == 0:
			for train_file in train_files:
				shutil.copy(train_file, osp.join(train_output_class, osp.basename(train_file)))
				train_file_w.write(osp.join(train_output_class, osp.basename(train_file) + ' ' + str(idx) + os.linesep))
			for val_file in val_files:
				shutil.copy(val_file, osp.join(val_output_class, osp.basename(val_file)))
				val_file_w.write(osp.join(val_output_class, osp.basename(val_file) + ' ' + str(idx) + os.linesep))
			for test_file in test_files:
				shutil.copy(test_file, osp.join(test_output_class, osp.basename(test_file)))
				train_file_w.write(osp.join(test_output_class, osp.basename(test_file) + ' ' + str(idx) + os.linesep))
		elif output_num >= balacne:
			for train_file in train_files[0 : balacne]:
				shutil.copy(train_file, osp.join(train_output_class, osp.basename(train_file)))
				train_file_w.write(osp.join(train_output_class, osp.basename(train_file) + ' ' + str(idx) + os.linesep))
			for val_file in val_files[0 : int(round(balacne * ratios[1]))]:
				shutil.copy(val_file, osp.join(val_output_class, osp.basename(val_file)))
				val_file_w.write(osp.join(val_output_class, osp.basename(val_file) + ' ' + str(idx) + os.linesep))
			for test_file in test_files[0 : int(round(balacne * ratios[2]))]:
				shutil.copy(test_file, osp.join(test_output_class, osp.basename(test_file)))
				train_file_w.write(osp.join(test_output_class, osp.basename(test_file) + ' ' + str(idx) + os.linesep))
		else:
			for num in range(balacne):
				shutil.copy(train_files[num % output_num], osp.join(train_output_class, osp.basename(train_files[num % output_num])))
				val_file_w.write(osp.join(train_output_class, osp.basename(train_files[num % output_num]) + ' ' + str(idx) + os.linesep))
			for val_file in val_files:
				shutil.copy(val_file, osp.join(val_output_class, osp.basename(val_file)))
				val_file_w.write(osp.join(val_output_class, osp.basename(val_file) + ' ' + str(idx) + os.linesep))
			for test_file in test_files:
				shutil.copy(test_file, osp.join(test_output_class, osp.basename(test_file)))
				train_file_w.write(osp.join(test_output_class, osp.basename(test_file) + ' ' + str(idx) + os.linesep))

	train_file_w.flush()
	train_file_w.close()
	val_file_w.flush()
	val_file_w.close()
	test_file_w.flush()
	test_file_w.close()

if __name__ == '__main__':
	getClassification()