#coding=utf-8
import os
import torch.utils.model_zoo as model_zoo

def download_model(url):
	pretrain_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pretrain')

	return model_zoo.load_url(url, pretrain_dir)