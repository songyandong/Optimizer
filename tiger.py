# coding=utf-8
#
# Tiger: A Budget-Conscious Neural Network Optimizer for PyTorch
# Tiger is an optimizer designed for cost-conscious neural network training.
# It started as a TensorFlow project (original repository),
# and this repository is a PyTorch adaptation of the original codebase.
#
# Features
# Achieves comparable performance to AdamW and LAMB.
# Minimizes memory requirements when using gradient accumulation.
# Adaptive learning rates per parameter, similar to LAMB.
# Simple strategy to prevent the model from collapsing to NaN.
# Can simulate any lr schedule with piecewise linear learning rates.
# We would like to express our gratitude to the original TensorFlow project (bojone/tiger)
# and its contributors for inspiring and providing the foundation for this PyTorch adaptation.
#
# https://github.com/balovess/Tiger-Optimizer-For-Pytorch/blob/main/tiger.py

import re

import torch
import torch.optim as optim


class Tiger(optim.Optimizer):
	"""Tiger Optimizer
	Link1: https://kexue.fm/archives/9512
	Link2: https://github.com/bojone/tiger
	Link3: https://kexue.fm/archives/8634
	"""
	def __init__(
			self,
			params,
			learning_rate=1e-3,
			beta=0.945,
			weight_decay=0.01,
			grad_accum_steps=64,
			lr_schedule={0: 1},
			shrink_ratio=0.99,
	):
		if not 0.0 <= learning_rate:
			raise ValueError("Invalid learning_rate: {}".format(learning_rate))
		if not 0.0 <= beta <= 1.0:
			raise ValueError("Invalid beta: {}".format(beta))
		if not 0.0 <= shrink_ratio <= 1.0:
			raise ValueError("Invalid shrink_ratio: {}".format(shrink_ratio))

		defaults = dict(
			learning_rate=learning_rate,
			beta=beta,
			weight_decay=weight_decay,
			grad_accum_steps=grad_accum_steps,
			lr_schedule={int(i): j for i, j in lr_schedule.items()},
			shrink_ratio=shrink_ratio
		)
		super(Tiger, self).__init__(params, defaults)

	def __setstate__(self, state):
		super(Tiger, self).__setstate__(state)

	def step(self, closure=None):
		loss = None
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue
				grad = p.grad.data

				if grad.is_sparse:
					raise RuntimeError(
						'Tiger optimizer does not support sparse gradients'
					)

				state = self.state[p]

				if 'step' not in state:
					state['step'] = 0
				if 'm' not in state:
					state['m'] = torch.zeros_like(p.data)

				t = state['step']
				d = group['weight_decay']
				k = group['grad_accum_steps']
				s = group['shrink_ratio']
				beta_tensor = torch.tensor(group['beta'])
				if t % k == 0:
					b1 = beta_tensor
				else:
					b1 = 1.0
				b2 = (1 - group['beta']) / k
				lr = group['learning_rate'] * self.piecewise_linear(t, group['lr_schedule'])
				if (t + 1) % k == 0:
					lr = lr
				else:
					lr = 0

				state['step'] += 1

				is_nan = torch.isnan(grad)
				b1 = torch.where(is_nan, torch.ones_like(b1), b1)
				g = torch.where(is_nan, torch.zeros_like(grad), grad)
				m = state['m']

				c = 0
				if p.name is None:
					name = "default_name"
				else:
					name = p.name

				if re.findall('bias|beta|gamma', name):
					lr, d = lr * 0.5, 0
					if 'gamma' in p.name:
						c = 1
				elif 'embeddings' in name:
					lr = lr * self.root_mean_square(p.data, axis=-1, keepdims=True)
				else:
					lr = lr * self.root_mean_square(p.data)

				m_t = b1 * m + b2 * g
				state['m'] = m_t

				# max(αt)∈[0.001, 0.002]
				if lr > 0.002:
					print('gama t 太大了:' + str(lr))
				# max(parameter) must be smaller then 1 if you want to train on bfloat16
				if torch.max(p.data) > 1:
					print('sita t 太大了:' + str(torch.max(p.data)))

				u = (torch.sign(m_t) + d * p.data) * lr
				v = torch.where(is_nan, (p.data - c) * s + c, p.data - u)
				p.data = v

		return loss

	@staticmethod
	def root_mean_square(x, axis=None, keepdims=False):
		"""Root Mean Square"""
		# 均方根: 数据点偏离平均值的一种度量|控制一组数据的离散程度
		# 参数x: 模型层的参数 分为3大类 bias|beta|gamma, embeddings, liner
		# 参数axis: 指定计算的均值的维度, embeddings层使用指定为最后一个dim(每个token内计算, 这样会得到每个token一个值)
		# 参数keepdim: 指定是否保持计算后维度不变, embeddings层使用指定为True(这样会得到总的一个值)
		return torch.sqrt(torch.mean((x - torch.mean(x)) ** 2, dim=axis, keepdim=keepdims))

	@staticmethod
	def piecewise_linear(t, schedule, from_zero=True):
		"""Piecewise Linear Function"""
		# 分段线性函数: 以简单的方式近似更复杂的函数
		# 参数t: 		当前步骤
		# 参数schedule: 	每个段的设置
		#     示例: {1000: 1, 2000: 0.1}
		#	      当 t ∈ [0, 1000] 时, 从0线性增长到1, 学习率预热;
		# 		  当 t ∈ [1000, 2000] 时, 从1线性降低到0.1, 学习率递减;
		# 		  当 t > 2000 时 保持0.1不变.
		# 参数from_zero: True or False, 是否从零开始进行

		# 排好顺序, 从小到大(0.1, 1)
		schedule = sorted(schedule.items())
		# 是否从零开始进行
		if from_zero and schedule[0][0] != 0:
			schedule = [(0, 0.0)] + schedule

		# 当前步骤(bfloat16)
		t = torch.tensor(t, dtype=torch.bfloat16)
		# 第一个段的学习率比例值(bfloat16)
		x = torch.tensor(schedule[0][1], dtype=torch.bfloat16)
		for i in range(len(schedule)):
			# 开始的时间步骤
			t_begin = schedule[i][0]
			# 开始的学习率比例值
			x_begin = x
			if i != len(schedule) - 1:
				# 找到后面一个和当前这个的差值
				dx = schedule[i + 1][1] - schedule[i][1]
				dt = schedule[i + 1][0] - schedule[i][0]
				slope = 1.0 * dx / dt
				# 非最后一段
				x = schedule[i][1] + slope * (t - t_begin)
			else:
				# 最后一段
				x = torch.tensor(schedule[i][1], dtype=torch.bfloat16)
			x = torch.where(t >= t_begin, x, x_begin)

		return x
