# -*- coding: UTF-8 -*-
# File Name：basic_op
# Author : Chen Quan
# Date：2019/3/2
# Description :
__author__ = 'Chen Quan'

import numpy as np
import torch

# 将np.arrary转为为tensor
a = np.random.randn(4, 2, 5)
a_tensor = torch.Tensor(a)
print(a_tensor)
a_tensor = torch.from_numpy(a)
print(a_tensor)
