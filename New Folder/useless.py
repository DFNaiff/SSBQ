# -*- coding: utf-8 -*-

import numpy as np

A = np.array([[1,1,1],[0,0,1],[9,3,1]])
print(np.linalg.cond(A))
print(np.linalg.inv(A))