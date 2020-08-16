# -*- coding: utf-8 -*-
import numpy as np
import h5py
import os
import numpy
import matplotlib.pyplot as plt
from scipy.io import loadmat

session_name='Chewie_10032013' # Chewie_10032013, Chewie_12192013,indy_20160407_02

annots = loadmat('../../Dataset/Chewie/'+ session_name  +'.mat')
print(annots.keys()) # out_struct

targets_corner = annots['out_struct']['targets'][0][0][0][0][0]
targets_rotation = annots['out_struct']['targets'][0][0][0][0][1]
pos = annots['out_struct']['pos'][0][0]
vel = annots['out_struct']['vel'][0][0]
acc = annots['out_struct']['acc'][0][0]

units = annots['out_struct']['units'][0][0] # 1x174
unit_id = 52
# units[0][unit_id][1]

print(pos.shape)
print(vel.shape)
print(acc.shape)
print(units.shape)
print('-'*30)
print(targets_corner.shape, '\n')
# print(targets_corner)
print(targets_rotation.shape, '\n')
# print(targets_corner)
print('-'*30)
print('units= ', units[0][unit_id], '\n')
print( len(units[0][unit_id][1]) )