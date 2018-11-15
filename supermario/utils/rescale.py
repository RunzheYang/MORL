from __future__ import absolute_import, division, print_function
import numpy as np
import torch
from skimage import color
from skimage.transform import resize

def rescale(state):
	state = color.rgb2gray(state)
	state = resize(state, (60,64,1), anti_aliasing=True, mode='constant')
	return state