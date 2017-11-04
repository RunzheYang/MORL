import visdom
import torch
import math
import numpy as np
vis = visdom.Visdom()

assert vis.check_connection()

# Add data
gamma    = 0.99
time	 = [ -1, -3, -5, -7, -8, -9,   -13,	  -14,   -17,   -19]
treasure = [0.5, 2.8, 5.2, 7.3, 8.2, 9.0, 11.5, 12.1, 13.5, 14.2]

dis_time = (np.power(gamma, -np.asarray(time)) * np.asarray(time) / 14.2).tolist()
dis_treasure = (np.power(gamma, -np.asarray(time)) * np.asarray(treasure) / 14.2).tolist()

# Create and style traces
trace = dict(x=dis_time, 
			 y=dis_treasure,
			 mode="markers+lines",
			 type='custom',
			 marker=dict(
			 		# color =('rgb(205, 12, 24)'), 
			 		symbol="circle", 
			 		size  = 10),
			 line = dict(
					# color = ('rgb(205, 12, 24)'),
					width = 1,
					dash  = 'dash'),
			 name='real dst Pareto curve')

layout=dict(title="DST Pareto Frontier (gamma=%0.2f)"%gamma,
			xaxis=dict(   title = 'time penalty',
					   zeroline = False), 
			yaxis=dict(   title = 'teasure value',
					   zeroline = False))

# send to visdom
vis._send({'data': [trace], 'layout': layout})


see_map = np.array(
				[[   0,   0,   0,   0,   0,   0,     0,      0,      0,     0, 0],
				 [ 0.5,   0,   0,   0,   0,   0,     0,      0,      0,     0, 0],
				 [ -10, 2.8,   0,   0,   0,   0,     0,      0,      0,     0, 0],
				 [ -10, -10, 5.2,   0,   0,   0,     0,      0,      0,     0, 0],
				 [ -10, -10, -10, 7.3, 8.2, 9.0,     0,      0,      0,     0, 0],
				 [ -10, -10, -10, -10, -10, -10,     0,      0,      0,     0, 0],
				 [ -10, -10, -10, -10, -10, -10,     0,      0,      0,     0, 0],
				 [ -10, -10, -10, -10, -10, -10,  11.5,   12.1,      0,     0, 0],
				 [ -10, -10, -10, -10, -10, -10,   -10,    -10,      0,     0, 0],
				 [ -10, -10, -10, -10, -10, -10,   -10,    -10,   13.5,     0, 0],
				 [ -10, -10, -10, -10, -10, -10,   -10,    -10,    -10,  14.2, 0]]
			)[::-1]

vis.heatmap(X=see_map,
			opts=dict(
				title="DST Map",
				xmin = -10,
				xmax = 14.5))
