import argparse
import visdom
import torch
import numpy as np
import plotly.graph_objs as go
import sys
import os

vis = visdom.Visdom()

with open('simple-preferences') as f:
	preferences = []
	for line in f:
		preferences.append([float(x) for x in next(f).split(',')])

with open('simple-rewards') as f:
	rewards = []
	for line in f:
		rewards.append([float(x) for x in next(f).split(',')])

preferences = np.array(preferences)
rewards = np.array(rewards)

# sort
sortind = preferences[:,1].argsort()
rewards = rewards[sortind]
preferences = preferences[sortind]
utility_c_x = []
utility_c_y = []

for i in xrange(len(preferences)):
	p = preferences[i]
	p_e = p / np.linalg.norm(preferences, ord=2)
	utility_c = p.dot(rewards[i]) * p_e
	utility_c_x.append(utility_c[0])
	utility_c_y.append(utility_c[1])

naive_act_opt = dict(x=utility_c_x,
                   y=utility_c_y,
                   mode="markers",
                   type='custom',
                   marker=dict(
                       symbol="circle",
                       color="rgb(51,148,148)",
                       # color="rgb(246,128,131)",
                       size=1),
                   fill='tozeroy',
                   name='Execution(Naive)')

layout_opt = dict(title="Dialogue Control Frontier",
        xaxis=dict(title='Length Penalty'),
        yaxis=dict(title='Success Reward'))

vis._send({'data': [naive_act_opt], 'layout': layout_opt})

print "done!"