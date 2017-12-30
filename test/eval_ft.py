import argparse
import visdom
import torch
import numpy as np
from sklearn.manifold import TSNE

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from envs.mo_env import MultiObjectiveEnv

parser = argparse.ArgumentParser(description='MORL-PLOT')
# CONFIG
parser.add_argument('--env-name', default='tf', metavar='ENVNAME',
                    help='environment to train on (default: tf)')
parser.add_argument('--method', default='crl-naive', metavar='METHODS',
                    help='methods: crl-naive | crl-envelope | crl-energy')
parser.add_argument('--model', default='linear', metavar='MODELS',
                    help='linear | cnn | cnn + lstm')
parser.add_argument('--gamma', type=float, default=0.99, metavar='GAMMA',
                    help='gamma for infinite horizonal MDPs')
# PLOT
parser.add_argument('--pltmap', default=False, action='store_true',
                    help='plot deep sea treasure map')
parser.add_argument('--pltpareto', default=False, action='store_true',
                    help='plot pareto frontier')
parser.add_argument('--pltcontrol', default=False, action='store_true',
                    help='plot control curve')
parser.add_argument('--pltdemo', default=False, action='store_true',
                    help='plot demo')
# LOG & SAVING
parser.add_argument('--save', default='crl/naive/saved/', metavar='SAVE',
                    help='address for saving trained models')
parser.add_argument('--name', default='', metavar='name',
                    help='specify a name for saving the model')
# Useless but I am too laze to delete them
parser.add_argument('--mem-size', type=int, default=10000, metavar='M',
                    help='max size of the replay memory')
parser.add_argument('--batch-size', type=int, default=256, metavar='B',
                    help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate')
parser.add_argument('--epsilon', type=float, default=0.3, metavar='EPS',
                    help='epsilon greedy exploration')
parser.add_argument('--epsilon-decay', default=False, action='store_true',
                    help='linear epsilon decay to zero')
parser.add_argument('--weight-num', type=int, default=32, metavar='WN',
                    help='number of sampled weights per iteration')
parser.add_argument('--episode-num', type=int, default=100, metavar='EN',
                    help='number of episodes for training')
parser.add_argument('--optimizer', default='Adam', metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--update-freq', type=int, default=32, metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--beta', type=float, default=0.02, metavar='BETA',
                    help='beta for evelope algorithm, default = 0.02')
parser.add_argument('--homotopy', default=False, action='store_true',
                    help='use homotopy optimization method')

args = parser.parse_args()

vis = visdom.Visdom()

assert vis.check_connection()

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

# Add data
FRUITS = [[0.26745039, 3.54435815, 4.39088762, 0.5898826, 7.7984232, 2.63110921],
          [0.46075946, 5.29084735, 7.92804145, 2.28448495, 1.01115855, 1.64300963],
          [0.5844333, 4.28059796, 7.00237899, 2.51448544, 4.32323182, 2.69974756],
          [4.01332296, 7.17080888, 1.46983043, 3.82182158, 2.20659648, 3.29195217],
          [3.74601154, 0.91228863, 5.92072559, 4.37056585, 2.73662976, 4.84656035],
          [2.42167773, 3.34415377, 6.35216354, 0.03806333, 0.66323198, 6.49313525],
          [5.26768145, 0.23364916, 0.23646111, 1.25030802, 1.41161868, 8.28161149],
          [0.19537027, 2.3433365, 6.62653841, 2.84247689, 1.71456358, 6.28809908],
          [5.9254461, 0.35473447, 5.4459742, 3.57702685, 0.95237377, 4.62628146],
          [2.22158757, 1.01733311, 7.9499714, 3.6379799, 3.77557594, 1.82692783],
          [4.43311346, 4.91328158, 5.11707495, 3.9065904, 2.22236853, 3.13406169],
          [6.44612546, 5.14526023, 1.37156642, 1.37449512, 0.62784821, 5.27343712],
          [2.39054781, 1.97492965, 4.51911017, 0.07046741, 1.74139824, 8.18077893],
          [3.26794393, 3.28877157, 2.91598351, 0.49403134, 7.86629258, 2.80694464],
          [3.96600091, 3.6266905, 4.44655634, 6.0366069, 1.58135473, 3.52204257],
          [6.15119272, 2.82397981, 4.24282686, 1.75378872, 4.80532629, 3.16535161],
          [2.7196025, 2.17993876, 2.79799651, 7.20950623, 4.70827355, 2.42446381],
          [0.29748325, 8.22965311, 0.07526586, 1.98395573, 1.77853129, 5.00793316],
          [6.37849798, 3.80507597, 2.5126212, 0.75632265, 2.49531244, 5.63243171],
          [0.79285198, 4.00586269, 0.36314749, 8.9344773, 1.82041716, 0.2318847],
          [0.24871352, 3.25946831, 3.9988045, 6.9335196, 4.81556096, 1.43535682],
          [5.2736312, 0.59346769, 0.73640014, 7.30730989, 4.09948515, 1.0448773],
          [1.74241088, 2.32320373, 9.17490044, 2.28211094, 1.47515927, 0.06168781],
          [1.65116829, 3.72063198, 5.63953167, 0.25461896, 6.35720791, 3.33875729],
          [2.5078766, 4.59291179, 0.81935207, 8.24752456, 0.33308447, 1.95237595],
          [1.05128312, 4.85979168, 3.28552824, 6.26921471, 3.39863537, 3.69171469],
          [6.30499955, 1.82204004, 1.93686289, 3.35062427, 1.83174219, 6.21238686],
          [4.74718378, 6.36499948, 4.05818821, 4.43996757, 0.42190953, 0.76864591],
          [1.25720612, 0.74301296, 1.3374366, 8.30597947, 5.08394071, 1.1148452],
          [0.63888729, 0.28507461, 4.87857435, 6.41971655, 5.85711844, 0.43757381],
          [0.74870183, 2.51804488, 6.59949427, 2.14794505, 6.05084902, 2.88429005],
          [3.57753129, 3.67307393, 5.43392619, 2.06131042, 2.63388133, 5.74420686],
          [3.94583726, 0.62586462, 0.72667245, 9.06686254, 1.13056724, 0.15630224],
          [2.53054533, 4.2406129, 2.22057705, 7.51774642, 3.47885032, 1.43654771],
          [1.63510684, 3.25906419, 0.37991887, 7.02694214, 2.53469812, 5.54598751],
          [7.11491625, 1.26647073, 5.01203819, 4.52740681, 1.16148237, 0.89835304],
          [2.75824608, 5.28476545, 2.49891273, 0.63079997, 7.07433925, 2.78829399],
          [4.92392025, 4.74424707, 2.56041791, 4.76935788, 1.43523334, 4.67811073],
          [2.43924518, 1.00523211, 6.09587506, 1.47285316, 6.69893956, 2.972341],
          [1.14431283, 4.55594834, 4.12473926, 5.80221944, 1.92147095, 4.85413307],
          [7.08401121, 1.66591657, 2.90546299, 2.62634248, 3.62934098, 4.30464879],
          [0.71623214, 3.11241519, 1.7018771, 7.50296641, 5.38823009, 1.25537605],
          [1.33651336, 4.76969307, 0.64008086, 6.48262472, 5.64538051, 1.07671362],
          [3.09497945, 1.2275849, 3.84351994, 7.19938601, 3.78799616, 2.82159852],
          [5.06781785, 3.12557557, 6.88555034, 1.21769126, 2.73086695, 2.86300362],
          [8.30192712, 0.40973443, 1.69099424, 4.54961192, 2.64473811, 0.59753994],
          [5.96294481, 6.46817991, 1.35988062, 2.83106174, 0.74946184, 3.48999411],
          [0.43320751, 1.24640954, 5.6313907, 1.62670791, 4.58871327, 6.54551489],
          [3.7064827, 7.60850058, 3.73003227, 2.71892257, 1.4363049, 2.23697394],
          [4.44128859, 1.8202686, 4.22272069, 2.30194565, 0.67272146, 7.30607281],
          [0.93689572, 0.77924846, 2.83896436, 1.98294555, 8.45958836, 3.86763124],
          [1.12281975, 2.73059913, 0.32294675, 2.84237021, 1.68312155, 8.95917647],
          [4.27687318, 2.83055698, 5.27541783, 5.03273808, 0.01475194, 4.53184284],
          [3.73578206, 6.07088863, 2.17391882, 4.89911933, 0.27124696, 4.51523815],
          [6.05671623, 0.7444296, 4.30057711, 3.09050824, 1.16194731, 5.77630391],
          [1.40468169, 5.19102545, 6.72110624, 4.75666122, 0.91486715, 1.56334486],
          [4.41604152, 0.86551038, 2.05709774, 4.70986355, 3.106477, 6.60944809],
          [5.95498781, 5.94146861, 4.17018388, 0.93397018, 0.89950814, 3.18829456],
          [9.59164585, 1.48925818, 0.72278285, 2.04850964, 1.0181982, 0.16402902],
          [4.4579775, 3.16479945, 1.00362159, 2.24428595, 7.91409455, 1.19729395],
          [2.12268361, 0.64607954, 6.43093367, 0.73854263, 6.94484318, 2.22341982],
          [3.08973572, 5.6223787, 0.9737901, 5.75218769, 3.94430958, 3.04119754],
          [2.5850297, 0.26144699, 2.28343938, 8.50777354, 3.93535625, 0.40734769],
          [4.72502594, 5.38532887, 5.40386645, 1.57883722, 0.24912224, 4.11288237]]

# apply gamma
FRUITS = np.array(FRUITS) * np.power(args.gamma, 5)


def matrix2lists(MATRIX):
    X, Y = [], []
    for x in list(MATRIX[:, 0]):
        X.append(float(x))
    for y in list(MATRIX[:, 1]):
        Y.append(float(y))
    return X, Y


def find_in(A, B):
    cnt = 0.0
    for a in A:
        for b in B:
            if np.linalg.norm(a - b, ord=1) < 1.0:
                cnt += 1.0
                break
    return cnt / len(A)


################# Control Frontier #################

if args.pltcontrol:

    # setup the environment
    env = MultiObjectiveEnv(args.env_name)

    # generate an agent for plotting
    agent = None
    if args.method == 'crl-naive':
        from crl.naive.meta import MetaAgent
    elif args.method == 'crl-envelope':
        from crl.envelope.meta import MetaAgent
    elif args.method == 'crl-energy':
        from crl.energy.meta import MetaAgent
    model = torch.load("{}{}.pkl".format(args.save,
                                         "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name)))
    agent = MetaAgent(model, args, is_train=False)

    # compute opt
    opt_x = []
    opt_y = []
    q_x = []
    q_y = []
    act_x = []
    act_y = []
    real_sol = FRUITS

    for i in range(2000):
        w = np.random.randn(6)
        w[2], w[3], w[4], w[5] = 0, 0, 0, 0
        w = np.abs(w) / np.linalg.norm(w, ord=1)
        # w = np.random.dirichlet(np.ones(2))
        w_e = w / np.linalg.norm(w, ord=2)
        if args.method == 'crl-naive' or args.method == 'crl-envelope':
            hq, _ = agent.predict(torch.from_numpy(w).type(FloatTensor))
        elif args.method == 'crl-energy':
            hq, _ = agent.predict(torch.from_numpy(w).type(FloatTensor), alpha=1e-5)
        realc = real_sol.dot(w).max() * w_e
        qc = w_e
        if args.method == 'crl-naive':
            qc = hq.data[0] * w_e
        elif args.method == 'crl-envelope':
            qc = w.dot(hq.data.cpu().numpy().squeeze()) * w_e
        elif args.method == 'crl-energy':
            qc = w.dot(hq.data.cpu().numpy().squeeze()) * w_e
        ttrw = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        terminal = False
        env.reset()
        cnt = 0
        while not terminal:
            state = env.observe()
            action = agent.act(state, preference=torch.from_numpy(w).type(FloatTensor))
            next_state, reward, terminal = env.step(action)
            if cnt > 30:
                terminal = True
            ttrw = ttrw + reward * np.power(args.gamma, cnt)
            cnt += 1
        ttrw_w = w.dot(ttrw) * w_e
        opt_x.append(realc[0])
        opt_y.append(realc[1])
        q_x.append(qc[0])
        q_y.append(qc[1])
        act_x.append(ttrw_w[0])
        act_y.append(ttrw_w[1])

    trace_opt = dict(x=opt_x,
                     y=opt_y,
                     mode="markers",
                     type='custom',
                     marker=dict(
                         symbol="circle",
                         size=1),
                     name='real')

    act_opt = dict(x=act_x,
                   y=act_y,
                   mode="markers",
                   type='custom',
                   marker=dict(
                       symbol="circle",
                       size=1),
                   name='policy')

    q_opt = dict(x=q_x,
                 y=q_y,
                 mode="markers",
                 type='custom',
                 marker=dict(
                     symbol="circle",
                     size=1),
                 name='predicted')

    ## quantitative evaluation
    policy_loss = 0.0
    predict_loss = 0.0
    for i in range(2000):
        w = np.random.randn(6)
        w = np.abs(w) / np.linalg.norm(w, ord=1)
        # w = np.random.dirichlet(np.ones(2))
        w_e = w / np.linalg.norm(w, ord=2)
        if args.method == 'crl-naive' or args.method == 'crl-envelope':
            hq, _ = agent.predict(torch.from_numpy(w).type(FloatTensor))
        elif args.method == 'crl-energy':
            hq, _ = agent.predict(torch.from_numpy(w).type(FloatTensor), alpha=1e-5)
        realc = real_sol.dot(w).max() * w_e
        qc = w_e
        if args.method == 'crl-naive':
            qc = hq.data[0] * w_e
        elif args.method == 'crl-envelope':
            qc = w.dot(hq.data.cpu().numpy().squeeze()) * w_e
        elif args.method == 'crl-energy':
            qc = w.dot(hq.data.cpu().numpy().squeeze()) * w_e
        ttrw = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        terminal = False
        env.reset()
        cnt = 0
        while not terminal:
            state = env.observe()
            action = agent.act(state, preference=torch.from_numpy(w).type(FloatTensor))
            next_state, reward, terminal = env.step(action)
            if cnt > 30:
                terminal = True
            ttrw = ttrw + reward * np.power(args.gamma, cnt)
            cnt += 1
        ttrw_w = w.dot(ttrw) * w_e

        policy_loss += np.linalg.norm(realc - ttrw_w, ord=1)
        predict_loss += np.linalg.norm(realc - qc, ord=1)

    policy_loss /= 2000.0
    predict_loss /= 2000.0


    print("discrepancies: policy-{}|predict-{}".format(policy_loss, predict_loss))

    layout_opt = dict(title="FT Control Frontier - {} {}({:.3f}|{:.3f})".format(
        args.method, args.name, policy_loss, predict_loss),
        xaxis=dict(title='1st objective'),
        yaxis=dict(title='2nd objective'))

    vis._send({'data': [trace_opt, act_opt, q_opt], 'layout': layout_opt})

################# Pareto Frontier #################

if args.pltpareto:

    # setup the environment
    env = MultiObjectiveEnv(args.env_name)

    # generate an agent for plotting
    agent = None
    if args.method == 'crl-naive':
        from crl.naive.meta import MetaAgent
    elif args.method == 'crl-envelope':
        from crl.envelope.meta import MetaAgent
    elif args.method == 'crl-energy':
        from crl.energy.meta import MetaAgent
    model = torch.load("{}{}.pkl".format(args.save,
                                         "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name)))
    agent = MetaAgent(model, args, is_train=False)

    # compute recovered Pareto
    act = []

    # predicted solution
    pred = []

    for i in range(2000):
        w = np.random.randn(6)
        w = np.abs(w) / np.linalg.norm(w, ord=1)
        # w = np.random.dirichlet(np.ones(6))
        ttrw = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        terminal = False
        env.reset()
        cnt = 0
        if args.method == "crl-envelope":
            hq, _ = agent.predict(torch.from_numpy(w).type(FloatTensor))
            pred.append(hq.data.cpu().numpy().squeeze() * 1.0)
        elif args.method == "crl-energy":
            hq, _ = agent.predict(torch.from_numpy(w).type(FloatTensor), alpha=1e-5)
            pred.append(hq.data.cpu().numpy().squeeze() * 1.0)
        while not terminal:
            state = env.observe()
            action = agent.act(state, preference=torch.from_numpy(w).type(FloatTensor))
            next_state, reward, terminal = env.step(action)
            if cnt > 50:
                terminal = True
            ttrw = ttrw + reward * np.power(args.gamma, cnt)
            cnt += 1

        act.append(ttrw)

    act = np.array(act)

    act_precition = find_in(act, FRUITS)
    act_recall = find_in(FRUITS, act)
    act_f1 = 2 * act_precition * act_recall / (act_precition + act_recall)
    pred_f1 = 0.0

    if not pred:
        pred = act
    else:
        pred = np.array(pred)
        pred_precition = find_in(pred, FRUITS)
        pred_recall = find_in(FRUITS, pred)
        if pred_precition > 1e-8 and pred_recall > 1e-8:
            pred_f1 = 2 * pred_precition * pred_recall / (pred_precition + pred_recall)

    FRUITS = np.tile(FRUITS, (30, 1))
    ALL = np.concatenate([FRUITS, act, pred])
    ALL = TSNE(n_components=2).fit_transform(ALL)
    p1 = FRUITS.shape[0]
    p2 = FRUITS.shape[0] + act.shape[0]

    fruit = ALL[:p1, :]
    act = ALL[p1:p2, :]
    pred = ALL[p2:, :]

    fruit_x, fruit_y = matrix2lists(fruit)
    act_x, act_y = matrix2lists(act)
    pred_x, pred_y = matrix2lists(pred)

    # Create and style traces
    trace_pareto = dict(x=fruit_x,
                        y=fruit_y,
                        mode="markers",
                        type='custom',
                        marker=dict(
                            symbol="circle",
                            size=10),
                        name='Pareto')

    act_pareto = dict(x=act_x,
                      y=act_y,
                      mode="markers",
                      type='custom',
                      marker=dict(
                          symbol="circle",
                          size=10),
                      name='Recovered')

    pred_pareto = dict(x=pred_x,
                       y=pred_y,
                       mode="markers",
                       type='custom',
                       marker=dict(
                           symbol="circle",
                           size=3),
                       name='Predicted')

    layout = dict(title="FT Pareto Frontier - {} {}({:.3f}|{:.3f})".format(
        args.method, args.name, act_f1, pred_f1))

    # send to visdom
    if args.method == "crl-naive":
        vis._send({'data': [trace_pareto, act_pareto], 'layout': layout})
    elif args.method == "crl-envelope":
        vis._send({'data': [trace_pareto, act_pareto, pred_pareto], 'layout': layout})
    elif args.method == "crl-energy":
        vis._send({'data': [trace_pareto, act_pareto, pred_pareto], 'layout': layout})

################# HEATMAP #################

if args.pltmap:
    FRUITS_EMB = TSNE(n_components=2).fit_transform(FRUITS)
    X, Y = matrix2lists(FRUITS_EMB)
    trace_fruit_emb = dict(x=X, y=Y,
                           mode="markers",
                           type='custom',
                           marker=dict(
                               symbol="circle",
                               size=10),
                           name='Pareto')
    layout = dict(title="FRUITS")
    vis._send({'data': [trace_fruit_emb], 'layout': layout})
