from __future__ import absolute_import, division, print_function
import random
import torch
import copy
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from collections import namedtuple
from collections import deque

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class MetaAgent(object):
    '''
    (1) act: how to sample an action to examine the learning
        outcomes or explore the environment;
    (2) memorize: how to store observed observations in order to
        help learing or establishing the empirical model of the
        enviroment;
    (3) learn: how the agent learns from the observations via
        explicitor implicit inference, how to optimize the policy
        model.
    '''

    def __init__(self, model, args, optimizer=None, is_train=False):
        self.model_ = model
        self.model = copy.deepcopy(model)
        self.is_train = is_train

        self.model_.share_memory()
        self.model.share_memory()

        self.method = args.method

        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        # decay to 0.05 for first 20% episode then keep steady 
        self.epsilon_delta = (args.epsilon - 0.05) / args.episode_num * 5.0

        self.mem_size = args.mem_size
        self.batch_size = args.batch_size
        self.weight_num = args.weight_num

        self.beta            = args.beta
        self.beta_init       = args.beta
        self.homotopy        = args.homotopy
        self.beta_uplim      = 1.00
        self.tau             = 1000.
        self.beta_expbase    = float(np.power(self.tau*(self.beta_uplim-self.beta), 1./args.episode_num))
        self.beta_delta      = self.beta_expbase / self.tau

        self.trans_mem = deque()
        # self.trans = namedtuple('trans', ['s', 'a', 's_', 'r', 'd'])
        # workaround for torch.multiprocessing...
        self.trans = lambda s, a, s_, r, d: dict(s=s, a=a, s_=s_, r=r, d=d)
        self.use_priority = args.priority
        self.priority_mem = deque()

        
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model_.parameters(), lr=args.lr)
        elif args.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model_.parameters(), lr=args.lr)
        
        if optimizer:
            self.optimizer.load_state_dict(optimizer.state_dict())

        self.w_kept = None
        self.update_count = 0
        self.update_freq = args.update_freq

        if self.is_train:
            self.model.train()
            self.model_.train()
        if use_cuda:
            self.model.cuda()
            self.model_.cuda()

    def act(self, state, preference=None):
        # random pick a preference if it is not specified
        if preference is None:
            if self.w_kept is None:
                self.w_kept = torch.randn(self.model_.reward_size)
                self.w_kept = (torch.abs(self.w_kept) / \
                               torch.norm(self.w_kept, p=1)).type(FloatTensor)
            preference = self.w_kept

        state = torch.from_numpy(state).type(FloatTensor)

        if self.method == "envelope":
            _, Q = self.model_(
                Variable(state.unsqueeze(0), requires_grad=False),
                Variable(preference.unsqueeze(0), requires_grad=False))

            Q = Q.view(-1, self.model_.reward_size)

            Q = torch.mv(Q.data, preference)

            action = Q.max(0)[1].cpu().numpy()
            action = int(action)

        elif self.method == "naive":
            _, Q = self.model_(
                Variable(state.unsqueeze(0), requires_grad=False),
                Variable(preference.unsqueeze(0), requires_grad=False))

            action = Q.max(1)[1].data.cpu().numpy()
            action = int(action[0])

        if self.is_train and (len(self.trans_mem) < self.batch_size or \
                              torch.rand(1)[0] < self.epsilon):
            action = np.random.choice(self.model_.action_size, 1)[0]
            action = int(action)
        elif torch.rand(1)[0] < self.epsilon:
            action = np.random.choice(self.model_.action_size, 1)[0]
            action = int(action)

        return action

    def memorize(self, state, action, next_state, reward, terminal):
        # self.trans_mem.append(self.trans(
        #     torch.from_numpy(state).type(FloatTensor),  # state
        #     action,  # action
        #     torch.from_numpy(next_state).type(FloatTensor),  # next state
        #     torch.from_numpy(reward).type(FloatTensor),  # reward
        #     terminal))  # terminal

        # save trasitions in RAM
        self.trans_mem.append(self.trans(
            torch.from_numpy(state).type(torch.FloatTensor),  # state
            action,  # action
            torch.from_numpy(next_state).type(torch.FloatTensor),  # next state
            torch.from_numpy(reward).type(torch.FloatTensor),  # reward
            terminal))  # terminal        

        # randomly produce a preference for calculating priority
        # preference = self.w_kept
        preference = torch.randn(self.model_.reward_size)
        preference = (torch.abs(preference) / torch.norm(preference, p=1)).type(FloatTensor)
        state = torch.from_numpy(state).type(FloatTensor)

        _, q = self.model_(Variable(state.unsqueeze(0), requires_grad=False),
                           Variable(preference.unsqueeze(0), requires_grad=False))

        q = q[0, action].data

        if self.method == "naive":
            wr = preference.dot(torch.from_numpy(reward).type(FloatTensor))
            if not terminal:
                next_state = torch.from_numpy(next_state).type(FloatTensor)
                hq, _ = self.model_(Variable(next_state.unsqueeze(0), requires_grad=False),
                                    Variable(preference.unsqueeze(0), requires_grad=False))
                hq = hq.data[0]
                p = abs(float(wr + self.gamma * hq - q))
            else:
                self.w_kept = None
                if self.epsilon_decay:
                    self.epsilon -= self.epsilon_delta
                p = abs(float(wr - q))

        elif self.method == "envelope":
            wq = preference.dot(q)
            wr = preference.dot(torch.from_numpy(reward).type(FloatTensor))
            if not terminal:
                next_state = torch.from_numpy(next_state).type(FloatTensor)
                hq, _ = self.model_(Variable(next_state.unsqueeze(0), requires_grad=False),
                                    Variable(preference.unsqueeze(0), requires_grad=False))
                hq = hq.data[0]
                whq = preference.dot(hq)
                p = abs(float(wr + self.gamma * whq - wq))

            else:
                print(self.beta)
                self.w_kept = None
                if self.epsilon_decay:
                    self.epsilon -= self.epsilon_delta
                if self.homotopy:
                    self.beta += self.beta_delta
                    self.beta_delta = (self.beta-self.beta_init)*self.beta_expbase+self.beta_init-self.beta
                p = abs(float(wr - wq))

        p += 1e-5

        if not self.use_priority:
            p = 1.0

        self.priority_mem.append(
            p
        )
        
        if len(self.trans_mem) > self.mem_size:
            self.trans_mem.popleft()
            self.priority_mem.popleft()

    def sample(self, pop, pri, k):
        pri = np.array(pri).astype(np.float)
        inds = np.random.choice(
            range(len(pop)), k,
            replace=False,
            p=pri / pri.sum()
        )
        return [pop[i] for i in inds]

    def actmsk(self, num_dim, index):
        mask = ByteTensor(num_dim).zero_()
        mask[index] = 1
        return mask.unsqueeze(0)

    def nontmlinds(self, terminal_batch):
        mask = ByteTensor(terminal_batch)
        inds = torch.arange(0, len(terminal_batch)).type(LongTensor)
        inds = inds[mask.eq(0)]
        return inds

    def learn(self, preferences=None):
        if len(self.trans_mem) > self.batch_size:

            self.update_count += 1

            action_size = self.model_.action_size
            reward_size = self.model_.reward_size

            minibatch = self.sample(self.trans_mem, self.priority_mem, self.batch_size)
            # minibatch = random.sample(self.trans_mem, self.batch_size)
            batchify = lambda x: list(x) * self.weight_num
            state_batch = batchify(map(lambda x: x['s'].type(FloatTensor).unsqueeze(0), minibatch))
            action_batch = batchify(map(lambda x: LongTensor([x['a']]), minibatch))
            reward_batch = batchify(map(lambda x: x['r'].type(FloatTensor).unsqueeze(0), minibatch))
            next_state_batch = batchify(map(lambda x: x['s_'].type(FloatTensor).unsqueeze(0), minibatch))
            terminal_batch = batchify(map(lambda x: x['d'], minibatch))

            w_batch = np.random.randn(self.weight_num, reward_size)
            
            if preferences is not None:
                w_batch = preferences.unsqueeze(0).cpu().numpy().repeat(self.batch_size, axis=0)
                w_batch = torch.from_numpy(w_batch).type(FloatTensor)
            else:
                w_batch = np.abs(w_batch) / \
                      np.linalg.norm(w_batch, ord=1, axis=1, keepdims=True)
                w_batch = torch.from_numpy(w_batch.repeat(self.batch_size, axis=0)).type(FloatTensor)


            if self.method == "naive":
                __, Q = self.model_(Variable(torch.cat(state_batch, dim=0)),
                                    Variable(w_batch))
                # detach since we don't want gradients to propagate
                # HQ, _    = self.model_(Variable(torch.cat(next_state_batch, dim=0), volatile=True),
                #                     Variable(w_batch, volatile=True))
                _, DQ = self.model(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
                                   Variable(w_batch, requires_grad=False))
                _, act = self.model_(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
                                     Variable(w_batch, requires_grad=False))[1].max(1)
                HQ = DQ.gather(1, act.unsqueeze(dim=1)).squeeze()

                w_reward_batch = torch.bmm(w_batch.unsqueeze(1),
                                           torch.cat(reward_batch, dim=0).unsqueeze(2)
                                           ).squeeze()

                nontmlmask = self.nontmlinds(terminal_batch)
                with torch.no_grad():
                    Tau_Q = Variable(torch.zeros(self.batch_size * self.weight_num).type(FloatTensor))
                    Tau_Q[nontmlmask] = self.gamma * HQ[nontmlmask]
                    Tau_Q += Variable(w_reward_batch)

                actions = Variable(torch.cat(action_batch, dim=0))

                # Compute Huber loss
                loss = F.smooth_l1_loss(Q.gather(1, actions.unsqueeze(dim=1)), Tau_Q.unsqueeze(dim=1))
            
            elif self.mehtod == "envelope":
                __, Q = self.model_(Variable(torch.cat(state_batch, dim=0)),
                                    Variable(w_batch), w_num=self.weight_num)

                # detach since we don't want gradients to propagate
                # HQ, _    = self.model_(Variable(torch.cat(next_state_batch, dim=0), volatile=True),
                #               Variable(w_batch, volatile=True), w_num=self.weight_num)
                _, DQ = self.model(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
                                   Variable(w_batch, requires_grad=False))
                w_ext = w_batch.unsqueeze(2).repeat(1, action_size, 1)
                w_ext = w_ext.view(-1, self.model.reward_size)
                _, tmpQ = self.model_(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
                                      Variable(w_batch, requires_grad=False))

                tmpQ = tmpQ.view(-1, reward_size)
                # print(torch.bmm(w_ext.unsqueeze(1),
                #            tmpQ.data.unsqueeze(2)).view(-1, action_size))
                act = torch.bmm(Variable(w_ext.unsqueeze(1), requires_grad=False),
                                tmpQ.unsqueeze(2)).view(-1, action_size).max(1)[1]

                HQ = DQ.gather(1, act.view(-1, 1, 1).expand(DQ.size(0), 1, DQ.size(2))).squeeze()

                nontmlmask = self.nontmlinds(terminal_batch)
                with torch.no_grad():
                    Tau_Q = Variable(torch.zeros(self.batch_size * self.weight_num,
                                                 reward_size).type(FloatTensor))
                    Tau_Q[nontmlmask] = self.gamma * HQ[nontmlmask]
                    # Tau_Q.volatile = False
                    Tau_Q += Variable(torch.cat(reward_batch, dim=0))

                actions = Variable(torch.cat(action_batch, dim=0))

                Q = Q.gather(1, actions.view(-1, 1, 1).expand(Q.size(0), 1, Q.size(2))
                             ).view(-1, reward_size)
                Tau_Q = Tau_Q.view(-1, reward_size)

                wQ = torch.bmm(Variable(w_batch.unsqueeze(1)),
                               Q.unsqueeze(2)).squeeze()

                wTQ = torch.bmm(Variable(w_batch.unsqueeze(1)),
                                Tau_Q.unsqueeze(2)).squeeze()

                # loss = F.mse_loss(Q.view(-1), Tau_Q.view(-1))
                loss = self.beta * F.mse_loss(wQ.view(-1), wTQ.view(-1))
                loss += (1-self.beta) * F.mse_loss(Q.view(-1), Tau_Q.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            # for param in self.model_.parameters():
            #     param.grad.data.clamp_(-5, 5)
            self.optimizer.step()

            if self.update_count % self.update_freq == 0:
                self.model.load_state_dict(self.model_.state_dict())

            return loss.data

        return 0.0

    def reset(self):
        self.w_kept = None
        if self.epsilon_decay:
            self.epsilon -= self.epsilon_delta
            if self.epsilon < 0.05:
                self.epsilon = 0.05
        if self.homotopy:
            self.beta += self.beta_delta
            self.beta_delta = (self.beta-self.beta_init)*self.beta_expbase+self.beta_init-self.beta

    def predict(self, state, probe):
        # random pick a preference if it is not specified
        if probe is None:
            if self.w_kept is None:
                self.w_kept = torch.randn(self.model_.reward_size)
                self.w_kept = (torch.abs(self.w_kept) / \
                               torch.norm(self.w_kept, p=1)).type(FloatTensor)
            probe = self.w_kept

        state = torch.from_numpy(state).type(FloatTensor)

        if self.method == "envelope":
            _, Q = self.model_(
                Variable(state.unsqueeze(0), requires_grad=False),
                Variable(probe.unsqueeze(0), requires_grad=False))

            Q = Q.view(-1, self.model_.reward_size)

            Q = torch.mv(Q.data, probe)

            pred_q = Q.max(0)[0].cpu().numpy()

        elif self.method == "naive":
            _, Q = self.model_(
                Variable(state.unsqueeze(0), requires_grad=False),
                Variable(probe.unsqueeze(0), requires_grad=False))

            pred_q = Q.max(1)[0].data.cpu().numpy()

        return pred_q

    def save(self, save_path, model_name):
        torch.save(self.model, "{}{}.pkl".format(save_path, model_name))
        torch.save(self.optimizer, "{}{}_opt.pkl".format(save_path, model_name))

