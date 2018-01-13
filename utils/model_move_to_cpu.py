import torch

model = torch.load("{}{}.pkl".format("crl/envelope/saved/", "m.linear_e.ft_n.s2_r0"))
torch.save(model.cpu(), "{}{}.pkl".format("crl/envelope/saved/cpu/", "m.linear_e.ft_n.s2_r0"))

model = torch.load("{}{}.pkl".format("crl/envelope/saved/", "m.linear_e.ft_n.s2_r1"))
torch.save(model.cpu(), "{}{}.pkl".format("crl/envelope/saved/cpu/", "m.linear_e.ft_n.s2_r1"))

model = torch.load("{}{}.pkl".format("crl/envelope/saved/", "m.linear_e.ft_n.s2_r2"))
torch.save(model.cpu(), "{}{}.pkl".format("crl/envelope/saved/cpu/", "m.linear_e.ft_n.s2_r2"))

model = torch.load("{}{}.pkl".format("crl/envelope/saved/", "m.linear_e.ft_n.s2_r3"))
torch.save(model.cpu(), "{}{}.pkl".format("crl/envelope/saved/cpu/", "m.linear_e.ft_n.s2_r3"))

model = torch.load("{}{}.pkl".format("crl/envelope/saved/", "m.linear_e.ft_n.s2_r4"))
torch.save(model.cpu(), "{}{}.pkl".format("crl/envelope/saved/cpu/", "m.linear_e.ft_n.s2_r4"))



model = torch.load("{}{}.pkl".format("crl/naive/saved/", "m.linear_e.ft_n.s2_r0"))
torch.save(model.cpu(), "{}{}.pkl".format("crl/naive/saved/cpu/", "m.linear_e.ft_n.s2_r0"))

model = torch.load("{}{}.pkl".format("crl/naive/saved/", "m.linear_e.ft_n.s2_r1"))
torch.save(model.cpu(), "{}{}.pkl".format("crl/naive/saved/cpu/", "m.linear_e.ft_n.s2_r1"))

model = torch.load("{}{}.pkl".format("crl/naive/saved/", "m.linear_e.ft_n.s2_r2"))
torch.save(model.cpu(), "{}{}.pkl".format("crl/naive/saved/cpu/", "m.linear_e.ft_n.s2_r2"))

model = torch.load("{}{}.pkl".format("crl/naive/saved/", "m.linear_e.ft_n.s2_r3"))
torch.save(model.cpu(), "{}{}.pkl".format("crl/naive/saved/cpu/", "m.linear_e.ft_n.s2_r3"))

model = torch.load("{}{}.pkl".format("crl/naive/saved/", "m.linear_e.ft_n.s2_r4"))
torch.save(model.cpu(), "{}{}.pkl".format("crl/naive/saved/cpu/", "m.linear_e.ft_n.s2_r4"))
