import os

import torch as th
import torch.nn.functional as F

import numpy as np
from torchviz import make_dot

device = th.device("cpu")

BoolTensor = th.BoolTensor
FloatTensor = th.FloatTensor


def get_folder(folder, root='trained', allow_exist=False):
    folder = os.path.join(root, folder)
    if not allow_exist:
        if os.path.exists(folder):
            raise FileExistsError

    log_path = os.path.join(folder, 'logs\\')
    graph_path = os.path.join(folder, 'graph\\')
    model_path = os.path.join(folder, 'model\\')

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    return {'folder': folder,
            'log_path': log_path,
            'graph_path': graph_path,
            'model_path': model_path}


def to_torch(np_array):
    return th.from_numpy(np_array)


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return th.Tensor(size).uniform_(-v, v)


def weight_init(m):
    if isinstance(m, th.nn.Conv2d) or isinstance(m, th.nn.Linear):
        m.weight.data.fill_(0.)
        m.bias.data.fill_(0.)


def net_visual(dim_input, net, **kwargs):
    xs = [th.randn(*dim).requires_grad_(True).to(device) for dim in dim_input]  # 定义一个网络的输入值
    y = net(*xs)  # 获取网络的预测值
    net_vis = make_dot(y, params=dict(list(net.named_parameters()) + [('x', x) for x in xs]))
    net_vis.render(**kwargs)     # 生成文件


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)


def interpolate_vars(old_vars, new_vars, epsilon):
    """
    Interpolate between two sequences of variables.
    """
    final_vars = old_vars.copy()
    for var_name, value in old_vars.items():
        mean_var_value = th.mean(th.stack([var_seq[var_name] for var_seq in new_vars]), dim=0)
        final_vars[var_name] = value + (mean_var_value - value) * epsilon
    return final_vars


def set_dynamic_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def gumbel_softmax(logits, discrete_list, noisy=False, var=1.0):
    actions = []
    for action in th.split(logits, discrete_list, dim=-1):
        if noisy:
            act_noisy = th.randn(action.shape) * var
            action += act_noisy
        actions.append(F.gumbel_softmax(action, hard=True))
    return th.cat(actions, dim=-1)
