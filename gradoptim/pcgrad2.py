import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


class PCGrad():
    def __init__(self, optimizer, num_tasks=3, beta= 0.2,reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        self.num_tasks = num_tasks
        self.admatrix = 0.1* torch.ones([self.num_tasks, self.num_tasks])
        self.beta = beta
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def state_dict(self):
        return {'optimizer': self._optim.state_dict(), 'admatrix': self.admatrix}

    def load_state_dict(self, state_dict: dict) -> None:
        self._optim.load_state_dict(state_dict['optimizer'])
        self.admatrix.copy_(state_dict['admatrix'])

    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grads, num_task = copy.deepcopy(grads), len(grads)
        for tn_i in range(num_task):
            task_index = list(range(num_task))
            task_index.remove(tn_i)
            random.shuffle(task_index)
            for tn_j in task_index:
                rho_ij = torch.dot(pc_grads[tn_i], grads[tn_j]) / (pc_grads[tn_i].norm() * grads[tn_j].norm())
                if rho_ij < self.admatrix[tn_i, tn_j]:
                    w = pc_grads[tn_i].norm() * (self.admatrix[tn_i, tn_j] * (1 - rho_ij ** 2).sqrt() - rho_ij * (
                            1 - self.admatrix[tn_i, tn_j] ** 2).sqrt()) / (grads[tn_j].norm() * (1 - self.admatrix[tn_i, tn_j] ** 2).sqrt())
                    pc_grads[tn_i] += grads[tn_j] * w
                self.admatrix[tn_i, tn_j] = (1 - self.beta) * self.admatrix[tn_i, tn_j] + self.beta * rho_ij
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grads]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grads]).sum(dim=0)
        else:
            exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grads]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            a= self._flatten_grad(grad, shape).norm()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 4)

    def forward(self, x):
        return self._linear(x)


class MultiHeadTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 2)
        self._head1 = nn.Linear(2, 4)
        self._head2 = nn.Linear(2, 4)
        self._head3 = nn.Linear(2, 4)

    def forward(self, x):
        feat = self._linear(x)
        return self._head1(feat), self._head2(feat), self._head3(feat)


if __name__ == '__main__':

    '''
    # fully shared network test
    torch.manual_seed(4)
    x, y = torch.randn(3, 3), torch.randn(3, 4)
    net = TestNet()
    y_pred = net(x)
    pc_adam = PCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred, y), loss2_fn(y_pred, y)
  
    pc_adam.pc_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)
    '''

    # print('-' * 80)
    # seperated shared network test

    torch.manual_seed(4)
    x, y = torch.randn(3, 3), torch.randn(3, 4)
    net = MultiHeadTestNet()
    y_pred_1, y_pred_2, y_pred_3 = net(x)
    pc_adam = PCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn, loss3_fn = nn.MSELoss(), nn.MSELoss(), nn.MSELoss()
    loss1, loss2, loss3 = loss1_fn(y_pred_1, y), loss2_fn(y_pred_2, y), loss3_fn(y_pred_3, y)

    pc_adam.pc_backward([loss1, loss2, loss3])
    for p in net.parameters():
        print(p.grad)

