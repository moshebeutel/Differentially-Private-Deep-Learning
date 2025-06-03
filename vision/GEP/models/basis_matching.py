from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import math

# package for computing individual gradients
from backpack import backpack, extend
from backpack.extensions import BatchGrad

# from vision.GEP.models.nonlinear_basis_matching import initialize_weights
from models.nonlinear_basis_matching import initialize_weights

SVD = True


def flatten_tensor(tensor_list):
    for i in range(len(tensor_list)):
        tensor_list[i] = tensor_list[i].reshape([tensor_list[i].shape[0], -1])
    flatten_param = torch.cat(tensor_list, dim=1)
    del tensor_list
    return flatten_param


@torch.jit.script
def orthogonalize(matrix):
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i: i + 1]
        col /= torch.sqrt(torch.sum(col ** 2))
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col


def clip_column(tsr, clip=1.0, inplace=False):
    if (inplace):
        inplace_clipping(tsr, torch.tensor(clip).cuda())
    else:
        norms = torch.norm(tsr, dim=1)

        scale = torch.clamp(clip / norms, max=1.0)
        return tsr * scale.view(-1, 1)


@torch.jit.script
def inplace_clipping(matrix, clip):
    n, m = matrix.shape
    for i in range(n):
        # Normalize the i'th row
        col = matrix[i:i + 1, :]
        col_norm = torch.sqrt(torch.sum(col ** 2))
        if (col_norm > clip):
            col /= (col_norm / clip)


def cosine_similarity(a, b):
    a = a.mean(0, keepdim=False)
    b = b.mean(0, keepdim=False)
    cosine = torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
    angle_rad = torch.acos(cosine)
    angle_deg = float(angle_rad * 180 / np.pi)
    return cosine, angle_deg


def check_approx_error(L, target, return_cosine=False):
    encode = torch.matmul(target, L)  # n x k
    decode = torch.matmul(encode, L.T)
    error = float(torch.sum(torch.square(target - decode)))
    target_sum_squares = float(torch.sum(torch.square(target)))
    if target_sum_squares == 0:
        return -1
    if not return_cosine:
        return error / target_sum_squares
    else:
        cosine, angle_deg = cosine_similarity(target, decode)
        return error / target_sum_squares, float(cosine), angle_deg


def get_bases(pub_grad, num_bases, power_iter=1, logging=False):
    assert not SVD, 'SVD enabled'
    num_k = pub_grad.shape[0]
    num_p = pub_grad.shape[1]

    num_bases = min(num_bases, num_p)
    L = torch.normal(0, 1.0, size=(pub_grad.shape[1], num_bases), device=pub_grad.device)
    for i in range(power_iter):
        R = torch.matmul(pub_grad, L)  # n x k
        L = torch.matmul(pub_grad.T, R)  # p x k
        orthogonalize(L)
    error_rate = check_approx_error(L, pub_grad)
    return L, num_bases, error_rate


def get_bases_svd(pub_grad: torch.Tensor, num_bases: int) -> tuple[torch.Tensor, int, float]:
    assert SVD, 'SVD not enabled'
    num_k = pub_grad.shape[0]
    num_p = pub_grad.shape[1]

    num_bases = min(num_bases, num_p)
    # U, S, Vh = torch.svd(pub_grad)
    U, S, Vh = torch.linalg.svd(pub_grad, full_matrices=False)
    L = Vh.mH[:, :num_bases]
    error_rate, cosine, angle_deg = check_approx_error(L, pub_grad, return_cosine=True)
    # print(f'get_bases_svd: error_rate: {100*error_rate:.2f}%  cosine: {cosine}, angle(deg): {angle_deg}')
    return Vh.mH, num_bases, error_rate


class LinearCombination(nn.Module):
    def __init__(self, num_bases):
        super(LinearCombination, self).__init__()
        self.weight = nn.Parameter(torch.empty(1, num_bases))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, in_vec, perp_bases):
        assert perp_bases.shape[1] == self.weight.shape[1], \
            f'Expected perp_bases.shape[1] == self.weight.shape[1] but {perp_bases.shape[1]} != {self.weight.shape[1]}'
        return in_vec + self.weight @ perp_bases.T


class GEP(nn.Module):

    def __init__(self, num_bases, batch_size, clip0=1, clip1=1, power_iter=1, add_perp_vector=True):
        super(GEP, self).__init__()

        self.num_bases = num_bases
        self.clip0 = clip0
        self.clip1 = clip1
        self.power_iter = power_iter
        self.batch_size = batch_size
        self.approx_error_private = {}
        self.approx_error_public = []
        self.num_anchor_grads: int = 0
        self.selected_bases_list: list[torch.Tensor] = []
        self.selected_bases_perp_list: list[torch.Tensor] = []
        self.centers: list[torch.Tensor] = []
        self.add_perp_vector = add_perp_vector
        self.num_bases = num_bases
        self.num_bases_list: list[int] = [num_bases]
        self._num_param_list = [268346]

    @property
    def num_param_list(self):
        return self._num_param_list

    @num_param_list.setter
    def num_param_list(self, value):
        self._num_param_list = value
        assert hasattr(self, 'num_public_examples'), f'Expected definition of num_public_examples'
        assert isinstance(value, list), f'Expected value to be a list, but got {type(value)}'
        assert len(value) > 0, f'Expected value to be non-empty, but got {value}'
        sqrt_num_param_list = np.sqrt(np.array(value))
        num_bases_list: np.ndarray = self.num_bases * (sqrt_num_param_list / np.sum(sqrt_num_param_list))
        num_bases_list = num_bases_list.astype(int)

        if self.add_perp_vector:
            assert hasattr(self, 'num_public_examples'), 'Expected attribute num_public_examples'
            public_samples_num = int(self.num_public_examples * self.public_perp_split)\
                if hasattr(self, 'public_perp_split') else self.num_public_examples
            self.nullspace_factors = nn.ModuleList(
                [LinearCombination(num_bases=public_samples_num - num_bases) for
                 num_bases in num_bases_list])

        # initialize_weights(self)

        self.num_bases_list = num_bases_list.tolist()

    def get_approx_grad(self, embedding):
        bases_list, num_bases_list, num_param_list = self.selected_bases_list, self.num_bases_list, self.num_param_list
        grad_list = []
        offset = 0
        if len(embedding.shape) > 1:
            bs = embedding.shape[0]
        else:
            bs = 1
        embedding = embedding.view(bs, -1)

        for i, bases in enumerate(bases_list):
            num_bases = num_bases_list[i]

            if self.add_perp_vector:
                perp_bases = self.selected_bases_perp_list[i]
                assert hasattr(self, 'nullspace_factors'), f'Expected definition of nullspace factor'
                in_vec = torch.matmul(embedding[:, offset:offset + num_bases].view(bs, -1), bases.T)
                grad_centered = self.nullspace_factors[i](in_vec, perp_bases)
            else:
                grad_centered = torch.matmul(embedding[:, offset:offset + num_bases].view(bs, -1), bases.T)

            grad = self.centers[i] + grad_centered

            if bs > 1:
                grad_list.append(grad.view(bs, -1))
            else:
                grad_list.append(grad.view(-1))
            offset += num_bases
        if bs > 1:
            return torch.cat(grad_list, dim=1)
        else:
            return torch.cat(grad_list)

    def get_anchor_gradients(self, net, loss_func):
        public_inputs, public_targets = self.public_inputs, self.public_targets
        if self.add_perp_vector and hasattr(self, 'public_perp_split'):
            full_size = public_inputs.shape[0]
            split_ind = int(full_size * self.public_perp_split)
            public_inputs, public_targets = public_inputs[:split_ind], public_targets[:split_ind]

        outputs = net(public_inputs)
        loss = loss_func(outputs, public_targets)
        with backpack(BatchGrad()):
            loss.backward()
        cur_batch_grad_list = []
        for p in net.parameters():
            if hasattr(p, 'grad_batch'):
                cur_batch_grad_list.append(p.grad_batch.reshape(p.grad_batch.shape[0], -1))
                del p.grad_batch

        return flatten_tensor(cur_batch_grad_list)

    def get_anchor_space(self, net, loss_func, logging=False):
        anchor_grads = self.get_anchor_gradients(net, loss_func)
        with torch.no_grad():
            num_param_list = self.num_param_list
            self.num_anchor_grads = anchor_grads.shape[0]
            num_group_p = len(num_param_list)

            selected_bases_list = []
            selected_bases_perp_list = []
            pub_errs = []
            centers = []

            sqrt_num_param_list = np.sqrt(np.array(num_param_list))
            num_bases_list: np.ndarray[int] = self.num_bases * (sqrt_num_param_list / np.sum(sqrt_num_param_list))
            num_bases_list = num_bases_list.astype(int)

            total_p = 0
            offset = 0

            for i, num_param in enumerate(num_param_list):
                pub_grad = anchor_grads[:, offset:offset + num_param]
                centers.append(torch.mean(pub_grad, dim=0, keepdim=True))
                assert pub_grad.shape == (self.num_anchor_grads, num_param), (f'pub_grad.shape: {pub_grad.shape},'
                                                                              f' Expected (self.num_anchor_grads, '
                                                                              f'num_param) '
                                                                              f'{(self.num_anchor_grads, num_param)}')
                pub_grad_centered = pub_grad - centers[i]
                assert pub_grad_centered.shape == pub_grad.shape, (
                    f'pub_grad_centered.shape: {pub_grad_centered.shape},'
                    f' pub_grad.shape: {pub_grad.shape}')
                offset += num_param

                num_bases: int = num_bases_list[i]

                if not SVD:
                    selected_bases, num_bases, pub_error = get_bases(pub_grad, num_bases, self.power_iter, logging)
                else:
                    Vh, num_bases, pub_error = get_bases_svd(pub_grad_centered, num_bases)
                    selected_bases = Vh[:, :num_bases]
                    if self.add_perp_vector:
                        selected_bases_perp = Vh[:, num_bases:]
                pub_errs.append(pub_error)

                num_bases_list[i] = num_bases
                selected_bases_list.append(selected_bases)
                if self.add_perp_vector:
                    selected_bases_perp_list.append(selected_bases_perp)

            self.selected_bases_list = selected_bases_list
            if self.add_perp_vector:
                self.selected_bases_perp_list = selected_bases_perp_list
            self.num_bases_list = num_bases_list
            self.approx_error_public = pub_errs
            self.centers = centers
        del anchor_grads

    def forward(self, target_grad, logging=True):
        # with torch.no_grad():
        num_param_list = self.num_param_list
        embedding_list = []

        offset = 0
        if (logging):
            print('group wise approx error')

        for i, num_param in enumerate(num_param_list):
            grad = target_grad[:, offset:offset + num_param]
            assert grad.shape == (self.batch_size, num_param), f'grad.shape: {grad.shape},'
            assert self.centers[i].shape == (1, num_param), f'centers[i].shape: {self.centers[i].shape},'
            grad_centered = grad - self.centers[i]
            assert grad_centered.shape == grad.shape, f'grad_centered.shape: {grad_centered.shape},'
            selected_bases = self.selected_bases_list[i]
            assert selected_bases.shape == (num_param, self.num_bases_list[i]), (
                f'selected_bases.shape: {selected_bases.shape},')
            assert grad_centered.shape == (self.batch_size, num_param), (f'grad_centered.shape: {grad_centered.shape},')
            embedding = torch.matmul(grad_centered, selected_bases)
            assert embedding.shape == (self.batch_size, self.num_bases_list[i]), (
                f'embedding.shape: {embedding.shape},')
            num_bases = self.num_bases_list[i]
            if logging:
                cur_error: float = check_approx_error(selected_bases, grad_centered)
                # cur_approx = torch.matmul(torch.mean(embedding, dim=0).view(1, -1), selected_bases.T).view(-1)
                # cur_target = torch.mean(grad, dim=0)
                # cur_error = torch.sum(torch.square(cur_approx-cur_target))/torch.sum(torch.square(cur_target))
                print('group %d, param: %d, num of bases: %d, group wise approx error: %.2f%%'
                      % (i,  num_param, self.num_bases_list[i], 100 * float(cur_error)))
                if i in self.approx_error_private:
                    self.approx_error_private[i].append(float(cur_error))
                else:
                    self.approx_error_private[i] = []
                    self.approx_error_private[i].append(float(cur_error))

            embedding_list.append(embedding)
            offset += num_param

        concatenated_embedding = torch.cat(embedding_list, dim=1)
        assert concatenated_embedding.shape == (self.batch_size, sum(self.num_bases_list)), \
            f'concatenated_embedding.shape: {concatenated_embedding.shape}, Expected batch_size x sum(num_bases_list) {self.batch_size} x {sum(self.num_bases_list)} '
        with torch.no_grad():
            norms = torch.norm(concatenated_embedding, dim=1)
            median_norm = torch.median(norms).item()
            clip_val = min(self.clip0, median_norm)
        # print('clipping embedding to median norm:', clip_val)
        clipped_embedding = clip_column(concatenated_embedding, clip=clip_val, inplace=False)
        # clipped_embedding = clip_column(concatenated_embedding, clip=self.clip0, inplace=False)
        assert clipped_embedding.shape == concatenated_embedding.shape, f'clipped_embedding.shape: {clipped_embedding.shape},'
        if logging:
            with torch.no_grad():
                print(f'clipping embedding to min(median_norm, args.clip0): min({median_norm}, {self.clip0}) =',
                      clip_val)
                print('average norm of embedding: ', torch.mean(norms).item(), 'max norm: ', torch.max(norms).item(),
                      'median norm: ', torch.median(norms).item())
                norms = torch.norm(clipped_embedding, dim=1)
                print('average norm of clipped embedding: ', torch.mean(norms).item(), 'max norm: ',
                      torch.max(norms).item(), 'median norm: ', torch.median(norms).item())
        avg_clipped_embedding = torch.sum(clipped_embedding, dim=0) / self.batch_size
        assert avg_clipped_embedding.shape == (
        sum(self.num_bases_list),), f'avg_clipped_embedding.shape: {avg_clipped_embedding.shape},'
        no_reduction_approx = self.get_approx_grad(concatenated_embedding)
        assert no_reduction_approx.shape == target_grad.shape, f'no_reduction_approx.shape: {no_reduction_approx.shape},'
        residual_gradients = target_grad - no_reduction_approx
        if logging:
            with torch.no_grad():
                norms = torch.norm(residual_gradients, dim=1)
                print('average norm of residual gradients: ', torch.mean(norms).item(), 'max norm: ',
                      torch.max(norms).item(), 'median norm: ', torch.median(norms).item())

                cosine, angle_deg = cosine_similarity(residual_gradients, no_reduction_approx)
                print('cosine similarity between reconstructed and residual gradients: ', float(cosine), 'angle(deg)',
                      angle_deg)

        clip_column(residual_gradients, clip=self.clip1)  # inplace clipping to save memory
        clipped_residual_gradients = residual_gradients
        if logging:
            with torch.no_grad():
                cosine, angle_deg = cosine_similarity(clipped_residual_gradients, no_reduction_approx)
                print('cosine similarity between reconstructed and clipped residual gradients: ', float(cosine),
                      'angle(deg)', angle_deg)

        avg_clipped_residual_gradients = torch.sum(clipped_residual_gradients, dim=0) / self.batch_size
        avg_target_grad = torch.sum(target_grad, dim=0) / self.batch_size
        return avg_clipped_embedding.view(-1), avg_clipped_residual_gradients.view(-1), avg_target_grad.view(-1), clip_val
