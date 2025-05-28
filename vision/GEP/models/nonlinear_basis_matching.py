import copy
import gc
from collections import OrderedDict
from itertools import product
from typing import Optional
import psutil

import torch
import torch.nn as nn
import numpy as np
import math

# package for computing individual gradients
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from torch import optim
from torch.utils.data import DataLoader
# from vision.GEP.utils import print_memory_usage


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


def clip_column(tsr, clip=1.0, inplace=True):
    if inplace:
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
    U, S, Vh = torch.svd(pub_grad)
    L = Vh[:, :num_bases]
    error_rate, cosine, angle_deg = check_approx_error(L, pub_grad, return_cosine=True)
    # print(f'get_bases_svd: error_rate: {100*error_rate:.2f}%  cosine: {cosine}, angle(deg): {angle_deg}')
    return Vh, num_bases, error_rate


class GEP(nn.Module):

    def __init__(self, num_bases, batch_size, clip0=1, clip1=1, power_iter=1):
        super(GEP, self).__init__()

        self.num_bases = num_bases
        self.clip0 = clip0
        self.clip1 = clip1
        self.power_iter = power_iter
        self.batch_size = batch_size
        self.approx_error = {}

    def get_approx_grad(self, embedding):
        bases_list, num_bases_list, num_param_list = self.selected_bases_list, self.num_bases_list, self.num_param_list
        grad_list = []
        offset = 0
        if (len(embedding.shape) > 1):
            bs = embedding.shape[0]
        else:
            bs = 1
        embedding = embedding.view(bs, -1)

        for i, bases in enumerate(bases_list):
            num_bases = num_bases_list[i]

            grad = torch.matmul(embedding[:, offset:offset + num_bases].view(bs, -1), bases.T)
            if (bs > 1):
                grad_list.append(grad.view(bs, -1))
            else:
                grad_list.append(grad.view(-1))
            offset += num_bases
        if (bs > 1):
            return torch.cat(grad_list, dim=1)
        else:
            return torch.cat(grad_list)

    def get_anchor_gradients(self, net, loss_func):
        public_inputs, public_targets = self.public_inputs, self.public_targets
        outputs = net(public_inputs)
        loss = loss_func(outputs, public_targets)
        with backpack(BatchGrad()):
            loss.backward()
        cur_batch_grad_list = []
        for p in net.parameters():
            cur_batch_grad_list.append(p.grad_batch.reshape(p.grad_batch.shape[0], -1))
            del p.grad_batch
        return flatten_tensor(cur_batch_grad_list)

    def get_anchor_space(self, net, loss_func, logging=False):
        anchor_grads = self.get_anchor_gradients(net, loss_func)
        with torch.no_grad():
            num_param_list = self.num_param_list
            num_anchor_grads = anchor_grads.shape[0]
            num_group_p = len(num_param_list)

            selected_bases_list = []
            num_bases_list = []
            pub_errs = []

            sqrt_num_param_list = np.sqrt(np.array(num_param_list))
            num_bases_list = self.num_bases * (sqrt_num_param_list / np.sum(sqrt_num_param_list))
            num_bases_list = num_bases_list.astype(int)

            total_p = 0
            offset = 0

            for i, num_param in enumerate(num_param_list):
                pub_grad = anchor_grads[:, offset:offset + num_param]
                offset += num_param

                num_bases = num_bases_list[i]

                selected_bases, num_bases, pub_error = get_bases(pub_grad, num_bases, self.power_iter, logging)
                pub_errs.append(pub_error)

                num_bases_list[i] = num_bases
                selected_bases_list.append(selected_bases)

            self.selected_bases_list = selected_bases_list
            self.num_bases_list = num_bases_list
            self.approx_errors = pub_errs
        del anchor_grads

    def forward(self, target_grad, logging=False):
        with torch.no_grad():
            num_param_list = self.num_param_list
            embedding_list = []

            offset = 0
            if (logging):
                print('group wise approx error')

            for i, num_param in enumerate(num_param_list):
                grad = target_grad[:, offset:offset + num_param]
                selected_bases = self.selected_bases_list[i]
                embedding = torch.matmul(grad, selected_bases)
                num_bases = self.num_bases_list[i]
                if (logging):
                    cur_approx = torch.matmul(torch.mean(embedding, dim=0).view(1, -1), selected_bases.T).view(-1)
                    cur_target = torch.mean(grad, dim=0)
                    cur_error = torch.sum(torch.square(cur_approx - cur_target)) / torch.sum(torch.square(cur_target))
                    print('group %d, param: %d, num of bases: %d, group wise approx error: %.2f%%' % (
                    i, num_param, self.num_bases_list[i], 100 * cur_error.item()))
                    if (i in self.approx_error):
                        self.approx_error[i].append(cur_error.item())
                    else:
                        self.approx_error[i] = []
                        self.approx_error[i].append(cur_error.item())

                embedding_list.append(embedding)
                offset += num_param

            concatnated_embedding = torch.cat(embedding_list, dim=1)
            clipped_embedding = clip_column(concatnated_embedding, clip=self.clip0, inplace=False)
            if (logging):
                norms = torch.norm(clipped_embedding, dim=1)
                print('average norm of clipped embedding: ', torch.mean(norms).item(), 'max norm: ',
                      torch.max(norms).item(), 'median norm: ', torch.median(norms).item())
            avg_clipped_embedding = torch.sum(clipped_embedding, dim=0) / self.batch_size

            no_reduction_approx = self.get_approx_grad(concatnated_embedding)
            residual_gradients = target_grad - no_reduction_approx
            clip_column(residual_gradients, clip=self.clip1)  # inplace clipping to save memory
            clipped_residual_gradients = residual_gradients
            if (logging):
                norms = torch.norm(clipped_residual_gradients, dim=1)
                print('average norm of clipped residual gradients: ', torch.mean(norms).item(), 'max norm: ',
                      torch.max(norms).item(), 'median norm: ', torch.median(norms).item())

            avg_clipped_residual_gradients = torch.sum(clipped_residual_gradients, dim=0) / self.batch_size
            avg_target_grad = torch.sum(target_grad, dim=0) / self.batch_size
            return avg_clipped_embedding.view(-1), avg_clipped_residual_gradients.view(-1), avg_target_grad.view(-1)


def initialize_weights(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()


class Unet1dAutoencoder(nn.Module):
    def __init__(self, input_length, num_layers=6, latent_dim=1000, base_channels=64):
        """
        Args:
            input_length (int): Length of the input 1D tensor.
            num_layers (int): Number of encoder/decoder layers.
            latent_dim (int): Size of the latent representation.
            base_channels (int): Number of channels in the first conv layer.
        """
        super(Unet1dAutoencoder, self).__init__()

        self.input_length = input_length
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.ModuleList()
        in_channels = 1
        out_channels = base_channels

        for i in range(num_layers):
            self.encoder.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=3, padding=0))
            # self.encoder.append(nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=4, padding=0))
            print(f'encoder: layer {i} in_channels {in_channels} out_channels {out_channels}')
            in_channels = out_channels
            out_channels *= 2

        # Compute the output size after encoding
        dummy_input = torch.zeros(1, 1, input_length)
        with torch.no_grad():
            x = dummy_input
            for layer in self.encoder:
                x = torch.relu(layer(x))
            encoded = x
            self.encoded_channels = encoded.shape[1]
            self.encoded_length = encoded.shape[2]
            encoded_size = self.encoded_channels * self.encoded_length

        print(f'encoded size {encoded_size}')
        print(f'encoded channels {self.encoded_channels}')
        print(f'encoded length {self.encoded_length}')

        dummy_input = dummy_input.cpu().detach()
        x = x.cpu().detach()
        dummy_input = None
        x = None
        del dummy_input, x
        gc.collect()
        torch.cuda.empty_cache()

        # Bottleneck (Latent Space)
        self.bottleneck_enc = nn.Linear(encoded_size, self.latent_dim)
        self.bottleneck_dec = nn.Linear(self.latent_dim, encoded_size)

        # Decoder
        self.decoder = nn.ModuleList()
        in_channels = self.encoded_channels
        out_channels = in_channels // 2

        for i in range(num_layers):
            print(f'decoder: layer {i} in_channels {in_channels} out_channels {out_channels}')

            self.decoder.append(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=3, padding=0, output_padding=1))
            # nn.ConvTranspose1d(in_channels, out_channels, kernel_size=5, stride=4, padding=0, output_padding=1))
            in_channels = out_channels
            out_channels //= 2

        # Final output layer
        self.output_layer = nn.ConvTranspose1d(in_channels, 1, kernel_size=3, stride=3, padding=0, output_padding=1)
        # self.output_layer = nn.ConvTranspose1d(in_channels, 1, kernel_size=5, stride=4, padding=0, output_padding=1)

        initialize_weights(self)

        numeles = sum([p.numel() for p in self.parameters()])

        print(f'autoencoder number of parameters: {numeles}')

    def encode(self, x):
        # print('encode')
        # print(x.shape)
        batch_size = x.size(0)
        # Encoding
        for i, layer in enumerate(self.encoder):
            x = torch.relu(layer(x))
            # print(f'layer {i} shape {x.shape}')

        # Latent space
        x_flat = x.view(batch_size, -1)
        # print(f'x_flat shape {x_flat.shape}')
        latent = self.bottleneck_enc(x_flat)
        # print(f'latent shape {latent.shape}')
        return latent

    def decode(self, latent):
        batch_size = latent.size(0)
        # print('decode')
        # print(latent.shape)
        # unflat
        latent = self.bottleneck_dec(latent)
        # print(f'latent after bottleneck dec {latent.shape}')
        x = latent.view(batch_size, self.encoded_channels, self.encoded_length)
        # print(f'latent after unflat {x.shape}')
        # Decoding
        for i, layer in enumerate(self.decoder):
            x = torch.relu(layer(x))
            # print(f'layer {i} shape {x.shape}')

        # Output reconstruction
        output = self.output_layer(x)
        # print(output.shape)
        # Ensure the output matches the input size exactly
        output = output[..., :self.input_length]
        # print(output.shape)
        return output

    def forward(self, x):
        # print(f'autoencoder forward. x shape {x.shape}')
        latent = self.encode(x)

        # Output reconstruction
        output = self.decode(latent)

        return output, latent


class UNetGEP(GEP):
    def __init__(self, public_data_loader, input_length, num_layers, base_channels, num_bases, batch_size, clip0=1,
                 clip1=1):
        super(UNetGEP, self).__init__(num_bases, batch_size, clip0, clip1)
        self.input_length = input_length
        self.num_layers = num_layers
        self.latent_dim = num_bases
        self._ae = Unet1dAutoencoder(input_length, num_layers, latent_dim=num_bases, base_channels=base_channels)
        self.public_loader = public_data_loader

    def get_anchor_space(self, net, loss_func, logging=False):
        self._train_nonlinear_dim_reduction(net)

    @torch.no_grad()
    def get_approx_grad(self, embedding):
        approx_grad = self._ae.decode(embedding)
        return approx_grad

    def _train_nonlinear_dim_reduction(self, net: nn.Module) -> None:
        device = [p.device for p in net.parameters()][0]
        # print_memory_usage(cpu=False, gpu=True, device=device)

        criteria = torch.nn.CrossEntropyLoss()

        net.train()
        flat_grads_tensor_list = []

        optimizer = torch.optim.SGD(net.parameters(), lr=1.0, weight_decay=0.1, momentum=0.9)

        public_num_batches = 5
        batch_num = 0
        for batch in self.public_loader:
            batch_num += 1
            self.pbar_dict.update({'get pub grad batch': batch_num})
            self.pbar.set_postfix(self.pbar_dict, refresh=True)
            if batch_num > public_num_batches:
                break
            batch_grad_list = []
            x, Y = tuple(t.to(device) for t in batch)

            # print_memory_usage(device=device)

            optimizer.zero_grad()
            pred = net(x)
            loss = criteria(pred, Y)
            with backpack(BatchGrad()):
                loss.backward()
            for p in net.parameters():
                if p.grad is None or p.grad_batch is None:
                    # print('nograd for', p)
                    pass
                else:
                    batch_grad_list.append(p.grad_batch.reshape(p.grad_batch.shape[0], -1))
                    # print(p.grad_batch.shape)
                    p.grad_batch = p.grad_batch.detach().cpu()
                    p.grad_batch = None
                    del p.grad_batch
            # optimizer.step()
            flat_grads_tensor_list.append(flatten_tensor(batch_grad_list))

            batch_grad_list = [t.detach().cpu() for t in batch_grad_list]
            batch_grad_list = None
            del batch_grad_list
            x, Y, pred, loss = x.detach().cpu(), Y.detach().cpu(), pred.detach().cpu(), loss.detach().cpu()
            x, Y, pred, loss = None, None, None, None
            del x, Y, pred, loss
            gc.collect()
            torch.cuda.empty_cache()
        optimizer.zero_grad()

        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.vstack(flat_grads_tensor_list)),
                                                 batch_size=self.batch_size, shuffle=True)

        flat_grads_tensor_list = [t.detach().cpu() for t in flat_grads_tensor_list]
        flat_grads_tensor_list = None
        del flat_grads_tensor_list

        # autoencoder = get_autoencoder()
        self._ae = self._ae.to(device)
        autoencoder_optimizer = optim.Adam(self._ae.parameters(), lr=0.01)
        # autoencoder_loss_fn = nn.CosineEmbeddingLoss()
        autoencoder_loss_fn = nn.MSELoss()
        self._ae.train()
        # print_memory_usage(cpu=True, gpu=False)

        epoch_loss = 1.0

        for epoch in range(3):
            # print(f'\nEpoch {epoch + 1}/5 of training autoencoder epoch loss {epoch_loss}\n')
            if epoch_loss < 0.0001:
                # print('Early stop training autoencoder before epoch ', epoch + 1, ' with loss ', epoch_loss, '\n')
                break

            epoch_loss = 0.0
            for i, batch in enumerate(dataloader):
                self.pbar_dict.update({'ae_epoch': epoch, 'ae_batch': i})

                # print_memory_usage(cpu=True, gpu=False)

                data = batch[0].to(device)

                autoencoder_optimizer.zero_grad()
                # print(f'data shape {data.shape}')
                reconstructed, latent = self._ae(data.unsqueeze(1))

                # Trim reconstructed tensor to match data size
                reconstructed = reconstructed[..., :data.size(-1)]

                autoencoder_loss = autoencoder_loss_fn(reconstructed.squeeze(), data)  # Minimize reconstruction error
                batch_loss = float(autoencoder_loss)
                epoch_loss += batch_loss
                autoencoder_loss.backward()
                autoencoder_optimizer.step()

                data, reconstructed, latent = data.detach().cpu(), reconstructed.detach().cpu(), latent.detach().cpu()
                data, reconstructed, latent = None, None, None
                del data, reconstructed, latent

                autoencoder_loss = autoencoder_loss.detach().cpu()
                autoencoder_loss = None
                del autoencoder_loss
                gc.collect()
                torch.cuda.empty_cache()
                self.pbar_dict.update({"ae_batch_loss": batch_loss, 'ae_epoch_loss': epoch_loss})
                self.pbar.set_postfix(self.pbar_dict, refresh=True)
            # print(f'Epoch {epoch}  batch_loss {batch_loss} epoch_loss {epoch_loss}')
        self._ae = self._ae.to('cpu')

    def forward(self, target_grad, logging=False) -> tuple[torch.Tensor, torch.Tensor, int]:

        self._ae = self._ae.to(target_grad.device)
        self._ae.eval()
        with torch.no_grad():
            embedding = self._ae.encode(target_grad.unsqueeze(1))

            embedding_norms = torch.norm(embedding, dim=1)
            median_norm = torch.median(embedding_norms).item()
            clip_val = min(self.clip0, median_norm)

            clipped_embedding = clip_column(embedding, clip=clip_val, inplace=False)
            if logging:
                print(f'clipping embedding to min(median_norm, args.clip0): min({median_norm}, {self.clip0}) =',
                      clip_val)
                clipped_embedding_norms = torch.norm(clipped_embedding, dim=1)
                gradient_norms = torch.norm(target_grad, dim=1)
                print('average norm of embedding: ', torch.mean(embedding_norms).item(), 'max norm: ',
                      torch.max(embedding_norms).item(), 'median norm: ', torch.median(embedding_norms).item())
                print('average norm of clipped embedding: ', torch.mean(clipped_embedding_norms).item(), 'max norm: ',
                      torch.max(clipped_embedding_norms).item(), 'median norm: ',
                      torch.median(clipped_embedding_norms).item())
                print('average norm of target gradient: ', torch.mean(gradient_norms).item(), 'max norm: ',
                      torch.max(gradient_norms).item(), 'median norm: ', torch.median(gradient_norms).item())

            avg_clipped_embedding = torch.sum(clipped_embedding, dim=0) / self.batch_size

            avg_target_grad = torch.sum(target_grad, dim=0) / self.batch_size
            return avg_clipped_embedding.view(-1), avg_target_grad.view(-1), clip_val


if __name__ == '__main__':
    import pandas as pd
    res = []
    for ln, ld, bc in product([2,4,6],[128,256], [4,16,64]):
        print(ln, ld, bc)
        # 16 layers
        # model = Unet1dAutoencoder(input_length=8928, num_layers=ln, latent_dim=ld, base_channels=bc)
        # 6 layers
        model = Unet1dAutoencoder(input_length=3118, num_layers=ln, latent_dim=ld, base_channels=bc)
        # model = cifar10Net()
        numel = sum([p.numel() for p in model.parameters()])
        print('number of parameters:', numel)
        res.append([ln, ld, bc, numel])
    print(res)
    df = pd.DataFrame(columns=['layers', 'latent_dim', 'base_channel', 'numel'], data=res)
    min_numel = df['numel'].min()
    print('$$$$$$$$$$$$$$$')
    print(df)
    print('*************************')
    print('min numel', min_numel)
    print(df[df['numel'] == min_numel])


