import copy
import gc
from collections import OrderedDict
from typing import Optional
import psutil

import numpy as np
import torch
import torch.nn as nn
# package for computing individual gradients
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm


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
        col = matrix[:, i : i + 1]
        col /= torch.sqrt(torch.sum(col ** 2))
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1 :]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col

def clip_column(tsr, clip=1.0, inplace=True):
    if inplace:
        inplace_clipping(tsr, torch.tensor(clip).cuda())
    else:
        norms = torch.norm(tsr, dim=1)

        scale = torch.clamp(clip/norms, max=1.0)
        return tsr * scale.view(-1, 1) 

@torch.jit.script
def inplace_clipping(matrix, clip):
    n, m = matrix.shape
    for i in range(n):
        # Normalize the i'th row
        col = matrix[i:i+1, :]
        col_norm = torch.sqrt(torch.sum(col ** 2))     
        if(col_norm > clip):
            col /= (col_norm/clip)

def check_approx_error(L, target):
    encode = torch.matmul(target, L) # n x k
    decode = torch.matmul(encode, L.T)
    error = torch.sum(torch.square(target - decode))
    target = torch.sum(torch.square(target))
    if(target.item()==0):
        return -1
    return error.item()/target.item()

def get_bases(pub_grad, num_bases, power_iter=1, logging=False):
    num_k = pub_grad.shape[0]
    num_p = pub_grad.shape[1]
  
    num_bases = min(num_bases, num_p)
    L = torch.normal(0, 1.0, size=(pub_grad.shape[1], num_bases), device=pub_grad.device)
    for i in range(power_iter):
        R = torch.matmul(pub_grad, L) # n x k
        L = torch.matmul(pub_grad.T, R) # p x k
        orthogonalize(L)
    error_rate = check_approx_error(L, pub_grad)
    return L, num_bases, error_rate



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
        if(len(embedding.shape)>1):
            bs = embedding.shape[0]
        else:
            bs = 1
        embedding = embedding.view(bs, -1)

        for i, bases in enumerate(bases_list):
            num_bases = num_bases_list[i]

            grad = torch.matmul(embedding[:, offset:offset+num_bases].view(bs, -1), bases.T)
            if(bs>1):
                grad_list.append(grad.view(bs, -1))
            else:
                grad_list.append(grad.view(-1))
            offset += num_bases           
        if(bs>1):
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
            num_bases_list = self.num_bases * (sqrt_num_param_list/np.sum(sqrt_num_param_list))
            num_bases_list = num_bases_list.astype(int)
            
            total_p = 0
            offset = 0

            for i, num_param in enumerate(num_param_list):
                pub_grad = anchor_grads[:, offset:offset+num_param]
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
            if(logging):
                print('group wise approx error')

            for i, num_param in enumerate(num_param_list): 
                grad = target_grad[:, offset:offset+num_param]
                selected_bases = self.selected_bases_list[i]
                embedding = torch.matmul(grad, selected_bases)
                num_bases = self.num_bases_list[i]
                if(logging):
                    cur_approx = torch.matmul(torch.mean(embedding, dim=0).view(1, -1), selected_bases.T).view(-1)
                    cur_target = torch.mean(grad, dim=0)
                    cur_error = torch.sum(torch.square(cur_approx-cur_target))/torch.sum(torch.square(cur_target))
                    print('group %d, param: %d, num of bases: %d, group wise approx error: %.2f%%'%(i, num_param, self.num_bases_list[i], 100*cur_error.item()))
                    if(i in self.approx_error):
                        self.approx_error[i].append(cur_error.item())
                    else:
                        self.approx_error[i] = []
                        self.approx_error[i].append(cur_error.item())

                embedding_list.append(embedding)
                offset += num_param


            concatnated_embedding = torch.cat(embedding_list, dim=1)
            clipped_embedding = clip_column(concatnated_embedding, clip=self.clip0, inplace=False) 
            if(logging):
                norms = torch.norm(clipped_embedding, dim=1)
                print('average norm of clipped embedding: ', torch.mean(norms).item(), 'max norm: ', torch.max(norms).item(), 'median norm: ', torch.median(norms).item())
            avg_clipped_embedding = torch.sum(clipped_embedding, dim=0) / self.batch_size

            no_reduction_approx = self.get_approx_grad(concatnated_embedding)
            residual_gradients = target_grad - no_reduction_approx
            clip_column(residual_gradients, clip=self.clip1) #inplace clipping to save memory
            clipped_residual_gradients = residual_gradients
            if(logging):
                norms = torch.norm(clipped_residual_gradients, dim=1)
                print('average norm of clipped residual gradients: ', torch.mean(norms).item(), 'max norm: ', torch.max(norms).item(), 'median norm: ', torch.median(norms).item())


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
    def __init__(self, input_length, num_layers=6, latent_dim=44, base_channels=64):
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

        for _ in range(num_layers):
            self.encoder.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=3, padding=0))
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

        for _ in range(num_layers):
            self.decoder.append(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=3, padding=0, output_padding=1))
            in_channels = out_channels
            out_channels //= 2

        # Final output layer
        self.output_layer = nn.ConvTranspose1d(in_channels, 1, kernel_size=3, stride=3, padding=0, output_padding=1)

        initialize_weights(self)


    def encode(self, x):
        # print('encode')
        # print(x.shape)
        batch_size = x.size(0)
        # Encoding
        for i,layer in enumerate(self.encoder):
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

        latent = self.encode(x)

        # Output reconstruction
        output = self.decode(latent)

        return output, latent



class UNetGEP(GEP):
    def __init__(self, public_data_loader, input_length, num_layers, base_channels, num_bases, batch_size, clip0=1, clip1=1):
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
        # allocated_memory = torch.cuda.memory_allocated(device) / 1024 ** 2
        # print(f"Allocated GPU memory: {allocated_memory:.2f} MB")

        criteria = torch.nn.CrossEntropyLoss()
        # for loader in public_loaders:

        net.train()
        batch_grad_list = []

        optimizer = torch.optim.SGD(net.parameters(), lr=1.0, weight_decay=0.1, momentum=0.9)
        # extend(net)
        criteria = extend(criteria)
        for batch in tqdm(self.public_loader):
            x, Y = tuple(t.to(device) for t in batch)
            # allocated_memory = torch.cuda.memory_allocated(device) / 1024 ** 2
            # print(f"Allocated GPU memory: {allocated_memory:.2f} MB")
            # cpu_memory = psutil.virtual_memory()
            # print(f"Available CPU memory: {cpu_memory.available / 1024 ** 2:.2f} MB")

            optimizer.zero_grad()
            pred = net(x)
            loss = criteria(pred, Y)
            with backpack(BatchGrad()):
                loss.backward()
            for p in net.parameters():
                if p.grad is None:
                    # print('nograd for', p)
                    pass
                else:
                    # print(p.grad.shape)
                    batch_grad_list.append(p.grad.reshape(p.grad.shape[0], -1))

            # optimizer.step()

            x, Y, pred, loss = x.detach().cpu(), Y.detach().cpu(), pred.detach().cpu(), loss.detach().cpu()
            x, Y, pred, loss = None, None, None, None
            del x, Y, pred, loss
            gc.collect()
            torch.cuda.empty_cache()
        optimizer.zero_grad()

        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(flatten_tensor(batch_grad_list)), batch_size=4, shuffle=True)


        # autoencoder = get_autoencoder()
        self._ae = self._ae.to(device)
        autoencoder_optimizer = optim.Adam(self._ae.parameters(), lr=0.01)
        # autoencoder_loss_fn = nn.CosineEmbeddingLoss()
        autoencoder_loss_fn = nn.MSELoss()
        self._ae.train()
        for epoch in range(5):
            for i, batch in enumerate(dataloader):
                data = batch[0].to(device)

                autoencoder_optimizer.zero_grad()

                reconstructed, latent = self._ae.forward(data.unsqueeze(1))

                # Trim reconstructed tensor to match data size
                reconstructed = reconstructed[..., :data.size(-1)]

                autoencoder_loss = autoencoder_loss_fn(reconstructed.squeeze(), data)  # Minimize reconstruction error
                # autoencoder_loss = autoencoder_loss_fn(reconstructed.squeeze(), data, torch.ones(reconstructed.shape[0]).to(device))  # Minimize reconstruction error

                data, reconstructed, latent = data.detach().cpu(), reconstructed.detach().cpu(), latent.detach().cpu()
                data, reconstructed, latent = None, None, None
                del data, reconstructed, latent
                gc.collect()
                torch.cuda.empty_cache()

                autoencoder_loss.backward()
                autoencoder_optimizer.step()
                # print(f'Epoch {epoch} iter {i} autoencoder_loss {autoencoder_loss.item()}')
        self._ae = self._ae.to('cpu')


    def forward(self, target_grad, logging=False) -> tuple[torch.Tensor, torch.Tensor]:
        self._ae.eval()
        with torch.no_grad():

            embedding = self._ae.encode(target_grad.unsqueeze(1))

            clipped_embedding = clip_column(embedding, clip=self.clip0, inplace=False)
            if (logging):
                embedding_norms = torch.norm(embedding, dim=1)
                clipped_embedding_norms = torch.norm(clipped_embedding, dim=1)
                gradient_norms = torch.norm(target_grad, dim=1)
                print('average norm of embedding: ', torch.mean(embedding_norms).item(), 'max norm: ',
                      torch.max(embedding_norms).item(), 'median norm: ', torch.median(embedding_norms).item())
                print('average norm of clipped embedding: ', torch.mean(clipped_embedding_norms).item(), 'max norm: ',
                      torch.max(clipped_embedding_norms).item(), 'median norm: ', torch.median(clipped_embedding_norms).item())
                print('average norm of target gradient: ', torch.mean(gradient_norms).item(), 'max norm: ',
                      torch.max(gradient_norms).item(), 'median norm: ', torch.median(gradient_norms).item())

            avg_clipped_embedding = torch.sum(clipped_embedding, dim=0) / self.batch_size

            avg_target_grad = torch.sum(target_grad, dim=0) / self.batch_size
            return avg_clipped_embedding.view(-1), avg_target_grad.view(-1)



