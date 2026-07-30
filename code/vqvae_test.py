'''
Copy of hierarchical_stage_1_autoencoder_mlp.py in almost every way except the Bottleneck

'''
import traceback
import concurrent.futures
import gc
import json
import math
import random
import re
import string
import uuid
from tqdm import tqdm
from statistics import mean
from typing import Optional
from collections import deque, defaultdict
from multiprocessing import Pool, cpu_count
import io
import pickle
import os
from datetime import datetime

import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast

from torchviz import make_dot

from prettytable import PrettyTable

import os
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

from torchvision.utils import save_image
from torch import nn
from torch.nn.attention import sdpa_kernel, SDPBackend
#from cosine_annealing_warm_restarts_decay import CosineAnnealingWarmRestartsDecay




import schedulefree
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint

class ModelConfig():
    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4096,
        num_encoder_layers=8,
        bottleneck_codebook_size=8,
        num_decoder_layers=8,
        patch_size=16,
        codebook_dropout=0.8,
    ):
        self.num_encoder_layers = num_encoder_layers
        self.bottleneck_codebook_size = bottleneck_codebook_size
        self.num_decoder_layers = num_decoder_layers
        
        self.hidden_size = hidden_size
        self.intermediate_size=intermediate_size
        
        self.patch_size = patch_size
        self.codebook_dropout = codebook_dropout
        

class RMSNorm2D(nn.Module):
    """RMSNorm over the channel dimension of an ``[B, C, H, W]`` tensor.

    ``nn.RMSNorm`` normalises the *last* dimension, which is wrong for NCHW
    feature maps, hence this variant.  The reduction is done in fp32 and cast
    back, so it is stable under bf16 autocast.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()

        rms = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        x = (x * rms).to(orig_dtype)

        return x * self.weight.view(1, -1, 1, 1)

    def extra_repr(self) -> str:
        return f"dim={self.weight.numel()}, eps={self.eps}"
        
class FusedSwiGLU2D(nn.Module):
    """SwiGLU MLP applied per-pixel to an ``[B, C, H, W]`` tensor via 1x1 convs.

    Mathematically the same as :class:`FusedSwiGLU` on a permuted tensor, but it
    skips two permutes per call, which matters inside the deep refinement stacks.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.fused_proj = nn.Conv2d(hidden_size, 2 * intermediate_size, kernel_size=1, bias=False)
        self.down_proj = nn.Conv2d(intermediate_size, hidden_size, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.fused_proj(x).chunk(2, dim=1)
        return self.down_proj(F.silu(gate) * up)


class MLPBlock2D(nn.Module):
    """Pre-norm residual SwiGLU block for ``[B, C, H, W]`` feature maps."""

    def __init__(self, input_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.premlp_norm = RMSNorm2D(input_dim)
        self.mlp = FusedSwiGLU2D(input_dim, intermediate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.premlp_norm(x)
        x = self.mlp(x)
        return res + x

       
class VQVAE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.encoder = Encoder(config)
        self.bottleneck = Bottleneck(config)
        self.decoder = Decoder(config)
    
    @torch.compile
    def forward(self, images):
        multiresolution_features = self.encoder(images)
        latents, codes = self.bottleneck(multiresolution_features)
        all_decoded = self.decoder(latents)
        return all_decoded, codes
        
class Encoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.unshuffle = nn.PixelUnshuffle(downscale_factor=config.patch_size)
        
        self.patch_proj = nn.Conv2d(config.patch_size*config.patch_size*3, config.hidden_size, kernel_size=1, bias=False)
        
        self.layers = nn.ModuleList([MLPBlock2D(config.hidden_size, config.intermediate_size) for _ in range(config.num_encoder_layers)])
    
    def forward(self, x):
        # patchify images
        x = self.unshuffle(x)
        x = self.patch_proj(x)
                
        for layer in self.layers:
            x = layer(x)
        return x
     
   
class Bottleneck(nn.Module):
    def __init__(self, config:ModelConfig):
        super().__init__()
        self.config = config
        
        self.norm = RMSNorm2D(config.hidden_size)
        
        self.state_to_codebook = nn.Linear(config.hidden_size, config.bottleneck_codebook_size, bias=False)
        self.codebook_to_state = nn.Embedding(config.bottleneck_codebook_size, config.hidden_size)
        
        
    
    def forward(self, hidden_state):
        hidden_state = self.norm(hidden_state)
        # B,C,H,W -> B,H,W,C
        hidden_state = hidden_state.permute(0,2,3,1)
        
        logits = self.state_to_codebook(hidden_state)
        mask = torch.rand_like(logits) < self.config.codebook_dropout
        logits[mask] = float('-inf')
        
        probs = F.softmax(logits, dim=-1)
        
        soft = probs @ self.codebook_to_state.weight
        
        # B,H,W
        chosen_codes = logits.argmax(dim=-1)
        hard = self.codebook_to_state(chosen_codes)
        
        result = soft + (hard - soft).detach()
        
        # B,H,W,C -> 
        result = result.permute(0,3,1,2)
        
        return result, chosen_codes
        
    
class Decoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.layers = nn.ModuleList([MLPBlock2D(config.hidden_size, config.intermediate_size) for _ in range(config.num_decoder_layers)])
        self.shuffle = nn.PixelShuffle(upscale_factor=config.patch_size)
        self.output_proj = nn.Conv2d(in_channels=config.hidden_size,out_channels=config.patch_size*config.patch_size*3, kernel_size=1, bias=False)
                
        self.receptive_field_expansion = nn.Conv2d(in_channels=config.hidden_size, out_channels=config.hidden_size, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x):
        x = self.receptive_field_expansion(x)
        
        for layer in self.layers:
            x = layer(x)
                
        x = self.output_proj(x)
        x = self.shuffle(x)        
        return x
        
class FFHQDataset(Dataset):
    def __init__(self, image_folder, device=None):
        self.image_folder = image_folder
        self.device = device
        self.all_images = set(range(70000))
        self.update_seen_images(set())
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def update_seen_images(self, new_seen_images):
        valid_seen_images = []
        for image_id in self.all_images:
            if image_id in new_seen_images:
                continue
            valid_seen_images.append(image_id)
        self.active_images = valid_seen_images
        
    def __len__(self):
        return len(self.active_images)
    
    def _get_image_path(self, image_id):
        return os.path.join(self.image_folder, f"{image_id:05d}.png")
    
    def __getitem__(self, idx):
        image_path = self._get_image_path(self.active_images[idx])
        with Image.open(image_path) as image:
            image = self.transform(image).to(self.device)
        image = image.to(self.device)
        image = normalize_image(image)
        
        if random.random() < 0.5: # 50% chance to h-flip
            image = torchvision.transforms.functional.hflip(image)
        
        return image, self.active_images[idx]
        

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
def generate_unique_prefix(length=8):
    return str(uuid.uuid4())[:length]
       
def get_warmup_scheduler(optimizer, num_warmup_steps, target_lr, last_epoch=-1):
    def lr_lambda(current_step):
        current_step = current_step + 0
        if current_step < num_warmup_steps:
            return (current_step+1.0)/num_warmup_steps#(current_step + 1.0) / num_warmup_steps * target_lr
        return 1.0#target_lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)
    
class PatchManager():
    def __init__(self, config, lr, device, name, model_type, label):
        """
        PatchManager: Utility for managing the patch generator, its optimizer, and scheduler.
        """
        self.config = config
        self.lr = lr
        self.device = device
        self.name = name
        self.model_type = model_type
        self.label = label
        
        if model_type == 'vqvae_mlp_hierarchical':
            self.model = VQVAE(config).to(device)
        else:
            raise ValueError("Unknown model_type passed to PatchManager!")
        
        self.model = self.model.to(device)
        
        self.model.train()
        
        # Set up the optimizer and scheduler.
        '''
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = CosineAnnealingWarmRestartsDecay(self.optimizer, T_0=100, T_mult=1.05, eta_min=self.lr*0.5, decay_factor=0.97)
        '''
        self.reset_optimizer()
    
    def reset_optimizer(self):
        
        self.optimizer = schedulefree.AdamWScheduleFree(self.model.parameters(), lr=self.lr, weight_decay=5e-5)
        
        self.optimizer.train()
        self.scheduler = get_warmup_scheduler(self.optimizer, 100, self.lr)
        
        
    def save(self, epoch, steps, total_microbatches, seen_ids, microbatch_size):
        """
        Save the state of the model, optimizer, and scheduler to a file.
        """
        resulting_filepath = f'/mnt/f/models_trained/{self.label}_{self.name}_epoch_{epoch}_step_{steps}_{self.model_type}.pth'
        self.optimizer.eval()
        torch.save({
            'epoch': epoch,
            'steps': steps,
            'total_microbatches': total_microbatches,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict' : self.scheduler.state_dict(),
            'seen_ids': seen_ids,
            'microbatch_size': microbatch_size
        }, resulting_filepath)
        self.optimizer.train()
        return resulting_filepath
        
    def load(self, filepath):
        """
        Load the model, optimizer, and scheduler states from a saved file.
        
        Parameters:
        - filepath: Path to the saved model file.
        """
        # Load the checkpoint data from the file.
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        unloaded_state_dict = self.model.state_dict()
        checkpoint_state_dict = checkpoint['model_state_dict']
        
        failed_to_load_fully = False
        for key in unloaded_state_dict.keys():
            if not key in checkpoint_state_dict:
                print(f"WARNING: UNSET KEY:{key}")
                checkpoint_state_dict[key] = unloaded_state_dict[key]
                failed_to_load_fully = True
        
        duplicate_state_dict = {}
        for key in checkpoint_state_dict.keys():
            if not key in unloaded_state_dict:
                print(f"WARNING: DELETED KEY:{key}")
                failed_to_load_fully = True
            else:
                duplicate_state_dict[key] = checkpoint_state_dict[key]
                
        for key in checkpoint_state_dict.keys():
            if key in unloaded_state_dict:
                value_loaded = checkpoint_state_dict[key]
                value_unloaded = unloaded_state_dict[key]
                if value_loaded.size() != value_unloaded.size():
                    print(f"SIZE CHANGED -- {value_loaded.size()} -> {value_unloaded.size()} ({key})")
                    failed_to_load_fully = True
                    
                    if value_loaded.dim() == value_unloaded.dim():
                        dims_changed = 0
                        slices = []
                        for dim_a, dim_b in zip(value_loaded.size(), value_unloaded.size()):
                            slices.append(slice(0,min(dim_a, dim_b)))
                            if dim_a != dim_b:
                                dims_changed += 1
                        slices = tuple(slices)
                        # if multiple dims changed writing old data would be weird...
                        if dims_changed == 1:
                            value_unloaded[slices] = value_loaded[slices]
                            print(f"\t └── Partially rewrote {[x.stop for x in slices]}")

                    duplicate_state_dict[key] = value_unloaded
        
        checkpoint_state_dict = duplicate_state_dict
        
        # Restore the model, optimizer, and scheduler states.
        self.model.load_state_dict(checkpoint_state_dict)
        if not failed_to_load_fully:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.optimizer.train()
        else:
            print("RESETTING OPTIMIZER STATES DUE TO DIFFERENT PARAMS...")
        
        epoch = checkpoint['epoch']
        steps = checkpoint['steps']
        total_microbatches = checkpoint['total_microbatches']
        seen_ids = checkpoint['seen_ids']
        microbatch_size = checkpoint['microbatch_size']
        
        print(f"Successfully loaded model from {filepath}")
        self.optimizer.train()
        
        return epoch, steps, total_microbatches, seen_ids, microbatch_size
    
def save_progressive_growth_images(output_images, iterations, label='_training', max_images=1, scale_factor=1):
       # Check that there are images to save.
    if not output_images:
        raise ValueError("No output images were provided")

    # Limit the number of images, in case max_images is more than what's available
    max_images = min(max_images, output_images[0].size(0))

    # First, we'll concatenate the images from different stages of transformation
    # side-by-side (horizontally). This will show the progressive growth per image.
    # Note: We assume that all tensors in output_images have the same dimensions.
    
    # Concatenate the transformations horizontally for each image in the batch
    # Each tensor in 'horizontal_concatenations' is [max_images, C, H, W * num_transformations]
    horizontal_concatenations = [torch.cat([img_set[i] for img_set in output_images], dim=-1)
                                 for i in range(max_images)]
    
    # Now, we concatenate these horizontally aligned images vertically to see the
    # different batches on top of each other. This makes it easy to compare the stages
    # for different starting images.
    
    # 'final_image' tensor has dimensions [C, H * max_images, W * num_transformations]
    final_image = torch.cat(horizontal_concatenations, dim=1)
    
    final_image = final_image.view(final_image.size(0), final_image.size(1)//scale_factor, scale_factor, final_image.size(2)//scale_factor, scale_factor).mean(dim=(2,4))

    # Save the image
    save_image(final_image, f'output_images/progress_{label}_{iterations}.png', normalize=True)
    
        

# must be applied before [-1,1] transformation, since these values are based on [0,1] statistics
# results in (on average), an image with a mean of 0 and a std of 1
def normalize_image(image):
    # Assuming image is in shape [C, H, W] or [B, C, H, W]
    mu = torch.tensor([0.5326635696186963, 0.4840257880077774, 0.4746593392437331]).view(-1, 1, 1).to(image.device)
    std = torch.tensor([0.23375539610753213, 0.22486911131803633, 0.22463677431223736]).view(-1, 1, 1).to(image.device)
    
    return (image - mu) / std
    
def denormalize_image(normalized_image):
    mu = torch.tensor([0.5326635696186963, 0.4840257880077774, 0.4746593392437331]).view(-1, 1, 1).to(normalized_image.device)
    std = torch.tensor([0.23375539610753213, 0.22486911131803633, 0.22463677431223736]).view(-1, 1, 1).to(normalized_image.device)
    
    return (normalized_image * std) + mu
        
def take_step(optimizers, models, schedulers):
    stop_propagating = False
    oob_optimizer = None
    
    for (optimizer, model) in zip(optimizers, models):
        for name, p in model.named_parameters():
            if p.grad is not None:
                if not torch.all(torch.isfinite(p.grad)):
                    stop_propagating = True
                    oob_optimizer = optimizer
                    break
        if stop_propagating:
            break
    
    if stop_propagating:
        print("Bad grads...")
        for optimizer in optimizers:
            optimizer.zero_grad()
        return False
    
    for optimizer in optimizers:
        optimizer.step()
        optimizer.zero_grad()
    for scheduler in schedulers:
        scheduler.step()
    return True

if __name__ == "__main__":
    torch._dynamo.config.cache_size_limit = 1024
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    prefix = generate_unique_prefix()
    
    patch_size=8
    writer = None
    dataset = FFHQDataset('/mnt/f/ffhq_diffusion/ffhq512',device)
            
    learning_rate = 0.0003
    

    model_config = ModelConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_encoder_layers=6,
        num_decoder_layers=6,
        patch_size=patch_size,
        bottleneck_codebook_size=1024,
        codebook_dropout=0.0,
    )
    
    celeb_manager = PatchManager(model_config, learning_rate, device, 'celeb', model_type='vqvae_mlp_hierarchical', label=prefix)
    
    # Hyperparameters etc.
    epochs = 10000
    samples_per_microbatch = 4#16
    base_grad_clip = 1.0
    
    patch_managers = [celeb_manager]
    

    
    
    total_param_out = []
    for m in patch_managers:
        total_params = count_parameters(m.model)
        total_param_out.append(f"{m.name} - {total_params}")
    for l in total_param_out:
        print(l)
    
    epoch_initial = 0
    
    iterations = 0
    total_microbatches = 0
    running_losses = defaultdict(list)
        
    skip_log = False
    warm_up_steps_remaining = 20
    
    microbatches_at_save = 0
    
    seen_ids = set()

    if True:
        epoch_initial, iterations, total_microbatches, seen_ids, microbatches_per_step = celeb_manager.load('/mnt/f/models_trained/d87e41f4_celeb_epoch_3_step_1276_vqvae_mlp_hierarchical.pth')
                
        microbatches_at_save = total_microbatches
        skip_log = False
    microbatches_per_step = 200 // samples_per_microbatch
        
        
    dataset.update_seen_images(seen_ids)
    data_loader = DataLoader(dataset, batch_size=samples_per_microbatch, shuffle=True, num_workers=0)
    print("Dataset size:", len(dataset))
    print("Samples per microbatch:", samples_per_microbatch)
    
    microbatches_since_last_step = 0
    
    start_time = None
    print("Prefix:",prefix)
    
    unique_codes = torch.zeros((model_config.bottleneck_codebook_size,),device=device, dtype=torch.bool)

    for epoch in range(epoch_initial, epochs):
        try:
            iterable_data_loader = iter(data_loader)
            
            for microbatch_i in tqdm(range(len(data_loader))):
                try:
                    batch = next(iterable_data_loader)
                    if batch is None:
                        print("Nothing in batch!")
                        continue
                    if batch[0].size(0) < samples_per_microbatch:
                        print("Batch too small!", batch[0].size(0))
                        continue
                except StopIteration:
                    break
                except Exception as e:
                    print(f"Error fetching or processing batch {microbatch_i}: {e}")
                    traceback.print_exc()
                    continue
                


                all_inputs = batch[0].to(device)
                ids = batch[1]
                seen_ids.update(ids.tolist())
                
                autoencoder_model = patch_managers[0].model
                
                
                # 1 - 512x512 (64x64 patches)
                # 2 - 256x256 (32x32 patches)
                # 4 - 128x128 (16x16 patches)
                # 8 -  64x64  ( 8x8  patches)
                #16 -  32x32  ( 4x4  patches)
                downscale_factors = [1,2,4,8,16]
                
                current_downscale_factor = downscale_factors[total_microbatches % len(downscale_factors)]
                
                with torch.no_grad():
                    all_inputs = all_inputs.reshape(all_inputs.size(0), all_inputs.size(1), all_inputs.size(2) // current_downscale_factor, current_downscale_factor, all_inputs.size(3) // current_downscale_factor, current_downscale_factor).mean(dim=(3,5), keepdim=False) 
                    
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    predicted, codes = autoencoder_model(all_inputs)
                    unique_codes[codes.view(-1)] = True
                    
                with torch.no_grad():
                    unnormalized_pred = denormalize_image(predicted)
                    unnormalized_input = denormalize_image(all_inputs)
                    mse = F.mse_loss(unnormalized_pred, unnormalized_input, reduction='none').mean(dim=(1,2,3))
                    psnr = 10*torch.log10(1/mse)
                    psnr = psnr.mean()
                    
                scaling_factor = 10.0 # makes MSE trigger far later (but MSE still triggers, different from setting beta to 0.1 because otherwise the grad drops from 1.0 to 0.01 at 0.1.  Here it's more continuous)
                all_losses = F.smooth_l1_loss(predicted*scaling_factor, all_inputs*scaling_factor, reduction='none', beta=1.0)
                loss = all_losses.mean()
                    
                scaled_total_loss = loss / microbatches_per_step
                scaled_total_loss.backward()
                
                running_losses[f'ae/downscaled_{current_downscale_factor}/loss'].append(loss.item())
                running_losses[f'ae/downscaled_{current_downscale_factor}/psnr'].append(psnr.item())
                running_losses['ae/total_loss'].append(loss.item())
                    
                outputting_images = iterations % 20 == 0 and microbatches_since_last_step == 0

                    
                total_microbatches += 1
                microbatches_since_last_step += 1
                if start_time is None:
                    start_time = datetime.now().timestamp()
                
                if outputting_images:
                    with torch.no_grad():
                        max_images = 2
                        
                        images = []
                        image = denormalize_image(predicted[:max_images])
                        image = torch.clamp(image,0,1)
                        images.append(image)
                        
                        save_progressive_growth_images(images, iterations, label=f'_training_{prefix}', max_images=max_images, scale_factor=4)
                        images = []

                            
                        
                    
                    
                if microbatches_since_last_step >= microbatches_per_step:
                    iterations += 1
                    microbatches_since_last_step = 0
                    
                    running_losses['vqvae/unique_codes'] = unique_codes.sum().item()
                    unique_codes = torch.zeros_like(unique_codes)
                    
                    model = patch_managers[0].model
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # random experiment
                    if False:
                        linear_layers = [module for module in model.modules() if isinstance(module, ThawingLinear)]
                        for layer in linear_layers:
                            target_grads = layer.linear.weight.grad
                            
                            with torch.no_grad():
                                random_positions = torch.rand_like(target_grads, requires_grad=False)
                                random_positions = random_positions * layer.trainable_mask # don't grab already disabled positions
                                random_positions = random_positions.reshape(-1).topk(k=1024).indices
                                random_grad_sampling = target_grads.reshape(-1)[random_positions]
                                cutoff_value = random_grad_sampling.abs().sort().values[-10]# 4%
                                
                                positions_to_disable_training_for = target_grads.abs() > cutoff_value
                                layer.thaw_time[positions_to_disable_training_for] = iterations+10
                                layer.trainable_mask = layer.thaw_time <= iterations
                        
                    result = take_step([m.optimizer for m in patch_managers], [m.model for m in patch_managers], [m.scheduler for m in patch_managers])
                    
                    if not result:
                        print("Failed to take step")
                    
   
                
                # Logging
                if iterations % 10 == 0 and microbatches_since_last_step == 0: #10
                    if skip_log:
                        skip_log = False
                        for m in patch_managers:
                            loss_totals[m.name] = 0
                            output_images = []
                        for loss_name in running_losses:
                            running_losses[loss_name] = []
                        continue
                    
                    
                    if writer is None:
                        writer = SummaryWriter(f'runs/{prefix}')

                    print(f"Epoch: {epoch}, Iteration: {iterations}, microbatches: {total_microbatches}, Microbatches per iteration:{microbatches_per_step}")
                    writer.add_scalar("Training/global/microbatches_per_step", microbatches_per_step, iterations)
                    writer.add_scalar("Training/global/total_microbatches", total_microbatches, iterations)
                    

                    
                    for loss_name in running_losses:
                        loss = torch.tensor(running_losses[loss_name]).float().mean()
                        if loss != 0 and math.isfinite(loss):
                            writer.add_scalar(f"Training/{loss_name}", loss, iterations)
                            print(f"Running Loss {loss_name}:\t{loss:.3f}")
                    
                    for m in patch_managers:
                        gen_current_lr = m.optimizer.param_groups[0]['lr']
                        

                        label = f"Training/{m.name}"
                        writer.add_scalar(f"{label}/Learning Rate", gen_current_lr, iterations)

                        print(f"Current LRs:\t({gen_current_lr:.6f}) -\t{m.name}")
                    
                    for loss_name in running_losses:
                        running_losses[loss_name] = []


                if microbatches_since_last_step == 0 and start_time is not None:
                    if datetime.now().timestamp()- start_time > 3600:
                        print("SAVING!!")
                        microbatches_at_save = total_microbatches
                        for m in patch_managers:
                            m.save(epoch, iterations, total_microbatches, seen_ids, microbatches_per_step)
                        start_time = datetime.now().timestamp()
            
            print("WOW!  YOU FINISHED AN EPOCH!!")
            seen_ids = set()
            dataset.update_seen_images(seen_ids)
        except KeyboardInterrupt:
            if (total_microbatches-microbatches_at_save) > 10000:
                print("Interrupted, saving model...")
                for m in patch_managers:
                    m.save(epoch, iterations, total_microbatches, seen_ids, microbatches_per_step)
            import pdb
            pdb.set_trace()
            writer.close()
                