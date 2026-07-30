'''
Went from a batch size of 200 to a batch size of 64 (200 seems like overkill)
GAN

1 generator, 1 discriminator, discriminator trained to 85% accuracy

64x64 patches

generator:
    input:
        - noise
        - coarse decoded image
    output:
        - fake
    loss:
        diversity loss between fakes
        discriminator needs to predict fake as real

discriminator:
    input:
        - coarse decoded image
        - fake image
    
    output:
        - mean pooled prediction of whether it is real or fake
    loss:
        - fake needs to be predicted as fake
        - real needs to be predicted as real
        
Training Loop:
    - discriminator trains until it is of sufficient accuracy
    - generator trains until discriminator no longer of sufficient accuracy
        
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
        hidden_size=512,
        intermediate_size=2048,
        num_layers=8,
        patch_size=8,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size=intermediate_size
        self.num_layers = num_layers
        self.patch_size = patch_size
        
class FusedSwiGLU2D(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        
        self.fused_proj = nn.Conv2d(hidden_size, 2 * intermediate_size, kernel_size=1, bias=False)
        self.down_proj = nn.Conv2d(intermediate_size, hidden_size, kernel_size=1, bias=False)

    def forward(self, x):
        gate, up = self.fused_proj(x).chunk(2, dim=1)
        return self.down_proj(F.silu(gate) * up)

class RMSNorm2D(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: B x C x H x W, normalize over C (dim=1)
        orig_dtype = x.dtype
        x = x.float()

        rms = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        x = (x * rms).to(orig_dtype)

        return x * self.weight.view(1, -1, 1, 1)
        
class UpscaleBlock(nn.Module):
    def __init__(self, input_dim, output_dim, intermediate_dim):
        super().__init__()
        
        self.preconv_norm = RMSNorm2D(input_dim)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False)
        
        self.premlp_norm = RMSNorm2D(output_dim)
        self.mlp = FusedSwiGLU2D(output_dim, intermediate_dim)
    
    def forward(self, x):
        x = self.preconv_norm(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        
        residual = x
        x = self.premlp_norm(x)
        x = self.mlp(x)
        
        return residual + x
        
        
class RefinementBlock(nn.Module):
    def __init__(self, input_dim, intermediate_dim):
        super().__init__()        
        self.preconv_norm = RMSNorm2D(input_dim)
        self.conv = nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, bias=False)
        self.premlp_norm = RMSNorm2D(input_dim)
        self.mlp = FusedSwiGLU2D(input_dim, intermediate_dim)
        
    def forward(self, x):
        res = x
        x = self.preconv_norm(x)
        x = self.conv(x)
        x = res + x
        res = x
        x = self.premlp_norm(x)
        x = self.mlp(x)
        x = res + x
        return x
        

        
        
class GeneratorCNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.unshuffle = nn.PixelUnshuffle(downscale_factor=config.patch_size)
        self.shuffle = nn.PixelShuffle(upscale_factor=config.patch_size)
        
        self.patch_proj = nn.Conv2d(config.patch_size*config.patch_size*3, config.hidden_size, kernel_size=1, bias=False)
        
        self.noise_mlp = FusedSwiGLU2D(config.hidden_size, config.intermediate_size)
        
        self.layers = nn.ModuleList([RefinementBlock(config.hidden_size, config.intermediate_size) for _ in range(config.num_layers)])
        
        
        self.upscale_block_0 = UpscaleBlock(512, 256, 1024) #64x64 -> 128x128
        self.refinement_block_0_a = RefinementBlock(256, 1024)
        self.refinement_block_0_b = RefinementBlock(256, 1024)
        self.upscale_block_1 = UpscaleBlock(256, 128, 512) #128x128 -> 256x256
        self.refinement_block_1_a = RefinementBlock(128, 512)
        self.refinement_block_1_b = RefinementBlock(128, 512)
        self.upscale_block_2 = UpscaleBlock(128, 64, 256) #256x256 -> 512x512
        self.refinement_block_2_a = RefinementBlock(64, 256)
        self.refinement_block_2_b = RefinementBlock(64, 256)
        
        self.final_norm = RMSNorm2D(64)
        self.final_out = nn.Conv2d(64,3,kernel_size=1,bias=False)
        
    @torch.compile
    def forward(self, x):
        # patchify images
        x = self.unshuffle(x)
        x = self.patch_proj(x)
        x = x + self.noise_mlp(torch.randn_like(x))
        for layer in self.layers:
            x = layer(x)
            
        x = self.upscale_block_0(x)
        x = self.refinement_block_0_a(x)
        x = self.refinement_block_0_b(x)
        x = self.upscale_block_1(x)
        x = self.refinement_block_1_a(x)
        x = self.refinement_block_1_b(x)
        x = self.upscale_block_2(x)
        x = self.refinement_block_2_a(x)
        x = self.refinement_block_2_b(x)
        
        x = self.final_norm(x)
        x = self.final_out(x)
        x = 2.5 * torch.tanh(x)
        return x

class DownscaleBlock(nn.Module):
    def __init__(self, input_dim, output_dim, intermediate_dim):
        super().__init__()
        
        self.preconv_norm = RMSNorm2D(input_dim)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.premlp_norm = RMSNorm2D(output_dim)
        self.mlp = FusedSwiGLU2D(output_dim, intermediate_dim)
    
    def forward(self, x):
        x = self.preconv_norm(x)
        x = self.conv(x)
        
        residual = x
        x = self.premlp_norm(x)
        x = self.mlp(x)
        
        return residual + x
        
class DiscriminatorCNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.unshuffle = nn.PixelUnshuffle(downscale_factor=config.patch_size)
        
        # 6x512x512 -> 64x512x512
        self.initial_proj = nn.Conv2d(6, 64, kernel_size=1, padding=0, bias=False)
        self.downscale_block_0 = DownscaleBlock(64, 128, 512) #512x512 -> 256x256
        self.downscale_block_1 = DownscaleBlock(128, 256, 1024) #256x256 -> 128x128
        self.downscale_block_2 = DownscaleBlock(256, config.hidden_size, 2048) #128x128 -> 64x64
                
        self.layers = nn.ModuleList([RefinementBlock(config.hidden_size, config.intermediate_size) for _ in range(config.num_layers)])
        
        self.final_norm = RMSNorm2D(config.hidden_size)
        self.final_out = nn.Conv2d(config.hidden_size, 1, kernel_size=1, bias=False)
    
    @torch.compile
    def forward(self, coarse_image, image):
        combined = torch.cat([coarse_image, image], dim=1)
        x = self.initial_proj(combined)
        x = self.downscale_block_0(x)
        x = self.downscale_block_1(x)
        x = self.downscale_block_2(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        x = self.final_out(x)
        x = x.mean(dim=(1,2,3)) # B,
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
        
        if model_type == 'generator_cnn':
            self.model = GeneratorCNN(config).to(device)
        elif model_type == 'discriminator_cnn':
            self.model = DiscriminatorCNN(config).to(device)
        else:
            raise ValueError("Unknown model_type passed to PatchManager!")
        
        self.model = self.model.to(device)
        self.model.train()
        
        self.reset_optimizer()
    
    def reset_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.5, 0.999),
            weight_decay=5e-5
        )
        #self.optimizer = schedulefree.AdamWScheduleFree(self.model.parameters(), lr=self.lr, weight_decay=5e-5)
        self.scheduler = get_warmup_scheduler(self.optimizer, 100, self.lr)
        #self.optimizer.train()
        self.step_count = 0
        self.is_in_good_state = False
        self.reset_gan_state()
    
    def take_step_safely(self):
        result = self.verify_step()
        if not result:
            self.reset_grads()
            return False
        
        self.take_step()
        return True
    
    def verify_step(self):
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                if not torch.all(torch.isfinite(p.grad)):
                    return False
        return True
    
    def reset_grads(self):
        self.optimizer.zero_grad()
    
    def take_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.step_count += 1
    
    def save(self, epoch, steps, total_microbatches, seen_ids, microbatch_size):
        """
        Save the state of the model, optimizer, and scheduler to a file.
        """
        save_dir = f'/mnt/f/models_trained/{self.label}_{steps}/'
        os.makedirs(save_dir, exist_ok=True)
        resulting_filepath = os.path.join(
            save_dir,
            f'{self.label}_{self.name}_epoch_{epoch}_step_{steps}_{self.model_type}.pth'
        )
        
        #self.optimizer.eval()
        torch.save({
            'epoch': epoch,
            'local_steps': self.step_count,
            'steps': steps,
            'total_microbatches': total_microbatches,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict' : self.scheduler.state_dict(),
            'seen_ids': seen_ids,
            'microbatch_size': microbatch_size
        }, resulting_filepath)
        #self.optimizer.train()
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
            #self.optimizer.train()
        else:
            print("RESETTING OPTIMIZER STATES DUE TO DIFFERENT PARAMS...")
        
        local_steps = checkpoint['local_steps']
        epoch = checkpoint['epoch']
        steps = checkpoint['steps']
        total_microbatches = checkpoint['total_microbatches']
        seen_ids = checkpoint['seen_ids']
        microbatch_size = checkpoint['microbatch_size']
        
        print(f"Successfully loaded model from {filepath}")
        self.step_count = local_steps
        #self.optimizer.train()
        
        return epoch, steps, total_microbatches, seen_ids, microbatch_size
        
    def reset_gan_state(self):
        self.is_in_good_state = False
        self.correct_evals_real = 0
        self.correct_evals_fake = 0
        self.total_evals_real = 0
        self.total_evals_fake = 0
        
    def record_eval_fake(self, logits, threshold=0.5):
        correct = (logits < -threshold)
        
        self.correct_evals_fake += correct.sum().item()
        self.total_evals_fake += correct.numel()

    def record_eval_real(self, logits, threshold=0.5):
        correct = logits > threshold
        
        self.correct_evals_real += correct.sum().item()
        self.total_evals_real += correct.numel()

    def update_good_state(self, acc_threshold=0.85):
        passed_fake = True
        passed_real = True
        
        if self.total_evals_fake > 0:
            passed_fake = (self.correct_evals_fake / self.total_evals_fake) > acc_threshold
        if self.total_evals_real > 0:
            passed_real = (self.correct_evals_real / self.total_evals_real) > acc_threshold
        
        self.reset_gan_state()
        self.is_in_good_state = passed_fake and passed_real
        
    
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
    save_image(final_image.clamp(0,1), f'output_images/progress_{label}_{iterations}.png', normalize=False)
    
        

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
        
def load_autoencoder(device):
    print("Loading pretrained autoencoder...")
    import stage_1_autoencoder_simple_ae_cnn_attn as autoencoder
    
    model_config = autoencoder.ModelConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_encoder_layers=6,
        lut_codebook_size=1024,
        bottleneck_dim=128,
        num_decoder_layers=6,
        num_attention_heads=16,
        input_patch_size=32,
        output_patch_size=32,
        num_positions=256,
    )
    
    autoencoder_manager = autoencoder.PatchManager(model_config, 0.00001, device, 'celeb_16x16', model_type='ae_cnn_attn', label='')
    autoencoder_manager.load('/mnt/f/models_trained_final/FINAL_039028f3_celeb_16x16_epoch_8_step_12060_ae_cnn_attn.pth')
    autoencoder_manager.optimizer.eval()
    return autoencoder_manager.model
    
def autoencode_image_using_autoencoder(image, autoencoder_model):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            _, latents = autoencoder_model(image)
            return latents

            
def decode_latents(latents, autoencoder_model):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            combined = torch.stack(latents)
            combined = combined.reshape(-1, *combined.size()[2:])
            all_decoded = autoencoder_model.decoder(combined)
            all_decoded = all_decoded.reshape(len(latents), latents[0].size(0), *all_decoded.size()[1:])
            return all_decoded
            
def load_latent_vocabulary_generator(device):
    print("Loading pretrained latent vocabulary generator...")
    import stage_2_latent_generator_16x16_mask_git_iterative_big as vocabulary_generator
    
    model_config = vocabulary_generator.ModelConfig(
        hidden_size=1024,
        intermediate_size=4096,
        lut_codebook_size=1024,
        lut_dropout=0.8,
        lut_duplication_rate=2,
        num_full_layers=12,
        num_noise_embedding_layers=8,
        bottleneck_latent_dim=128,
        num_attention_heads=16,
        num_positions=32*32,
        total_guesses=16,
    )
    
    vocabulary_manager = vocabulary_generator.PatchManager(model_config, 0.00001, device, 'celeb_16x16', model_type='latent_generator', label='')
    vocabulary_manager.load('/mnt/f/models_trained_final/FINAL_d650fa63_celeb_16x16_epoch_5_step_4167_latent_generator.pth')
    vocabulary_manager.optimizer.eval()
    return vocabulary_manager.model
    
def predict_coarse_latents(targets_16x16, vocabulary_model):
    inputs_current = torch.zeros_like(targets_16x16)
    
    for i in range(3):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                # G,B,C,H,W
                predictions_16x16 = vocabulary_model.forward_16(inputs_current.to(torch.bfloat16))
                                        
            
                multiplier = 10
                losses_16x16 = F.mse_loss(input=predictions_16x16*multiplier, target=targets_16x16[None].expand(*predictions_16x16.size())*multiplier, reduction='none')
                losses_16x16 = losses_16x16.mean(dim=2, keepdim=True)
                good_selection = losses_16x16.argmin(dim=0,keepdim=True)
                random_selection = (torch.rand_like(good_selection, dtype=torch.float)*16).floor().to(torch.long)
                cutoffs = torch.rand((samples_per_microbatch,),device=device) * 0.1
                valid_data_mask = torch.rand_like(good_selection, dtype=torch.float) > cutoffs[None,:,None,None,None] # 90%-100% closest chosen, cutoff per batch
                actual_selection = good_selection*valid_data_mask + random_selection*(~valid_data_mask) # indices
                best_predictions = predictions_16x16.gather(index=actual_selection.repeat(1,1,predictions_16x16.size(2),1,1),dim=0).squeeze(0).detach()
                inputs_current = best_predictions
    
    return inputs_current

def scramble_and_combine(fake_image_a, all_inputs):
    sorted_inputs = torch.stack([fake_image_a, all_inputs])
    which_slot_first = (torch.rand((all_inputs.size(0),),device=all_inputs.device) > 0.5).to(torch.long)
    which_slot_second = 1-which_slot_first
    
    slot_a = sorted_inputs.gather(index=which_slot_first[None,:,None,None,None].expand(1,*sorted_inputs.size()[1:]), dim=0).squeeze(dim=0)
    slot_b = sorted_inputs.gather(index=which_slot_second[None,:,None,None,None].expand(1,*sorted_inputs.size()[1:]), dim=0).squeeze(dim=0)
    
    # slot_a contains [1,0,0,1]
    # slot_b contains [0,1,1,0]
    # we want to get [1,1,1,1]
    # [1,1,1,1] is at 0,1,1,0 (slot_b)
    # we want to get [0,0,0,0]
    # [0,0,0,0] is at [1,0,0,1] (slot_a)
    
    scrambled_inputs = torch.cat([slot_a,slot_b],dim=1)
    return sorted_inputs, scrambled_inputs, which_slot_first, which_slot_second
                        
if __name__ == "__main__":
    torch._dynamo.config.cache_size_limit = 1024
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    prefix = generate_unique_prefix()
    
    image_size = 512
    patch_size=8
    writer = None
    dataset = FFHQDataset('/mnt/f/ffhq_diffusion/ffhq512',device)
    
    # might be too high (?)
    # limits ultimate quality
    learning_rate_g = 0.00005
    learning_rate_d = 0.00005
    
    gen_layers = 12

    generator_config = ModelConfig(
        hidden_size=512,
        intermediate_size=2048,
        num_layers=gen_layers,
        patch_size=patch_size,
    )
    
    celeb_generator = PatchManager(generator_config, learning_rate_g, device, f'gen_{gen_layers}', model_type='generator_cnn', label=prefix)
    
    all_generators = [celeb_generator]
    
    all_discriminators = []
    for i in range(12,13,2):
        discriminator_config = ModelConfig(
            hidden_size=512,
            intermediate_size=2048,
            num_layers=i,
            patch_size=patch_size,
        )
        
        celeb_discriminator = PatchManager(discriminator_config, learning_rate_d, device, f'disc_{i}', model_type='discriminator_cnn', label=prefix)
        all_discriminators.append(celeb_discriminator)
        
    # Hyperparameters etc.
    epochs = 10000
    samples_per_microbatch = 2
    base_grad_clip = 1.0
    
    patch_managers = all_generators + all_discriminators
    

    
    
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

    if False:
        load_prefix = '8f22e28e'
        load_epoch = 14
        load_step = 15434
        load_dir = f'/mnt/f/models_trained/{load_prefix}_{load_step}/'

        for m in patch_managers:
            filepath = os.path.join(
                load_dir,
                f'{load_prefix}_{m.name}_epoch_{load_epoch}_step_{load_step}_{m.model_type}.pth'
            )
            epoch_initial, iterations, total_microbatches, seen_ids, microbatches_per_step = m.load(filepath)

        microbatches_at_save = total_microbatches
        skip_log = False
    
    microbatches_per_step = 64 // samples_per_microbatch
        
        
    dataset.update_seen_images(seen_ids)
    data_loader = DataLoader(dataset, batch_size=samples_per_microbatch, shuffle=True, num_workers=0, drop_last=True)
    print("Dataset size:", len(dataset))
    print("Samples per microbatch:", samples_per_microbatch)
    
    microbatches_since_last_step = 0
    
    start_time = None
    print("Prefix:",prefix)
    
    autoencoder_model = load_autoencoder(device)
    vocabulary_generator_model = load_latent_vocabulary_generator(device)
    torch.cuda.empty_cache()
    
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
                
                # this block creates coarse conditioning (represents predicted latent from stage 3)
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        latents = autoencode_image_using_autoencoder(all_inputs, autoencoder_model) # 4,128,16,16; [-1 to 1]
                        coarse_latents = predict_coarse_latents(latents, vocabulary_generator_model)
                        
                        coarse_image = autoencoder_model.decoder(coarse_latents)
                        coarse_image = denormalize_image(coarse_image)
                        coarse_image = coarse_image.clamp(0,1)
                        coarse_image = normalize_image(coarse_image)
                        
                # determine what we're training (generators or discriminators)
                training_generator = all(d.is_in_good_state for d in all_discriminators)
                if microbatches_since_last_step == 0:
                    for d in all_discriminators:
                        running_losses[f'state/{d.name}_trained'].append(0.0 if d.is_in_good_state else 1.0)
                    
                    running_losses[f'state/generator_trained'].append(1.0 if training_generator else 0.0)
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    with torch.set_grad_enabled(training_generator):
                        fake_image_a = celeb_generator.model(coarse_image)
                        fake_image_b = celeb_generator.model(coarse_image)
                    if training_generator:
                        diff = (fake_image_a - fake_image_b).abs().mean(dim=(1,2,3))
                        # average pixel diff between generations should be roughly 0.2 (-2.5 - 2.5, or 10/255 per channel)
                        # The diff can be allocated anywhere (aka high diff localized patch, rather than low diff noise everywhere), it's just that the average change should be 0.2
                        diversity_loss = 50*(((diff - 0.2)/0.2)**2).mean()
                        running_losses[f'generator/diversity_loss'].append( diversity_loss.item())
                        
                        # stabilize color
                        downscale_amount = 512 // 8
                        downscaled_fake = fake_image_a.reshape(fake_image_a.size(0), fake_image_a.size(1), fake_image_a.size(2) // downscale_amount, downscale_amount, fake_image_a.size(3) // downscale_amount, downscale_amount).mean(dim=(3,5), keepdim=False)
                        downscaled_real = all_inputs.reshape(all_inputs.size(0), all_inputs.size(1), all_inputs.size(2) // downscale_amount, downscale_amount, all_inputs.size(3) // downscale_amount, downscale_amount).mean(dim=(3,5), keepdim=False)
                        
                        # HIGH(!), almost invariant requirement that color not drift, but since it's downscaled, model can generate plenty of detail within each patch
                        # basically defines allowable amount of drift around target value, and we don't want drift at this coarse statistical level pretty much at all
                        # this really only makes sense because the conditioning provides most of this signal already
                        # in some sense this represents the limit of the 'creativity' of the GAN; the lower this multiplier, the higher the allowed creativity
                        # The discriminator doesn't, in theory, need this at all, but it helps clamp down on the churn and I believe likely results in faster training
                        color_loss = F.mse_loss(downscaled_fake, downscaled_real)*50.0
                        running_losses[f'generator/color_loss'].append(color_loss.item())
                        
                        sum_of_discriminator_losses = 0
                        for discriminator in all_discriminators:
                            fake_logits = discriminator.model(coarse_image, fake_image_a)
                            g_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
                            
                            discriminator.record_eval_fake(fake_logits.detach())
                                                            
                            sum_of_discriminator_losses += g_loss
                            running_losses[f'generator/vs_{discriminator.name}'].append(g_loss.item())
                            
                        total_generator_loss = ((sum_of_discriminator_losses / len(all_discriminators)) + color_loss + diversity_loss) / microbatches_per_step
                        total_generator_loss.backward()
                    else:
                            
                        for discriminator in all_discriminators:
                            if discriminator.is_in_good_state:
                                continue
                            
                            real_logits = discriminator.model(coarse_image, all_inputs)
                            fake_logits = discriminator.model(coarse_image, fake_image_a)

                            d_loss = (
                                F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                                + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
                            )
                            
                            #d_loss = F.softplus(real_logits-fake_logits).mean()
                            discriminator.record_eval_real(real_logits.detach())
                            discriminator.record_eval_fake(fake_logits.detach())
                                    
                            running_losses[f'discriminator/{discriminator.name}'].append(d_loss.item())
                            
                            total_discriminator_loss = d_loss / microbatches_per_step
                            total_discriminator_loss.backward()
                
                outputting_images = iterations % 20 == 0 and microbatches_since_last_step == 0
                
                total_microbatches += 1
                microbatches_since_last_step += 1
                if start_time is None:
                    start_time = datetime.now().timestamp()
                        
                
                if outputting_images:
                    with torch.no_grad():
                        images = [denormalize_image(coarse_image), denormalize_image(fake_image_a).clamp(0,1),denormalize_image(fake_image_b).clamp(0,1),denormalize_image(all_inputs)]
                        #images = [denormalize_image(coarse_image), denormalize_image(fake_image_a).clamp(0,1), denormalize_image(fake_image_b).clamp(0,1)]
                        #diff = ((images[1] - images[2]).abs() * 5.0).clamp(0, 1)
                        #images.append(diff)
                        
                        save_progressive_growth_images(images, iterations, label=f'_training_{prefix}', max_images=2, scale_factor=2)

                            
                        
                    
                    
                if microbatches_since_last_step >= microbatches_per_step:
                    iterations += 1
                    microbatches_since_last_step = 0
                    
                    

                    if training_generator:
                        torch.nn.utils.clip_grad_norm_(celeb_generator.model.parameters(), max_norm=base_grad_clip)
                        result = celeb_generator.take_step_safely()
                        if not result:
                            print("Generator step failed (non-finite grads)")

                        # discs accumulated grads only as conduits; discard those.
                        # Do NOT blanket-invalidate: re-evaluate GOOD/BAD from the
                        # evals recorded against this window's generator. Discs that
                        # still separate confidently stay GOOD (and stay untrained);
                        # discs the generator fooled drop to BAD and get trained.
                        
                        for d in all_discriminators:
                            d.reset_grads()
                            d.update_good_state()
                    else:
                        for d in all_discriminators:
                            if d.is_in_good_state:
                                continue
                            torch.nn.utils.clip_grad_norm_(d.model.parameters(), max_norm=base_grad_clip)
                            result = d.take_step_safely()
                            if not result:
                                print(f"{d.name} step failed (non-finite grads)")
                            d.update_good_state()
                    
   
                
                # Logging
                if iterations % 10 == 0 and microbatches_since_last_step == 0: #10
                    
                    if writer is None:
                        writer = SummaryWriter(f'runs2/{prefix}')

                    print(f"Epoch: {epoch}, Iteration: {iterations}, microbatches: {total_microbatches}, Microbatches per iteration:{microbatches_per_step}")
                    writer.add_scalar("Training/global/microbatches_per_step", microbatches_per_step, iterations)
                    writer.add_scalar("Training/global/total_microbatches", total_microbatches, iterations)
                    
                    for discriminator in all_discriminators:
                        writer.add_scalar(f"Training/{discriminator.name}/step_count", discriminator.step_count, iterations)
                    

                    
                    for loss_name in sorted(running_losses):
                        loss = torch.tensor(running_losses[loss_name]).float().mean()
                        if math.isfinite(loss):
                            writer.add_scalar(f"Training/{loss_name}", loss, iterations)
                            print(f"Running Loss {loss_name}:\t{loss:.3f}")
                    
                    for m in patch_managers:
                        gen_current_lr = m.optimizer.param_groups[0]['lr']
                        
                        label = f"Training/{m.name}"
                        writer.add_scalar(f"{label}/Learning Rate", gen_current_lr, iterations)
                    for m in patch_managers:
                        writer.add_scalar(f"Training/{m.name}/step_count", m.step_count, iterations)

                        print(f"Step Count:\t({m.step_count}) -\t{m.name}")
                    
                    running_losses = defaultdict(list)


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
            import pdb
            pdb.set_trace()
            writer.close()
                