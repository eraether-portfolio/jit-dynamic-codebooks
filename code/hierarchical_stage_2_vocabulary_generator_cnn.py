'''
Upscales and refines, starting from 4x4 (all 1s) -> 8x8 -> 16x16 -> 32x32 -> 64x64

10 refinement steps total
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

import schedulefree
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint

class ModelConfig():
    def __init__(
        self,
        hidden_size=512,
        intermediate_size=2048,
        num_deep_layers=4,
        mlps_per_deep_layer=2,
        num_noise_embedding_layers=6,
        num_attention_heads=16,
        latent_dim=32,
        patch_size=8,
        total_guesses=8,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size=intermediate_size
        self.num_deep_layers = num_deep_layers
        self.mlps_per_deep_layer = mlps_per_deep_layer
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.total_guesses = total_guesses
        self.num_attention_heads = num_attention_heads
        self.num_noise_embedding_layers = num_noise_embedding_layers
        
class FusedSwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size, output_size=None):
        super().__init__()
        
        if output_size is None:
            output_size = hidden_size
        self.fused_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, output_size, bias=False)

    def forward(self, x):
        gate, up = self.fused_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
        
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
        
class MLPBlock(nn.Module):
    def __init__(self, input_dim, intermediate_dim):
        super().__init__()
        
        self.premlp_norm = RMSNorm2D(input_dim)
        self.mlp = FusedSwiGLU2D(input_dim, intermediate_dim)
    
    def forward(self, x):
        res = x
        x = self.premlp_norm(x)
        x = self.mlp(x)
        return res + x
     
class DeepRefinementBlock(nn.Module):
    def __init__(self, input_dim, intermediate_dim, num_refinements=2):
        super().__init__()
        self.preconv_norm = RMSNorm2D(input_dim)
        self.conv = nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, bias=False)
        self.layers = nn.ModuleList([MLPBlock(input_dim, intermediate_dim) for _ in range(num_refinements)])
    
    def forward(self, x):
        res = x
        x = self.preconv_norm(x)
        x = self.conv(x)
        x = res + x
        for layer in self.layers:
            x = layer(x)
        return x
        
class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self_attn = FastNonCausalAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
        )
        self.mlp = FusedSwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
        
class FastNonCausalAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Fused QKV projection
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        B, S, C = x.shape
        
        # 1. Fused linear projection + reshape
        # We project to [B, S, 3, NH, HD] then permute to [3, B, NH, S, HD]
        qkv = self.qkv_proj(x).view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2) # split into Q, K, V along the 3rd dimension
        
        # Transpose for SDPA: [B, NH, S, HD]
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]

        # 2. Modern Context Manager to force FlashAttention
        # This tells PyTorch: "Use FlashAttention or throw an error if impossible."
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                q, k, v, 
                is_causal=False,
                dropout_p=0.0
            )

        # 3. Output projection
        out = out.transpose(1, 2).reshape(B, S, C)
        return self.out_proj(out)
        
        
class LatentVocabularyGeneratorCNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.previous_latent_proj = nn.Conv2d(config.latent_dim, config.hidden_size, kernel_size=1, bias=False)
        
        self.layers = nn.ModuleList([DeepRefinementBlock(config.hidden_size, config.intermediate_size, config.mlps_per_deep_layer) for _ in range(config.num_deep_layers)])
        
        self.noise_embedding = FusedSwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            output_size=config.hidden_size
        )
        
        self.noise_layers = nn.ModuleList([TransformerDecoderLayer(config) for _ in range(config.num_noise_embedding_layers)])
        
        self.final_norm = nn.RMSNorm(config.hidden_size)
        self.final_out = nn.Linear(config.hidden_size,config.latent_dim,bias=False)
        
    @torch.compile
    def forward(self, previous_latents):
        B,C,H,W = previous_latents.size()
        x = self.previous_latent_proj(previous_latents)
        for layer in self.layers:
            x = layer(x)
        
        x = x.permute(0,2,3,1).reshape(x.size(0), x.size(2)*x.size(3), 1, x.size(1))
        x = x.repeat(1,1,self.config.total_guesses,1)
        x = x + self.noise_embedding(torch.randn_like(x))
        B,T,G,_ = x.size()
        x = x.reshape(x.size(0)*x.size(1), x.size(2), x.size(3))
        
        for noise_layer in self.noise_layers:
            x = noise_layer(x)
        
        x = self.final_norm(x)
        x = self.final_out(x)
        x = x.reshape(B,H,W,G,x.size(-1))
        x = x.permute(3,0,4,1,2) # G,B,C,H,W
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
            return (current_step+1.0)/num_warmup_steps
        return 1.0

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
        
        if model_type == 'latent_generator_cnn_hierarchical':
            self.model = LatentVocabularyGeneratorCNN(config).to(device)
        else:
            raise ValueError("Unknown model_type passed to PatchManager!")
        
        self.model = self.model.to(device)
        
        self.model.train()
        
        # Set up the optimizer and scheduler.
        self.reset_optimizer()
    
    def reset_optimizer(self):
        self.optimizer = schedulefree.AdamWScheduleFree(self.model.parameters(), lr=self.lr, weight_decay=5e-5)
        self.scheduler = get_warmup_scheduler(self.optimizer, 100, self.lr)
        self.optimizer.train()
        
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
    save_image(final_image, f'output_images/progress_{label}_{iterations}.png', normalize=False)
    
        

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
    print("Loading pretrained hierarchical autoencoder...")
    import hierarchical_stage_1_autoencoder_mlp as autoencoder
    
    model_config = autoencoder.ModelConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_encoder_layers=6,
        bottleneck_dim=64,
        num_decoder_layers=6,
        patch_size=8,
    )
    
    autoencoder_manager = autoencoder.PatchManager(model_config, 0.00001, device, 'celeb', model_type='ae_mlp_hierarchical', label='')
    autoencoder_manager.load('/mnt/f/ffhq_models_final/9e3e077d_celeb_epoch_4_step_1428_ae_mlp_hierarchical.pth')
    autoencoder_manager.optimizer.eval()
    return autoencoder_manager.model
    

if __name__ == "__main__":
    torch._dynamo.config.cache_size_limit = 1024
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    prefix = generate_unique_prefix()
    print("Prefix:",prefix)
    
    image_size = 512
    input_patch_size=8
    output_patch_size=8
    writer = None
    dataset = FFHQDataset('/mnt/f/ffhq_diffusion/ffhq512',device)
            
    learning_rate = 0.0003
    

    model_config = ModelConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_deep_layers=7,
        mlps_per_deep_layer=2,
        num_noise_embedding_layers=6,
        latent_dim=64,
        total_guesses=8,
    )
    
    
    
    latent_generator_manager = PatchManager(model_config, learning_rate, device, 'celeb', model_type='latent_generator_cnn_hierarchical', label=prefix)
    
    
    

   
    

    
    # Hyperparameters etc.
    epochs = 10000
    samples_per_microbatch = 1#16
    base_grad_clip = 1.0
    
    patch_managers = [latent_generator_manager]
    

    
    
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
    
    # allowed 1x downscale at step 2912
    if True:
        epoch_initial, iterations, total_microbatches, seen_ids, microbatches_per_step = latent_generator_manager.load('/mnt/f/models_trained/059ff6af_celeb_epoch_4_step_2912_latent_generator_cnn_hierarchical.pth')
                
        microbatches_at_save = total_microbatches
        skip_log = False
    microbatches_per_step = 100 // samples_per_microbatch
        
        
    dataset.update_seen_images(seen_ids)
    data_loader = DataLoader(dataset, batch_size=samples_per_microbatch, shuffle=True, num_workers=0)
    print("Dataset size:", len(dataset))
    print("Samples per microbatch:", samples_per_microbatch)
    
    microbatches_since_last_step = 0
    
    start_time = None
    
    autoencoder_model = load_autoencoder(device)
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
                
                downscale_factors = [16,8,4,2,1]
                latent_generator_model = patch_managers[0].model
                
                inputs_current = None
                
                for current_downscale_factor in downscale_factors:
                    current_image = all_inputs.reshape(all_inputs.size(0), all_inputs.size(1), all_inputs.size(2) // current_downscale_factor, current_downscale_factor, all_inputs.size(3) // current_downscale_factor, current_downscale_factor).mean(dim=(3,5), keepdim=False)
                    
                    
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        with torch.no_grad():
                            _, target_latents = autoencoder_model(current_image)
                            
                            if inputs_current is None:
                                # we start with ones only so that the model can figure out the absolute positioning of each token on iteration 0
                                inputs_current = torch.ones_like(target_latents)
                            else:
                                # 2x upsample
                                inputs_current = inputs_current.reshape(inputs_current.size(0), inputs_current.size(1), inputs_current.size(2), 1, inputs_current.size(3), 1).repeat(1,1,1,2,1,2)
                                inputs_current = inputs_current.reshape(*target_latents.size())
                        
                        num_refinements = 2
                        backprop_every_n = 2
                        # can potentially go for 3, but slower (!)
                        for refinement_iteration in range(num_refinements):
                            label = f"{current_downscale_factor}_{refinement_iteration}"
                            
                            backprop_this_step = (refinement_iteration % backprop_every_n) == (total_microbatches % backprop_every_n)
                            
                            with torch.set_grad_enabled(backprop_this_step):
                                predictions = latent_generator_model(inputs_current.to(torch.bfloat16))
                                multiplier = 10
                                losses = F.mse_loss(input=predictions*multiplier, target=target_latents[None].expand(*predictions.size())*multiplier, reduction='none')
                                
                                losses = losses.mean(dim=2, keepdim=True)

                                total_loss = losses.gather(index=losses.argmin(dim=0,keepdim=True),dim=0).mean()
                                # we only need overflow because we keep passing previous output to next input without BPTT
                                overflow = total_loss > 100.0
                        
                            with torch.no_grad():
                                good_selection = losses.argmin(dim=0,keepdim=True)
                                random_selection = (torch.rand_like(good_selection, dtype=torch.float)*model_config.total_guesses).floor().to(torch.long)
                                cutoffs = torch.rand((samples_per_microbatch,),device=device) * 0.1 # 0.0 - 0.1 (0% to 10%)
                                valid_data_mask = torch.rand_like(good_selection, dtype=torch.float) > cutoffs[None,:,None, None,None] # 5% on average
                                actual_selection = good_selection*valid_data_mask + random_selection*(~valid_data_mask)
                                best_predictions = predictions.gather(index=actual_selection.repeat(1,1,predictions.size(2),1,1),dim=0).squeeze(0).detach()
                                inputs_current = best_predictions
                                
                                #all_best_outputs.append(best_predictions)
                                if not overflow:
                                    running_losses[patch_managers[0].name+f'/total_loss'].append(total_loss.item())
                                    running_losses[patch_managers[0].name+f'/loss_{label}'].append(total_loss.item())
                                
                            if backprop_this_step:
                                if overflow:
                                    total_loss = 20*(total_loss/total_loss.detach())
                                    print("Overflow!!")
                                total_loss = total_loss / ( num_refinements / backprop_every_n)
                                scaled_total_loss = total_loss / microbatches_per_step
                                scaled_total_loss.backward()
                        
                outputting_images = False#iterations % 20 == 0 and microbatches_since_last_step == (backprop_every_n-1)

                    
                total_microbatches += 1
                microbatches_since_last_step += 1
                if start_time is None:
                    start_time = datetime.now().timestamp()
                
                if outputting_images:
                    with torch.no_grad():
                        max_images = 2
                        #all_best_outputs.append(targets_16x16)
                        outputs = decode_latents(all_best_outputs+all_worst_outputs, autoencoder_model)
                        images = []
                        for i in range(outputs.size(0)):
                            image = denormalize_image(outputs[i]).clamp(0,1)
                            images.append(image)
                        
                        save_progressive_growth_images(images, iterations, label=f'_training_{prefix}', max_images=max_images, scale_factor=8)
                        images = []
                        outputs = None

                            
                        
                    
                    
                if microbatches_since_last_step >= microbatches_per_step:
                    iterations += 1
                    microbatches_since_last_step = 0
                    
                    
                        
                    torch.nn.utils.clip_grad_norm_(patch_managers[0].model.parameters(), max_norm=1.0)
                    
                   
                    
                    result = take_step([m.optimizer for m in patch_managers], [m.model for m in patch_managers], [m.scheduler for m in patch_managers])
                    
                    if not result:
                        print("Failed to take step")
                    
   
                
                # Logging
                if iterations % 10 == 0 and microbatches_since_last_step == 0: #10
                    
                    if writer is None:
                        writer = SummaryWriter(f'runs2/{prefix}')

                    print(f"Epoch: {epoch}, Iteration: {iterations}, microbatches: {total_microbatches}, Microbatches per iteration:{microbatches_per_step}")
                    writer.add_scalar("Training/global/microbatches_per_step", microbatches_per_step, iterations)
                    writer.add_scalar("Training/global/total_microbatches", total_microbatches, iterations)
                    

                    
                    for loss_name in sorted(running_losses):
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
            import pdb
            pdb.set_trace()
            writer.close()
                