from copy import deepcopy

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
import warnings
import cv2


import PIL
import torchvision.transforms as transforms
import my_transforms as my_transforms
from time import time
import logging
import operators


def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (384, 384, 3)# default(32, 32, 3)  VIT(384, 384, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0), 
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms


class Continual_MAE(nn.Module):
    def __init__(self, model, optimizer, hogs=None, projections=None, mask_token=None, hog_ratio=1, block_size=16, mask_ratio=0.5, steps=1, episodic=False, mt_alpha=0.99, rst_m=0.1, ap=0.9):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "Continual_MAE requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        
        self.model_state, self.optimizer_state, self.model_temp, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.transform = get_tta_transforms()    
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap
        self.block_size = block_size
        self.mask_ratio = mask_ratio

        self.hogs = hogs
        self.projections = projections
        self.mask_token = mask_token
        self.hog_ratio = hog_ratio
        self.mse_func = nn.MSELoss(reduction="mean")

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.optimizer, self.block_size, self.mask_ratio, self.mask_token)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_temp, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def _get_hog_label_2d(self, input_frames, output_mask, block_size):
        # input_frames, B C H W
        feat_size = input_frames.shape[-1] // block_size # num of the window ; ori = [1, H//patch_size, W//patch_size]
        tmp_hog = self.hogs(input_frames).flatten(1, 2)
        unfold_size = tmp_hog.shape[-1] // feat_size
        tmp_hog = (
            tmp_hog.permute(0, 2, 3, 1)
            .unfold(1, unfold_size, unfold_size)
            .unfold(2, unfold_size, unfold_size)
            .flatten(1, 2)
            .flatten(2)
        )
        tmp_hog = tmp_hog[output_mask]
        return tmp_hog

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, optimizer, block_size, mask_ratio, mask_token):
            
        #n times forward to compute uncertainty
        n_outputs = []
        n_forward = 10
        for i in range(n_forward):
            _, outputs_  = self.model_temp(self.transform(x), return_norm=True)
            outputs_ = outputs_[:,1:,:].mean(dim=2)
            n_outputs.append(outputs_)
        stacked_outputs_ = torch.stack(n_outputs, dim=0)
        variance = torch.var(stacked_outputs_, dim=0)
        sorted_data, sorted_indices = torch.sort(variance, dim=1, descending=True)
        top_k = int(sorted_indices.shape[1]*mask_ratio)
        masked_indices = sorted_indices[:, :top_k]
        mask_chosed = torch.zeros_like(sorted_data)
        mask_chosed.scatter_(1, masked_indices, 1)
        
        outputs, outputs_hog = self.model(x, mask_token, mask_chosed, return_norm=True)

        if self.hogs is not None:
            output_mask = mask_chosed.to(bool) #mask_chosed
            hog_preds = self.projections(outputs_hog[:,1:,:]) # cls token not need
            hog_preds = hog_preds[output_mask] 
            hog_labels = self._get_hog_label_2d(x, output_mask, block_size=block_size)
            hog_loss = self.mse_func(hog_preds, hog_labels) 
        

        outputs_temp = self.model_temp(x)

        loss_ori = (softmax_entropy(outputs, outputs_temp)).mean(0)
        if self.hogs is not None:
            loss = loss_ori + self.hog_ratio*hog_loss
            print("loss_ori:",loss_ori, "hog_loss:", hog_loss)
        else:
            loss = loss_ori
            print("loss_ori:",loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # # one model setting
        # for temp_param, param in zip(self.model_temp.parameters(), self.model.parameters()):
        #     temp_param.data[:] = param[:].data[:]
        # Stochastic restore
        if True:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<self.rst).float().cuda() 
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)
        # one model setting
        for temp_param, param in zip(self.model_temp.parameters(), self.model.parameters()):
            temp_param.data[:] = param[:].data[:]
        return outputs_temp


@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def collect_params(model, projections=None, mask_token=None):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True: #isinstance(m, nn.BatchNorm2d): collect all 
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    if projections is not None:
            for np, p in projections.named_parameters():
                    params.append(p)
                    names.append(f"{np}")
    if mask_token is not None:
        for np, p in mask_token.named_parameters():
                params.append(p)
                names.append(f"{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    model_temp = deepcopy(model)
    for param in model_temp.parameters():
        param.detach_()
    return model_state, optimizer_state, model_temp, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"