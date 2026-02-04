import os
from contextlib import contextmanager
from pathlib import Path
from typing import cast
import torch
from torch import Tensor
from torch import nn as nn
import numpy as np

@contextmanager
def temporary_change_cwd(destination):
    current_dir = os.getcwd()
    try:
        os.chdir(destination)
        yield
    finally:
        os.chdir(current_dir)

class TMR_Wrapper(nn.Module):
    def __init__(self, model_dir: Path):
        super().__init__()

        # self.clip_dim_layer = nn.Linear(256,512)
        self.concat_clip_dim_layer = nn.Linear(512,512)

        import sys

        from hydra.utils import instantiate

        from third_packages.TMR.src.config import read_config
        from third_packages.TMR.src.data.collate import collate_x_dict
        from third_packages.TMR.src.data.text import TokenEmbeddings
        from third_packages.TMR.src.load import load_model_from_cfg
        from third_packages.TMR.src.model.tmr import TMR

        self.collate_x_dict = collate_x_dict

        # model_dir = Path("third_packages/TMR/models/tmr_humanml3d_guoh3dfeats").resolve()
        cfg = read_config(str(model_dir)) # model_dir: 'third_packages/TMR/models/tmr_humanml3d_guoh3dfeats'
        # print(cfg)
        cfg.run_dir = model_dir.relative_to("third_packages/TMR")
        # cfg.run_dir = model_dir.relative_to(os.getcwd())
        # cfg.run_dir = model_dir

        with temporary_change_cwd("third_packages/TMR"):
            sys.path.append("third_packages/TMR")

            cfg.data.text_to_token_emb['_target_'] = 'third_packages.TMR.src.data.text.TokenEmbeddings'

            self.text_model: TokenEmbeddings = instantiate(cfg.data.text_to_token_emb)
            self.tmr_model: TMR = load_model_from_cfg(cfg)

            if model_dir.parts[-1] == "tmr_humanml3d_guoh3dfeats":
                self.mean = torch.load("stats/humanml3d/guoh3dfeats/mean.pt")
                self.std = torch.load("stats/humanml3d/guoh3dfeats/std.pt")
            elif model_dir.parts[-1] == "tmr_kitml_guoh3dfeats":
                self.mean = torch.load("stats/kitml/guoh3dfeats/mean.pt")
                self.std = torch.load("stats/kitml/guoh3dfeats/std.pt")
            else:
                self.mean = torch.load(f"stats/humanml3d_{model_dir.parts[-1]}/guoh3dfeats/mean.pt")
                self.std = torch.load(f"stats/humanml3d_{model_dir.parts[-1]}/guoh3dfeats/std.pt")

        self.latent_dim = 512

        self.humanml3d_all = torch.from_numpy(np.load('third_packages/TMR/models/tmr_humanml3d_guoh3dfeats/latents/humanml3d_all.npy')).cuda()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def normalize(self, motion):
        return (motion - self.mean.to(motion.device)) / self.std.to(motion.device)

    
    def encode_motion(self, motions):

        device = next(self.parameters()).device

        motion_x_dicts = [{"x": motion, "length": len(motion)} for motion in motions]

        motion_x_dict = self.collate_x_dict(motion_x_dicts, device=str(device))

        latent = cast(Tensor, self.tmr_model.encode(motion_x_dict, sample_mean=True))

        # latent = self.clip_dim_layer(latent)

        return latent

    def encode_text(self, texts):
        device = next(self.parameters()).device
        text_x_dict = self.collate_x_dict(self.text_model(texts), device=str(device))
        latent = cast(Tensor, self.tmr_model.encode(text_x_dict, sample_mean=True)) # (128, 256), (128, L, 256)
        # latent = self.clip_dim_layer(latent) # (128, 512)
        return latent

    def forward(self, x):
        if isinstance(x[0], Tensor):
            return self.encode_motion(cast(list[Tensor], x))
        else:
            return self.encode_text(cast(list[str], x))

    def encode_text_motion(self, texts):
        device = next(self.parameters()).device
        text_x_dict = self.collate_x_dict(self.text_model(texts), device=str(device))
        latent = cast(Tensor, self.tmr_model.encode(text_x_dict, sample_mean=True))

        '''
        ra_latent = []

        for l in latent:
            # cos_text_to_motion = []
            
            # for h in self.humanml3d_all:
                # cos_text_to_motion.append(self.cos(l.unsqueeze(0), h.unsqueeze(0)))

            cos_text_to_motion = self.cos(l.unsqueeze(0), self.humanml3d_all)

            top_n = 1
            sorted_indices = sorted(range(len(cos_text_to_motion)), key=lambda i: cos_text_to_motion[i], reverse=True)
            top_indices = sorted_indices[:top_n]

            ra_latent.append(self.humanml3d_all[top_indices[0]].unsqueeze(0))

        ra_latent = torch.cat(ra_latent, dim=0)
        '''

        humanml3d_all_expanded = self.humanml3d_all.unsqueeze(0)  # Shape: (1, N, D)
        latent_expanded = latent.unsqueeze(1)  # Shape: (M, 1, D)

        cos_text_to_motion = self.cos(latent_expanded, humanml3d_all_expanded)  # Shape: (M, N)
        # top_indices = torch.argmax(cos_text_to_motion, dim=1)  # Shape: (M,)
        # ra_latent = self.humanml3d_all[top_indices]

        top_values, top_indices = torch.topk(cos_text_to_motion, k=3, dim=1)  # Shape: (M, 3)

        random_indices = torch.randint(0, 3, (top_indices.size(0),), device=top_indices.device)
        chosen_indices = top_indices[torch.arange(top_indices.size(0)), random_indices]  # Shape: (M,)

        ra_latent = self.humanml3d_all[chosen_indices]
        

        latent = torch.concat([latent, ra_latent], dim=-1)
        latent = self.concat_clip_dim_layer(latent)

        return latent
