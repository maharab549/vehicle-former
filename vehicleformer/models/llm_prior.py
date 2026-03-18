"""
LLM Policy Prior — Contribution 3
===================================
Uses a quantized LLM (Phi-3-mini) with LoRA fine-tuning as a policy prior
for KL-regularized SAC. The LLM provides a "common-sense" action
distribution π_LLM(a|s) that regularizes the SAC policy:

    π* = argmax E[R(s,a)] - λ · D_KL(π || π_LLM)

This avoids catastrophically poor exploration in novel network states
by grounding the policy in the LLM's pretrained knowledge of network
management heuristics.

Architecture:
    graph_embedding → ProjectionIn → LLM (LoRA) → ProjectionOut → μ, log_σ
    
The LLM processes a "virtual token" representation of the graph state,
not text. This is a continuous-valued embedding injection approach.

Target journal: IEEE Journal on Selected Areas in Communications (JSAC)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict
import numpy as np
import os


class LLMPolicyPrior(nn.Module):
    """
    LLM-based policy prior using Phi-3-mini with 4-bit quantization + LoRA.
    
    The LLM acts as an informed prior: its pretrained representations
    encode general knowledge about network optimization which we
    leverage via LoRA fine-tuning on our ICV task.
    """

    def __init__(self, cfg: dict, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        llm_cfg = cfg['llm_prior']
        self.kl_weight = llm_cfg['kl_weight']
        self.update_freq = llm_cfg['update_frequency']

        emb_dim = cfg['hetgnn']['embedding_dim']  # 256
        act_dim = cfg['action']['action_dim']       # 6

        self.llm = None
        self.tokenizer = None
        self._llm_hidden_dim = None

        # These projection layers are initialized after LLM loads
        self._emb_dim = emb_dim
        self._act_dim = act_dim

        # Action output heads (initialized after LLM loads)
        self.action_mean = None
        self.action_log_std = None
        self.input_proj = None

    def load_model(self, model_name: Optional[str] = None):
        """
        Download and load the quantized LLM with LoRA adapters.
        Call this explicitly after __init__ to control when the
        large download happens.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import get_peft_model, LoraConfig, TaskType

        llm_cfg = self.cfg['llm_prior']
        model_name = model_name or llm_cfg['model_name']

        print(f"  ⏳ Loading LLM: {model_name} ...")

        # Quantization config — 4-bit NF4 for minimal VRAM
        quant_config = None
        if llm_cfg.get('load_in_4bit', True):
            try:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except Exception as e:
                print(f"  ⚠ bitsandbytes 4-bit not available ({e}), using float16")
                quant_config = None

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        load_kwargs = {
            "dtype": torch.float16,
            "attn_implementation": "eager",
        }
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = {"": self.device}

        self.llm = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        # Get LLM hidden dimension
        if hasattr(self.llm.config, 'hidden_size'):
            self._llm_hidden_dim = self.llm.config.hidden_size
        else:
            self._llm_hidden_dim = 3072  # Phi-3-mini default

        print(f"  ✓ LLM loaded: {model_name} (hidden_dim={self._llm_hidden_dim})")

        # Apply LoRA
        lora_config = LoraConfig(
            r=llm_cfg['lora_rank'],
            lora_alpha=llm_cfg['lora_alpha'],
            lora_dropout=llm_cfg['lora_dropout'],
            target_modules=llm_cfg['target_modules'],
            task_type=TaskType.CAUSAL_LM,
        )
        self.llm = get_peft_model(self.llm, lora_config)
        trainable, total = self.llm.get_nb_trainable_parameters()
        print(f"  ✓ LoRA applied: {trainable:,} trainable / {total:,} total params")

        # Freeze base model, only LoRA adapters train
        self.llm.print_trainable_parameters()

        # Build projection layers
        self.input_proj = nn.Sequential(
            nn.Linear(self._emb_dim, self._llm_hidden_dim),
            nn.LayerNorm(self._llm_hidden_dim),
            nn.GELU(),
        ).to(self.device)

        self.action_mean = nn.Sequential(
            nn.Linear(self._llm_hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, self._act_dim),
            nn.Tanh(),
        ).to(self.device)

        self.action_log_std = nn.Sequential(
            nn.Linear(self._llm_hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, self._act_dim),
        ).to(self.device)

        print(f"  ✓ Projection layers built (emb={self._emb_dim} → LLM={self._llm_hidden_dim} → act={self._act_dim})")

    def _get_llm_features(self, graph_embedding: Tensor) -> Tensor:
        """
        Project graph embedding into LLM hidden space, run through
        a single forward pass, and extract the output representation.
        
        Instead of text tokens, we inject the graph state as a
        continuous "virtual token" into the LLM's embedding space.
        """
        B = graph_embedding.shape[0]

        # Project graph embedding → LLM hidden dim
        virtual_tokens = self.input_proj(graph_embedding)  # (B, llm_hidden)
        virtual_tokens = virtual_tokens.unsqueeze(1)        # (B, 1, llm_hidden)

        # Create a minimal inputs_embeds sequence (1 virtual token)
        # Run through the LLM backbone
        with torch.amp.autocast('cuda', dtype=torch.float16):
            outputs = self.llm(
                inputs_embeds=virtual_tokens.to(self.llm.dtype),
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract last hidden state from the virtual token position
        last_hidden = outputs.hidden_states[-1]  # (B, 1, hidden_dim)
        features = last_hidden[:, -1, :]          # (B, hidden_dim)

        return features.float()  # back to float32 for action heads

    def forward(
        self, graph_embedding: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute LLM prior action distribution parameters.
        
        Args:
            graph_embedding: (B, emb_dim) from HetGNN
            
        Returns:
            mean:    (B, act_dim) — prior action mean
            log_std: (B, act_dim) — prior action log-std
        """
        features = self._get_llm_features(graph_embedding)
        mean = self.action_mean(features)
        log_std = self.action_log_std(features).clamp(-5, 2)
        return mean, log_std

    def get_log_prob(
        self, graph_embedding: Tensor, actions: Tensor
    ) -> Tensor:
        """
        Compute log π_LLM(a|s) for the KL divergence term.
        
        Args:
            graph_embedding: (B, emb_dim)
            actions:         (B, act_dim)  — actions in [-1, 1]
            
        Returns:
            log_prob: (B, 1)
        """
        mean, log_std = self.forward(graph_embedding)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)

        # Compute log-prob (handling tanh squashing)
        # actions are already tanh-squashed, so un-squash for Gaussian log-prob
        # atanh(y) = 0.5 * log((1+y)/(1-y))
        actions_clamped = actions.clamp(-0.999, 0.999)
        pre_tanh = torch.atanh(actions_clamped)
        log_prob = dist.log_prob(pre_tanh)
        # Jacobian correction for tanh squashing
        log_prob -= torch.log(1 - actions_clamped.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # (B, 1)

        return log_prob

    def get_trainable_parameters(self):
        """Return parameters that should be optimized."""
        params = []
        if self.input_proj is not None:
            params.extend(self.input_proj.parameters())
        if self.action_mean is not None:
            params.extend(self.action_mean.parameters())
        if self.action_log_std is not None:
            params.extend(self.action_log_std.parameters())
        # LoRA parameters
        if self.llm is not None:
            for name, param in self.llm.named_parameters():
                if param.requires_grad:
                    params.append(param)
        return params

    def save(self, path: str):
        """Save LoRA adapters and projection layers."""
        state = {
            "input_proj": self.input_proj.state_dict() if self.input_proj else None,
            "action_mean": self.action_mean.state_dict() if self.action_mean else None,
            "action_log_std": self.action_log_std.state_dict() if self.action_log_std else None,
        }
        torch.save(state, path)
        # Save LoRA adapters separately
        if self.llm is not None:
            lora_dir = path.replace(".pt", "_lora")
            self.llm.save_pretrained(lora_dir)
        print(f"  ✓ LLM prior saved to {path}")

    def load(self, path: str):
        """Load projection layers (LoRA adapters loaded via PEFT)."""
        state = torch.load(path, map_location=self.device)
        if self.input_proj and state.get("input_proj"):
            self.input_proj.load_state_dict(state["input_proj"])
        if self.action_mean and state.get("action_mean"):
            self.action_mean.load_state_dict(state["action_mean"])
        if self.action_log_std and state.get("action_log_std"):
            self.action_log_std.load_state_dict(state["action_log_std"])
