"""LLM prior over network selection with API, Ollama, and heuristic backends."""
from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional
from urllib import request

import numpy as np
import torch


class LLMPolicyPrior:
    """Constructs a categorical prior over networks from natural-language state prompts."""

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.llm_cfg = cfg["llm_prior"]
        self.cache_size = int(self.llm_cfg.get("cache_size", 10000))
        self.cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self.backend_name = "heuristic"
        self._openai_endpoint = self.llm_cfg.get("openai_endpoint", "https://api.openai.com/v1/chat/completions")
        self._ollama_endpoint = self.llm_cfg.get("ollama_endpoint", "http://localhost:11434/api/generate")

    def load_model(self, model_name: Optional[str] = None) -> None:
        """Configure the available backend without requiring heavyweight local models."""
        backend = self.llm_cfg.get("backend", "auto")
        if backend in {"auto", "openai"} and os.getenv("OPENAI_API_KEY"):
            self.backend_name = model_name or self.llm_cfg.get("openai_model", "gpt-4o-mini")
            return
        if backend in {"auto", "ollama"}:
            self.backend_name = model_name or self.llm_cfg.get("ollama_model", "llama3:8b")
            return
        self.backend_name = "heuristic"

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Compatibility shim for the existing trainer."""
        return []

    def save(self, path: str) -> None:
        """Persist cache and backend metadata to disk."""
        payload = {
            "backend_name": self.backend_name,
            "cache": list(self.cache.items()),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def load(self, path: str) -> None:
        """Restore cache and backend metadata from disk."""
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.backend_name = payload.get("backend_name", self.backend_name)
        self.cache = OrderedDict(payload.get("cache", []))

    def beta_at_step(self, step: int) -> float:
        """Return the configured adaptive KL weight at a given step."""
        beta0 = float(self.llm_cfg.get("beta_0", 1.0))
        decay = float(self.llm_cfg.get("decay_rate", 1.0e-5))
        beta_min = float(self.llm_cfg.get("beta_min", 0.01))
        return max(beta_min, beta0 * np.exp(-decay * step))

    def get_prior_probs(self, obs_batch: np.ndarray) -> np.ndarray:
        """Return a batch of network probabilities for the provided observations."""
        priors = [self._prior_for_observation(obs) for obs in obs_batch]
        return np.asarray(priors, dtype=np.float32)

    def _prior_for_observation(self, obs: np.ndarray) -> List[float]:
        state = self._structured_state(obs)
        key = self._state_hash(state)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        prompt = self._state_to_prompt(state)
        probabilities = self._query_backend(prompt, state)
        self.cache[key] = probabilities
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return probabilities

    def _structured_state(self, obs: np.ndarray) -> Dict[str, float]:
        vehicle_dim = int(self.cfg["observation"]["vehicle_dim"])
        max_vehicles = int(self.cfg["observation"]["max_vehicles"])
        max_rsus = int(self.cfg["observation"]["max_rsus"])
        max_bs = int(self.cfg["observation"]["max_base_stations"])
        rsu_dim = int(self.cfg["observation"]["rsu_dim"])
        bs_dim = int(self.cfg["observation"]["bs_dim"])
        vehicle_block = obs[: vehicle_dim * max_vehicles].reshape(max_vehicles, vehicle_dim)
        ego = vehicle_block[0]
        network_offset = vehicle_dim * max_vehicles + rsu_dim * max_rsus + bs_dim * max_bs
        network_obs = obs[network_offset:]
        return {
            "speed_kmh": float(max(ego[2], 0.0) * 50.0),
            "rssi_5g_dbm": float(ego[4] * 90.0 - 120.0),
            "rssi_v2x_dbm": float(ego[5] * 90.0 - 120.0),
            "sat_latency_ms": float(np.clip(ego[6], 0.0, 1.0) * 200.0),
            "edge_load": float(np.clip(ego[7], 0.0, 1.0)),
            "neighbors": int(round(np.clip(network_obs[5] * 10.0, 0.0, 10.0))),
            "deadline_remaining_ms": float(np.clip(ego[9], 0.0, 1.0) * 100.0),
        }

    @staticmethod
    def _state_hash(state: Dict[str, float]) -> str:
        """Discretize the state before hashing to keep cache reuse high."""
        discretized = {
            "speed": int(state["speed_kmh"] // 5),
            "g5": int(state["rssi_5g_dbm"] // 5),
            "v2x": int(state["rssi_v2x_dbm"] // 5),
            "sat": int(state["sat_latency_ms"] // 10),
            "load": int(state["edge_load"] * 10),
            "neighbors": int(state["neighbors"]),
            "deadline": int(state["deadline_remaining_ms"] // 10),
        }
        return hashlib.sha256(json.dumps(discretized, sort_keys=True).encode("utf-8")).hexdigest()

    @staticmethod
    def _signal_word(rssi_dbm: float) -> str:
        """Convert an RSSI measurement to a qualitative description."""
        if rssi_dbm > -75:
            return "strong"
        if rssi_dbm > -90:
            return "moderate"
        return "weak"

    def _state_to_prompt(self, state: Dict[str, float]) -> str:
        """Format the state as the requested natural-language prompt."""
        if state["deadline_remaining_ms"] <= 30.0:
            latency_requirement = "safety-critical"
        elif state["deadline_remaining_ms"] <= 60.0:
            latency_requirement = "delay-sensitive"
        else:
            latency_requirement = "throughput-oriented"
        v2x_clear = "clear" if state["neighbors"] <= 3 else "congested"
        sat_status = "available, low congestion" if state["sat_latency_ms"] < 90 else "available, elevated latency"
        return (
            f"Vehicle speed: {state['speed_kmh']:.0f} km/h. "
            f"5G signal: {state['rssi_5g_dbm']:.0f} dBm ({self._signal_word(state['rssi_5g_dbm'])}). "
            f"C-V2X channel: {v2x_clear}, {state['neighbors']} neighbors. "
            f"Satellite: {sat_status}. "
            f"Current latency requirement: {latency_requirement}. "
            "Which network should the vehicle select? Return JSON with keys 5g, cv2x, sat summing to 1."
        )

    def _query_backend(self, prompt: str, state: Dict[str, float]) -> List[float]:
        """Query the configured backend and fall back to the heuristic prior on failure."""
        backend = self.llm_cfg.get("backend", "auto")
        if backend in {"auto", "openai"} and os.getenv("OPENAI_API_KEY"):
            try:
                return self._openai_prior(prompt)
            except Exception:
                pass
        if backend in {"auto", "ollama"}:
            try:
                return self._ollama_prior(prompt)
            except Exception:
                pass
        return self._heuristic_prior(state)

    def _openai_prior(self, prompt: str) -> List[float]:
        """Query an OpenAI-compatible chat completion endpoint."""
        payload = {
            "model": self.llm_cfg.get("openai_model", "gpt-4o-mini"),
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You are a vehicular network expert. Output only JSON."},
                {"role": "user", "content": prompt},
            ],
        }
        req = request.Request(
            self._openai_endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "Content-Type": "application/json",
            },
        )
        with request.urlopen(req, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        return self._parse_prior_response(content)

    def _ollama_prior(self, prompt: str) -> List[float]:
        """Query a local Ollama server."""
        payload = {
            "model": self.llm_cfg.get("ollama_model", "llama3:8b"),
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
        req = request.Request(
            self._ollama_endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
        return self._parse_prior_response(body.get("response", "{}"))

    @staticmethod
    def _parse_prior_response(raw: str) -> List[float]:
        """Parse and normalize JSON network probabilities returned by the LLM."""
        data = json.loads(raw)
        probs = np.asarray([
            float(data.get("5g", data.get("g5", 0.34))),
            float(data.get("cv2x", data.get("c_v2x", 0.33))),
            float(data.get("sat", data.get("satellite", 0.33))),
        ], dtype=np.float32)
        probs = np.clip(probs, 1e-4, None)
        probs = probs / probs.sum()
        return probs.tolist()

    @staticmethod
    def _heuristic_prior(state: Dict[str, float]) -> List[float]:
        """Fallback prior when no LLM backend is available."""
        g5_score = 0.08 * (state["rssi_5g_dbm"] + 110.0) - 1.4 * state["edge_load"]
        v2x_score = 0.09 * (state["rssi_v2x_dbm"] + 105.0) - 0.12 * state["neighbors"]
        sat_score = 4.0 - 0.04 * state["sat_latency_ms"]
        if state["deadline_remaining_ms"] <= 30.0:
            g5_score += 0.9
            v2x_score += 0.6
            sat_score -= 0.8
        logits = np.asarray([g5_score, v2x_score, sat_score], dtype=np.float32)
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()
        return probs.tolist()


LLMPrior = LLMPolicyPrior"""
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
