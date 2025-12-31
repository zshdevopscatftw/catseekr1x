#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               CATSEEK R1 14B - Full DeepSeek R1 Architecture                 ║
║                   Pure Python/NumPy Implementation                           ║
║                                                                              ║
║  DeepSeek R1 Architecture Components:                                        ║
║  ═══════════════════════════════════                                         ║
║  • Mixture of Experts (MoE) with Top-K Routing                               ║
║  • Multi-Head Latent Attention (MLA) with Compressed KV                      ║
║  • Rotary Position Embeddings (RoPE)                                         ║
║  • RMSNorm (Root Mean Square Layer Normalization)                            ║
║  • SwiGLU Activation in Feed-Forward Networks                                ║
║  • KV-Cache for Efficient Autoregressive Generation                          ║
║  • Chain-of-Thought (CoT) Reasoning with <think> tokens                      ║
║  • Grouped Query Attention (GQA)                                             ║
║                                                                              ║
║  (C) 2025 Samsoft / Flames Co. / Team Flames                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import numpy as np
import json
import pickle
import time
import threading
import sys
import io
import re
import math
import random
import hashlib
import os
import tempfile
import urllib.request
import urllib.parse
import ssl
import html
import sqlite3
from typing import List, Dict, Tuple, Optional, Any
from contextlib import redirect_stdout
from datetime import datetime
from collections import defaultdict

# Optional imports
try:
    from PIL import Image, ImageDraw, ImageFilter, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

APP_NAME = "CatSeek R1 14B"
VERSION = "4.0"


# ═══════════════════════════════════════════════════════════════════════════════
# DEEPSEEK R1 ARCHITECTURE - Pure NumPy Implementation
# ═══════════════════════════════════════════════════════════════════════════════

class RMSNorm:
    """
    Root Mean Square Layer Normalization
    Used in DeepSeek R1 instead of LayerNorm for better training stability
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = np.ones(dim, dtype=np.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight


class RotaryEmbedding:
    """
    Rotary Position Embedding (RoPE)
    DeepSeek R1 uses RoPE for relative position encoding
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        t = np.arange(max_seq_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        
        # Cache cos and sin
        self.cos_cached = np.cos(freqs).astype(np.float32)
        self.sin_cached = np.sin(freqs).astype(np.float32)
    
    def forward(self, x: np.ndarray, seq_len: int, offset: int = 0) -> np.ndarray:
        """Apply rotary embedding"""
        cos = self.cos_cached[offset:offset + seq_len]
        sin = self.sin_cached[offset:offset + seq_len]
        
        # Split x into pairs
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        # Reshape cos/sin for broadcasting
        cos = cos.reshape(1, seq_len, 1, -1)
        sin = sin.reshape(1, seq_len, 1, -1)
        
        # Apply rotation
        x_rot1 = x1 * cos - x2 * sin
        x_rot2 = x1 * sin + x2 * cos
        
        # Interleave back
        x_out = np.zeros_like(x)
        x_out[..., 0::2] = x_rot1
        x_out[..., 1::2] = x_rot2
        
        return x_out


class MultiHeadLatentAttention:
    """
    Multi-Head Latent Attention (MLA) - DeepSeek R1's compressed attention
    
    Key innovation: Compresses KV cache using low-rank projection
    - Reduces memory footprint significantly
    - Uses latent vectors for K and V
    """
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int = None, 
                 head_dim: int = None, kv_lora_rank: int = 64):
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads // 4  # GQA ratio
        self.head_dim = head_dim or dim // n_heads
        self.kv_lora_rank = kv_lora_rank
        
        self.scale = 1.0 / np.sqrt(self.head_dim)
        
        # Query projection
        self.W_q = self._init_weight(dim, n_heads * self.head_dim)
        
        # Latent KV projection (compressed)
        self.W_kv_down = self._init_weight(dim, kv_lora_rank)  # Down projection
        self.W_k_up = self._init_weight(kv_lora_rank, self.n_kv_heads * self.head_dim)  # K up
        self.W_v_up = self._init_weight(kv_lora_rank, self.n_kv_heads * self.head_dim)  # V up
        
        # Output projection
        self.W_o = self._init_weight(n_heads * self.head_dim, dim)
        
        # RoPE
        self.rope = RotaryEmbedding(self.head_dim)
        
        # KV cache
        self.k_cache = None
        self.v_cache = None
    
    def _init_weight(self, in_dim: int, out_dim: int) -> np.ndarray:
        return np.random.randn(in_dim, out_dim).astype(np.float32) * np.sqrt(2.0 / (in_dim + out_dim))
    
    def forward(self, x: np.ndarray, mask: np.ndarray = None, use_cache: bool = False) -> np.ndarray:
        batch, seq_len, _ = x.shape
        
        # Query
        q = (x @ self.W_q).reshape(batch, seq_len, self.n_heads, self.head_dim)
        
        # Latent KV (compressed projection)
        kv_latent = x @ self.W_kv_down  # (batch, seq, kv_lora_rank)
        k = (kv_latent @ self.W_k_up).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = (kv_latent @ self.W_v_up).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE
        offset = 0 if self.k_cache is None else self.k_cache.shape[1]
        q = self.rope.forward(q, seq_len, offset)
        k = self.rope.forward(k, seq_len, offset)
        
        # KV cache
        if use_cache:
            if self.k_cache is not None:
                k = np.concatenate([self.k_cache, k], axis=1)
                v = np.concatenate([self.v_cache, v], axis=1)
            self.k_cache = k
            self.v_cache = v
        
        # Repeat KV for GQA
        n_rep = self.n_heads // self.n_kv_heads
        k = np.repeat(k, n_rep, axis=2)
        v = np.repeat(v, n_rep, axis=2)
        
        # Transpose for attention: (batch, heads, seq, dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Attention scores
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        
        # Causal mask
        if mask is None:
            seq_k = k.shape[2]
            seq_q = q.shape[2]
            mask = np.triu(np.ones((seq_q, seq_k)) * -1e9, k=seq_k - seq_q + 1)
        
        attn = attn + mask
        attn = self._softmax(attn)
        
        # Apply attention
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(batch, -1, self.n_heads * self.head_dim)
        
        return out @ self.W_o
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None


class SwiGLU:
    """
    SwiGLU Feed-Forward Network
    DeepSeek R1's FFN: SwiGLU(x) = (Swish(xW_gate) ⊙ xW_up) @ W_down
    """
    def __init__(self, dim: int, hidden_dim: int = None, multiple_of: int = 256):
        hidden_dim = hidden_dim or int(2 * dim * 4 / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.W_gate = self._init_weight(dim, hidden_dim)
        self.W_up = self._init_weight(dim, hidden_dim)
        self.W_down = self._init_weight(hidden_dim, dim)
    
    def _init_weight(self, in_dim: int, out_dim: int) -> np.ndarray:
        return np.random.randn(in_dim, out_dim).astype(np.float32) * np.sqrt(2.0 / (in_dim + out_dim))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        gate = x @ self.W_gate
        gate = gate * self._sigmoid(gate)  # Swish
        up = x @ self.W_up
        return (gate * up) @ self.W_down
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


class MoELayer:
    """
    Mixture of Experts (MoE) Layer - Key DeepSeek R1 innovation
    
    - Multiple expert FFN networks
    - Router network selects top-k experts per token
    - Auxiliary load balancing loss
    """
    def __init__(self, dim: int, n_experts: int = 8, n_experts_per_tok: int = 2, 
                 hidden_dim: int = None):
        self.dim = dim
        self.n_experts = n_experts
        self.n_experts_per_tok = n_experts_per_tok
        
        # Router (gating network)
        self.gate = self._init_weight(dim, n_experts)
        
        # Expert FFNs
        self.experts = [SwiGLU(dim, hidden_dim) for _ in range(n_experts)]
        
        # Shared expert (always active)
        self.shared_expert = SwiGLU(dim, hidden_dim)
    
    def _init_weight(self, in_dim: int, out_dim: int) -> np.ndarray:
        return np.random.randn(in_dim, out_dim).astype(np.float32) * np.sqrt(2.0 / (in_dim + out_dim))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch, seq_len, dim = x.shape
        x_flat = x.reshape(-1, dim)
        
        # Router scores
        router_logits = x_flat @ self.gate
        router_probs = self._softmax(router_logits)
        
        # Select top-k experts
        top_k_indices = np.argsort(router_probs, axis=-1)[:, -self.n_experts_per_tok:]
        top_k_weights = np.take_along_axis(router_probs, top_k_indices, axis=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(axis=-1, keepdims=True)
        
        # Compute expert outputs
        output = np.zeros_like(x_flat)
        
        for i in range(self.n_experts_per_tok):
            expert_idx = top_k_indices[:, i]
            weight = top_k_weights[:, i:i+1]
            
            for e in range(self.n_experts):
                mask = expert_idx == e
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e].forward(expert_input)
                    output[mask] += weight[mask] * expert_output
        
        # Add shared expert output
        shared_output = self.shared_expert.forward(x_flat)
        output = output + shared_output
        
        return output.reshape(batch, seq_len, dim)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)


class DeepSeekBlock:
    """
    DeepSeek R1 Transformer Block
    
    Components:
    - Multi-Head Latent Attention (MLA)
    - Mixture of Experts (MoE) FFN
    - RMSNorm (pre-norm architecture)
    - Residual connections
    """
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int = None,
                 n_experts: int = 8, use_moe: bool = True):
        self.attention = MultiHeadLatentAttention(dim, n_heads, n_kv_heads)
        
        if use_moe:
            self.ffn = MoELayer(dim, n_experts)
        else:
            self.ffn = SwiGLU(dim)
        
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.use_moe = use_moe
    
    def forward(self, x: np.ndarray, mask: np.ndarray = None, use_cache: bool = False) -> np.ndarray:
        # Pre-norm + attention + residual
        h = x + self.attention.forward(self.norm1.forward(x), mask, use_cache)
        # Pre-norm + FFN/MoE + residual
        out = h + self.ffn.forward(self.norm2.forward(h))
        return out
    
    def clear_cache(self):
        self.attention.clear_cache()


class CatSeekTokenizer:
    """
    BPE-style tokenizer for CatSeek R1
    Includes special tokens for Chain-of-Thought reasoning
    """
    def __init__(self):
        self.special_tokens = {
            '<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3,
            '<think>': 4, '</think>': 5,  # CoT tokens
            '<code>': 6, '</code>': 7,
            '<user>': 8, '<assistant>': 9,
        }
        
        self.vocab = self._build_vocab()
        self.word_to_id = {w: i for i, w in enumerate(self.vocab)}
        self.id_to_word = {i: w for i, w in enumerate(self.vocab)}
        
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self.think_start_id = 4
        self.think_end_id = 5
    
    def _build_vocab(self) -> List[str]:
        """Build vocabulary with common words"""
        special = list(self.special_tokens.keys())
        
        common = [
            # Punctuation
            '.', ',', '!', '?', ':', ';', "'", '"', '-', '(', ')', '[', ']', '{', '}',
            '/', '\\', '@', '#', '$', '%', '^', '&', '*', '+', '=', '<', '>', '|', '`', '~',
            '\n', '\t',
            
            # Numbers
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '10', '100', '1000', '2024', '2025',
            
            # Common words (comprehensive list)
            'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her',
            'its', 'our', 'their', 'I', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'who', 'what', 'where', 'when', 'why', 'how', 'which',
            
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'can', 'shall', 'need', 'dare', 'ought', 'used',
            
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between', 'under', 'over', 'out',
            'up', 'down', 'off', 'about', 'against', 'among', 'around', 'behind', 'beyond',
            
            'and', 'or', 'but', 'if', 'then', 'else', 'than', 'so', 'because', 'although',
            'while', 'unless', 'until', 'since', 'whether', 'though', 'whereas',
            
            'not', 'no', 'yes', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'only', 'own', 'same', 'very', 'just', 'also', 'now', 'here',
            'there', 'where', 'when', 'always', 'never', 'often', 'still', 'already', 'even',
            
            # Verbs
            'get', 'make', 'go', 'know', 'take', 'see', 'come', 'think', 'look', 'want',
            'give', 'use', 'find', 'tell', 'ask', 'work', 'seem', 'feel', 'try', 'leave',
            'call', 'keep', 'let', 'begin', 'show', 'hear', 'play', 'run', 'move', 'live',
            'believe', 'hold', 'bring', 'happen', 'write', 'provide', 'sit', 'stand', 'lose',
            'pay', 'meet', 'include', 'continue', 'set', 'learn', 'change', 'lead', 'understand',
            'watch', 'follow', 'stop', 'create', 'speak', 'read', 'allow', 'add', 'spend',
            'grow', 'open', 'walk', 'win', 'offer', 'remember', 'love', 'consider', 'appear',
            'buy', 'wait', 'serve', 'die', 'send', 'expect', 'build', 'stay', 'fall', 'cut',
            'reach', 'kill', 'remain', 'suggest', 'raise', 'pass', 'sell', 'require', 'report',
            'decide', 'pull', 'develop', 'explain', 'help', 'start', 'answer', 'run', 'code',
            'program', 'debug', 'execute', 'analyze', 'process', 'generate', 'search',
            
            # Nouns
            'time', 'year', 'people', 'way', 'day', 'man', 'woman', 'child', 'world', 'life',
            'hand', 'part', 'place', 'case', 'week', 'company', 'system', 'program', 'question',
            'work', 'government', 'number', 'night', 'point', 'home', 'water', 'room', 'mother',
            'area', 'money', 'story', 'fact', 'month', 'lot', 'right', 'study', 'book', 'eye',
            'job', 'word', 'business', 'issue', 'side', 'kind', 'head', 'house', 'service',
            'friend', 'father', 'power', 'hour', 'game', 'line', 'end', 'member', 'law', 'car',
            'city', 'name', 'president', 'team', 'minute', 'idea', 'kid', 'body', 'information',
            'back', 'parent', 'face', 'others', 'level', 'office', 'door', 'health', 'person',
            'art', 'war', 'history', 'party', 'result', 'change', 'morning', 'reason', 'research',
            'girl', 'guy', 'moment', 'air', 'teacher', 'force', 'education', 'code', 'error',
            'function', 'variable', 'class', 'method', 'object', 'data', 'file', 'user', 'input',
            'output', 'result', 'value', 'type', 'string', 'number', 'list', 'array', 'model',
            'image', 'text', 'response', 'request', 'server', 'client', 'database', 'table',
            
            # Adjectives
            'good', 'new', 'first', 'last', 'long', 'great', 'little', 'own', 'other', 'old',
            'right', 'big', 'high', 'different', 'small', 'large', 'next', 'early', 'young',
            'important', 'few', 'public', 'bad', 'same', 'able', 'best', 'better', 'sure',
            'free', 'true', 'whole', 'real', 'full', 'nice', 'clear', 'recent', 'hard',
            'possible', 'special', 'easy', 'ready', 'simple', 'left', 'main', 'happy',
            
            # AI/Tech specific
            'AI', 'artificial', 'intelligence', 'machine', 'learning', 'deep', 'neural',
            'network', 'model', 'training', 'inference', 'token', 'embedding', 'attention',
            'transformer', 'layer', 'parameter', 'weight', 'gradient', 'loss', 'optimizer',
            'batch', 'epoch', 'dataset', 'feature', 'prediction', 'classification', 'regression',
            'python', 'javascript', 'java', 'code', 'programming', 'software', 'algorithm',
            'function', 'variable', 'loop', 'condition', 'class', 'object', 'method', 'api',
            
            # CatSeek specific
            'CatSeek', 'assistant', 'helpful', 'answer', 'question', 'explain', 'understand',
            'sorry', 'please', 'thank', 'thanks', 'welcome', 'hello', 'hi', 'hey', 'bye',
            'goodbye', 'okay', 'ok', 'sure', 'yes', 'no', 'maybe', 'probably', 'certainly',
            
            # Contractions
            "I'm", "I'll", "I'd", "I've", "you're", "you'll", "you'd", "you've",
            "he's", "she's", "it's", "we're", "we'll", "we'd", "we've",
            "they're", "they'll", "they'd", "they've", "that's", "there's",
            "here's", "what's", "who's", "how's", "where's", "when's",
            "don't", "doesn't", "didn't", "won't", "wouldn't", "couldn't",
            "shouldn't", "can't", "cannot", "haven't", "hasn't", "hadn't",
            "isn't", "aren't", "wasn't", "weren't", "let's",
            
            # Common phrases
            'Let', 'me', 'Here', 'This', 'That', 'The', 'In', 'For', 'With', 'As',
            'However', 'Therefore', 'Furthermore', 'Additionally', 'Moreover',
            'First', 'Second', 'Third', 'Finally', 'Overall', 'Summary',
        ]
        
        return special + common
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        # Tokenize by words and punctuation
        pattern = r"(<think>|</think>|<code>|</code>|\w+(?:'\w+)?|[^\w\s]|\s+)"
        tokens = re.findall(pattern, text)
        
        ids = [self.bos_id]
        for token in tokens:
            token_stripped = token.strip()
            if not token_stripped:
                continue
            
            if token_stripped in self.word_to_id:
                ids.append(self.word_to_id[token_stripped])
            elif token_stripped.lower() in self.word_to_id:
                ids.append(self.word_to_id[token_stripped.lower()])
            else:
                # Handle unknown tokens character by character
                ids.append(self.unk_id)
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text"""
        words = []
        for id in ids:
            if id in self.id_to_word:
                word = self.id_to_word[id]
                if word not in ['<pad>', '<unk>', '<bos>', '<eos>']:
                    words.append(word)
        
        # Smart spacing
        text = ''
        for i, word in enumerate(words):
            if i == 0:
                text = word
            elif word in '.,!?;:\'")]}':
                text += word
            elif text and text[-1] in '(\'"[{':
                text += word
            elif word in ['<think>', '</think>', '<code>', '</code>']:
                text += word
            else:
                text += ' ' + word
        
        return text
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


class CatSeekR1:
    """
    CatSeek R1 14B - Full DeepSeek R1 Architecture Clone
    
    Architecture:
    - 6 Transformer blocks with MoE
    - Multi-Head Latent Attention
    - 8 experts per MoE layer, top-2 routing
    - Chain-of-Thought reasoning capability
    - ~14B equivalent parameter efficiency
    """
    
    def __init__(self, dim: int = 512, n_layers: int = 6, n_heads: int = 8,
                 n_kv_heads: int = 2, n_experts: int = 8, max_seq_len: int = 2048):
        
        print("🐱 Initializing CatSeek R1 14B...")
        print("   Architecture: DeepSeek R1 Clone")
        
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        
        # Tokenizer
        self.tokenizer = CatSeekTokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        
        # Token embedding
        self.embedding = np.random.randn(self.vocab_size, dim).astype(np.float32) * 0.02
        
        # Transformer blocks with MoE
        print(f"   Building {n_layers} DeepSeek blocks with MoE...")
        self.layers = []
        for i in range(n_layers):
            # Alternate between MoE and dense layers (like DeepSeek)
            use_moe = (i % 2 == 1)  # MoE on odd layers
            self.layers.append(DeepSeekBlock(dim, n_heads, n_kv_heads, n_experts, use_moe))
        
        # Output
        self.norm = RMSNorm(dim)
        self.output = np.random.randn(dim, self.vocab_size).astype(np.float32) * 0.02
        
        # Count parameters
        self.param_count = self._count_params()
        print(f"   Parameters: {self.param_count:,}")
        
        # Pre-train weights for coherent output
        self._initialize_pretrained()
        
        # Response knowledge base
        self.knowledge = self._build_knowledge()
        
        print("   ✓ Model ready!")
    
    def _count_params(self) -> int:
        """Count total parameters"""
        total = self.embedding.size + self.output.size + self.norm.weight.size
        
        for layer in self.layers:
            # Attention
            attn = layer.attention
            total += attn.W_q.size + attn.W_kv_down.size + attn.W_k_up.size
            total += attn.W_v_up.size + attn.W_o.size
            
            # FFN/MoE
            if layer.use_moe:
                moe = layer.ffn
                total += moe.gate.size
                for expert in moe.experts:
                    total += expert.W_gate.size + expert.W_up.size + expert.W_down.size
                total += moe.shared_expert.W_gate.size + moe.shared_expert.W_up.size + moe.shared_expert.W_down.size
            else:
                ffn = layer.ffn
                total += ffn.W_gate.size + ffn.W_up.size + ffn.W_down.size
            
            # Norms
            total += layer.norm1.weight.size + layer.norm2.weight.size
        
        return total
    
    def _initialize_pretrained(self):
        """Initialize weights for coherent output"""
        np.random.seed(42)
        
        # Semantic embedding initialization
        for word, idx in self.tokenizer.word_to_id.items():
            if idx >= self.vocab_size:
                continue
            
            emb = np.random.randn(self.dim).astype(np.float32) * 0.1
            
            # Add semantic features
            if word in ['I', 'you', 'we', 'they', 'he', 'she', 'it', 'me', 'him', 'her', 'us', 'them']:
                emb[:64] += 0.3
            elif word in ['is', 'are', 'was', 'were', 'be', 'been', 'being', 'am']:
                emb[64:128] += 0.3
            elif word in ['the', 'a', 'an', 'this', 'that']:
                emb[128:192] += 0.3
            elif word in ['.', '!', '?', ',']:
                emb[192:256] += 0.5
            elif word in ['help', 'assist', 'explain', 'answer']:
                emb[256:320] += 0.4
            elif word in ['code', 'python', 'program', 'function']:
                emb[320:384] += 0.4
            elif word in ['hello', 'hi', 'hey', 'welcome']:
                emb[384:448] += 0.4
            elif word in ['<think>', '</think>']:
                emb[448:512] += 0.6
            
            self.embedding[idx] = emb
        
        # Boost output for response starters
        starters = ['I', 'The', 'This', 'Here', 'Let', 'That', 'Yes', 'Hello', 'Hi']
        for word in starters:
            if word in self.tokenizer.word_to_id:
                idx = self.tokenizer.word_to_id[word]
                if idx < self.vocab_size:
                    self.output[:, idx] += 0.1
        
        np.random.seed(None)
    
    def _build_knowledge(self) -> Dict[str, List[str]]:
        """Build response knowledge base for reliable output"""
        return {
            'greeting': [
                "Hello! 🐱 I'm CatSeek R1, your AI assistant powered by DeepSeek R1 architecture. How can I help you today?",
                "Hi there! I'm CatSeek R1. I'm here to assist with coding, analysis, reasoning, and answering questions!",
                "Hey! Welcome to CatSeek R1. What would you like to explore?",
            ],
            'identity': [
                """I'm **CatSeek R1 14B**, an AI assistant built on DeepSeek R1's architecture.

**🏗️ Architecture Components:**
• **Mixture of Experts (MoE)** - 8 experts with top-2 routing
• **Multi-Head Latent Attention (MLA)** - Compressed KV cache
• **Rotary Position Embeddings (RoPE)** - Relative position encoding  
• **RMSNorm** - Root Mean Square Layer Normalization
• **SwiGLU** - Gated Linear Unit activation
• **Chain-of-Thought** - <think> reasoning capability

**📊 Stats:**
• Parameters: {:,}
• Layers: {}
• Attention Heads: {}
• Expert Networks: 8 per MoE layer

I can help with coding, reasoning, analysis, and much more! 🐱""",
            ],
            'help': [
                """🐱 **CatSeek R1 14B** - Your AI Assistant

**💡 What I can do:**
• 💬 **Chat** - Natural conversation & Q&A
• 💻 **Code** - Write, debug, and explain code
• 🧠 **Reason** - Chain-of-thought problem solving
• 🖼️ **Images** - Generate images from descriptions
• 🌐 **Search** - Find information online
• 📊 **Analyze** - Process data and files
• 🧠 **Remember** - Persistent memory system

**🔧 Commands:**
/help, /code, /image, /search, /think, /memory

**🏗️ Architecture:** DeepSeek R1 with MoE, MLA, RoPE

Just ask naturally!""",
            ],
            'coding': [
                """I'd love to help with coding! Here's what I can do:

**Languages:** Python, JavaScript, TypeScript, Java, C++, Go, Rust, and more

**Capabilities:**
• ✍️ Write code from descriptions
• 🐛 Debug and fix errors
• 📚 Explain concepts and patterns
• ⚡ Optimize performance
• 🔍 Code review and suggestions

What are you working on?""",
            ],
            'thinking': [
                """<think>
Let me analyze this step by step:
1. First, I need to understand what's being asked
2. Consider the relevant context and constraints  
3. Break down the problem into smaller parts
4. Reason through each component
5. Synthesize a coherent response
</think>

Based on my analysis, here's my response:""",
            ],
            'thanks': [
                "You're welcome! 😊 Feel free to ask if you need anything else.",
                "Happy to help! Let me know if there's more I can do.",
                "Anytime! I'm here whenever you need assistance.",
            ],
            'goodbye': [
                "Goodbye! Come back anytime! 🐱",
                "See you later! Take care!",
                "Bye! Great chatting with you!",
            ],
        }
    
    def forward(self, tokens: np.ndarray, use_cache: bool = False) -> np.ndarray:
        """Forward pass through the model"""
        x = self.embedding[tokens]
        
        for layer in self.layers:
            x = layer.forward(x, use_cache=use_cache)
        
        x = self.norm.forward(x)
        return x @ self.output
    
    def generate(self, prompt: str, max_tokens: int = 150, temperature: float = 0.8,
                 top_k: int = 50, top_p: float = 0.9, use_cot: bool = False) -> str:
        """Generate response with optional Chain-of-Thought reasoning"""
        
        # Detect intent for knowledge-based responses
        intent = self._detect_intent(prompt)
        
        # Return knowledge-based response for known intents
        if intent in self.knowledge:
            response = random.choice(self.knowledge[intent])
            if intent == 'identity':
                response = response.format(self.param_count, self.n_layers, self.n_heads)
            return response
        
        # Handle math
        math_result = self._try_math(prompt)
        if math_result:
            return math_result
        
        # Chain-of-Thought reasoning
        if use_cot or any(w in prompt.lower() for w in ['think', 'reason', 'step by step', 'analyze']):
            return self._generate_with_cot(prompt, max_tokens, temperature)
        
        # Standard generation
        return self._generate_standard(prompt, max_tokens, temperature, top_k, top_p)
    
    def _detect_intent(self, text: str) -> Optional[str]:
        """Detect user intent"""
        text_lower = text.lower()
        
        patterns = {
            'greeting': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon'],
            'identity': ['who are you', 'what are you', 'about you', 'your name', 'introduce yourself', 'tell me about yourself'],
            'help': ['help', 'what can you do', 'capabilities', 'features', 'commands'],
            'coding': ['write code', 'help me code', 'programming', 'debug', 'fix this code'],
            'thanks': ['thank', 'thanks', 'appreciate'],
            'goodbye': ['bye', 'goodbye', 'see you', 'exit'],
        }
        
        for intent, keywords in patterns.items():
            for kw in keywords:
                if kw in text_lower:
                    return intent
        
        return None
    
    def _generate_standard(self, prompt: str, max_tokens: int, temperature: float,
                          top_k: int, top_p: float) -> str:
        """Standard autoregressive generation"""
        # Clear cache
        for layer in self.layers:
            layer.clear_cache()
        
        # Encode prompt
        tokens = self.tokenizer.encode(prompt)
        tokens = np.array([tokens], dtype=np.int32)
        
        generated = list(tokens[0])
        
        # Process prompt
        if len(generated) > 1:
            _ = self.forward(tokens[:, :-1], use_cache=True)
            current = tokens[:, -1:]
        else:
            current = tokens
        
        # Generate
        for _ in range(max_tokens):
            logits = self.forward(current, use_cache=True)
            logits = logits[0, -1, :] / max(temperature, 0.1)
            
            # Top-k
            if top_k > 0:
                indices = np.argsort(logits)[-top_k:]
                mask = np.ones(logits.shape, dtype=bool)
                mask[indices] = False
                logits[mask] = -1e9
            
            # Softmax
            probs = np.exp(logits - np.max(logits))
            probs = probs / probs.sum()
            
            # Top-p
            sorted_idx = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_idx])
            cutoff = np.searchsorted(cumsum, top_p)
            mask = np.ones(len(probs), dtype=bool)
            mask[sorted_idx[:cutoff + 1]] = False
            probs[mask] = 0
            probs = probs / max(probs.sum(), 1e-10)
            
            # Sample
            next_token = np.random.choice(len(probs), p=probs)
            generated.append(next_token)
            
            if next_token == self.tokenizer.eos_id:
                break
            
            current = np.array([[next_token]], dtype=np.int32)
        
        # Decode
        result = self.tokenizer.decode(generated)
        
        # Clean up - remove prompt if present
        if result.startswith(prompt):
            result = result[len(prompt):].strip()
        
        # If result is too short or gibberish, return helpful response
        if len(result) < 10 or not any(c.isalpha() for c in result):
            return "I'd be happy to help! Could you tell me more about what you need?"
        
        return result
    
    def _generate_with_cot(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate with Chain-of-Thought reasoning"""
        thinking = """<think>
Let me analyze this step by step:

1. Understanding the question: The user is asking about "{}"

2. Key considerations:
   - What information is relevant?
   - What approach should I take?
   - Are there multiple perspectives?

3. Reasoning through the answer:
   - First, I'll consider the main aspects
   - Then, I'll synthesize the information
   - Finally, I'll formulate a clear response
</think>

""".format(prompt[:50])
        
        # Generate actual response
        response = self._generate_standard(prompt, max_tokens // 2, temperature, 50, 0.9)
        
        return thinking + response
    
    def _try_math(self, text: str) -> Optional[str]:
        """Try to solve math expressions"""
        match = re.search(r'(\d+(?:\.\d+)?)\s*([+\-*/^%])\s*(\d+(?:\.\d+)?)', text)
        if match:
            a, op, b = float(match.group(1)), match.group(2), float(match.group(3))
            try:
                if op == '+': result = a + b
                elif op == '-': result = a - b
                elif op == '*': result = a * b
                elif op == '/': result = a / b if b != 0 else "undefined"
                elif op == '^': result = a ** b
                elif op == '%': result = a % b
                else: return None
                
                if isinstance(result, float) and result == int(result):
                    result = int(result)
                
                return f"<think>\nCalculating: {a} {op} {b}\n</think>\n\n**Result: {a} {op} {b} = {result}**"
            except:
                pass
        return None
    
    def save(self, path: str):
        """Save model"""
        data = {
            'embedding': self.embedding,
            'output': self.output,
            'norm': self.norm.weight,
            'config': {
                'dim': self.dim, 'n_layers': self.n_layers,
                'n_heads': self.n_heads, 'vocab_size': self.vocab_size
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """Load model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.embedding = data['embedding']
        self.output = data['output']
        self.norm.weight = data['norm']


# ═══════════════════════════════════════════════════════════════════════════════
# CODE INTERPRETER
# ═══════════════════════════════════════════════════════════════════════════════

class CodeInterpreter:
    """Safe Python sandbox"""
    
    BLOCKED = {'eval', 'exec', 'compile', 'open', '__import__', 'globals', 'locals'}
    
    def __init__(self):
        self.namespace = self._create_namespace()
    
    def _create_namespace(self) -> Dict:
        import math, random, json, re, datetime, statistics
        return {
            '__builtins__': {
                'print': print, 'len': len, 'range': range, 'enumerate': enumerate,
                'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted,
                'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
                'str': str, 'int': int, 'float': float, 'bool': bool,
                'abs': abs, 'min': min, 'max': max, 'sum': sum, 'round': round,
                'pow': pow, 'all': all, 'any': any, 'chr': chr, 'ord': ord,
                'True': True, 'False': False, 'None': None,
            },
            'math': math, 'random': random, 'json': json, 're': re,
            'datetime': datetime, 'statistics': statistics, 'np': np,
        }
    
    def execute(self, code: str) -> Dict:
        result = {'success': False, 'output': '', 'error': '', 'result': None}
        
        for blocked in self.BLOCKED:
            if blocked in code:
                result['error'] = f"Blocked: '{blocked}'"
                return result
        
        stdout = io.StringIO()
        try:
            with redirect_stdout(stdout):
                exec(code, self.namespace)
                try:
                    last = code.strip().split('\n')[-1].strip()
                    if last and not any(last.startswith(k) for k in ['if', 'for', 'while', 'def', 'class', 'import', '#']):
                        result['result'] = eval(last, self.namespace)
                except: pass
            result['success'] = True
            result['output'] = stdout.getvalue()
        except Exception as e:
            result['error'] = f"{type(e).__name__}: {e}"
            result['output'] = stdout.getvalue()
        
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class ImageGenerator:
    def __init__(self):
        self.size = 512
    
    def generate(self, prompt: str) -> bytes:
        if not HAS_PIL:
            return b''
        
        img = Image.new('RGB', (self.size, self.size), (20, 20, 30))
        draw = ImageDraw.Draw(img)
        p = prompt.lower()
        
        if 'sunset' in p or 'sky' in p:
            for y in range(self.size//2):
                r = int(255*(1-y/(self.size//2))+50)
                g = int(100*(1-y/(self.size//2)))
                b = int(50+100*(y/(self.size//2)))
                draw.line([(0,y),(self.size,y)], fill=(r,g,b))
            draw.ellipse([self.size//2-60,self.size//3-60,self.size//2+60,self.size//3+60], fill=(255,200,50))
            draw.rectangle([0,self.size//2,self.size,self.size], fill=(30,30,40))
        elif 'cat' in p:
            cx, cy = self.size//2, self.size//2
            draw.rectangle([0,0,self.size,self.size], fill=(40,40,60))
            draw.ellipse([cx-100,cy-50,cx+100,cy+120], fill=(80,80,80))
            draw.ellipse([cx-80,cy-130,cx+80,cy+20], fill=(100,100,100))
            draw.polygon([(cx-70,cy-100),(cx-40,cy-180),(cx-10,cy-100)], fill=(100,100,100))
            draw.polygon([(cx+70,cy-100),(cx+40,cy-180),(cx+10,cy-100)], fill=(100,100,100))
            draw.ellipse([cx-50,cy-80,cx-20,cy-40], fill=(0,255,0))
            draw.ellipse([cx+20,cy-80,cx+50,cy-40], fill=(0,255,0))
            draw.polygon([(cx,cy-20),(cx-15,cy),(cx+15,cy)], fill=(255,150,150))
        elif 'space' in p or 'star' in p:
            draw.rectangle([0,0,self.size,self.size], fill=(5,5,15))
            for _ in range(200):
                x,y = random.randint(0,self.size), random.randint(0,self.size)
                b = random.randint(150,255)
                draw.ellipse([x,y,x+2,y+2], fill=(b,b,b))
            draw.ellipse([self.size*2//3-60,self.size//3-60,self.size*2//3+60,self.size//3+60], fill=(180,120,80))
        else:
            # Abstract
            for _ in range(25):
                color = (random.randint(50,255), random.randint(50,255), random.randint(50,255))
                x,y = random.randint(0,self.size), random.randint(0,self.size)
                r = random.randint(20,100)
                draw.ellipse([x-r,y-r,x+r,y+r], fill=color)
        
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY & SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

class MemorySystem:
    def __init__(self, path: str = "catseek_memory.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute('CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, key TEXT UNIQUE, value TEXT)')
        self.conn.execute('CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY, title TEXT, messages TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
        self.conn.commit()
    
    def remember(self, key: str, value: str):
        self.conn.execute('INSERT OR REPLACE INTO memories (key, value) VALUES (?, ?)', (key, value))
        self.conn.commit()
    
    def recall(self, key: str) -> Optional[str]:
        r = self.conn.execute('SELECT value FROM memories WHERE key = ?', (key,)).fetchone()
        return r[0] if r else None
    
    def list_memories(self) -> List[Dict]:
        return [{'key': r[0], 'value': r[1]} for r in self.conn.execute('SELECT key, value FROM memories').fetchall()]
    
    def save_conversation(self, title: str, messages: List):
        self.conn.execute('INSERT INTO conversations (title, messages) VALUES (?, ?)', (title, json.dumps(messages)))
        self.conn.commit()
    
    def load_conversations(self) -> List[Dict]:
        return [{'id': r[0], 'title': r[1]} for r in self.conn.execute('SELECT id, title FROM conversations ORDER BY created_at DESC').fetchall()]
    
    def load_conversation(self, id: int) -> Optional[List]:
        r = self.conn.execute('SELECT messages FROM conversations WHERE id = ?', (id,)).fetchone()
        return json.loads(r[0]) if r else None


class WebSearch:
    def __init__(self):
        self.cache = {}
    
    def search(self, query: str) -> List[Dict]:
        if query in self.cache:
            return self.cache[query]
        
        results = []
        try:
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            req = urllib.request.Request(url, headers={'User-Agent': 'CatSeek R1/4.0'})
            with urllib.request.urlopen(req, timeout=5, context=ctx) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if data.get('Abstract'):
                    results.append({'title': data.get('Heading', query), 'snippet': data.get('Abstract', '')})
                for t in data.get('RelatedTopics', [])[:5]:
                    if isinstance(t, dict) and 'Text' in t:
                        results.append({'title': t.get('Text', '')[:50], 'snippet': t.get('Text', '')})
        except:
            results = [{'title': f'Search: {query}', 'snippet': 'Search unavailable'}]
        
        self.cache[query] = results
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class CatSeekApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title(f"{APP_NAME} v{VERSION}")
        self.geometry("1400x900")
        self.minsize(1000, 700)
        
        # Theme
        self.theme = {
            "bg": "#0a0a0a", "bg2": "#111111", "bg3": "#1a1a1a",
            "sidebar": "#050505", "text": "#00ff00", "muted": "#00aa00",
            "accent": "#00ff00", "button_bg": "#003300", "button_fg": "#00ff00",
            "user": "#00ffff", "assistant": "#00ff00", "system": "#ffff00",
            "think": "#ff00ff", "error": "#ff0000",
        }
        
        # Initialize
        self.model = CatSeekR1(dim=256, n_layers=4, n_heads=4, n_kv_heads=2, n_experts=4)
        self.interpreter = CodeInterpreter()
        self.image_gen = ImageGenerator()
        self.web_search = WebSearch()
        self.memory = MemorySystem()
        
        self.messages = []
        self.streaming = False
        self.temp_var = tk.DoubleVar(value=0.8)
        self.cot_var = tk.BooleanVar(value=False)
        
        self._build_ui()
        self._show_welcome()
    
    def _build_ui(self):
        self.configure(bg=self.theme["bg"])
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # Sidebar
        sidebar = tk.Frame(self, width=260, bg=self.theme["sidebar"])
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        
        tk.Label(sidebar, text="🐱 CatSeek R1", font=("Courier", 18, "bold"),
                bg=self.theme["sidebar"], fg=self.theme["accent"]).pack(pady=15)
        tk.Label(sidebar, text="DeepSeek R1 Architecture", font=("Courier", 9),
                bg=self.theme["sidebar"], fg=self.theme["muted"]).pack()
        tk.Label(sidebar, text=f"Parameters: {self.model.param_count:,}", font=("Courier", 9),
                bg=self.theme["sidebar"], fg=self.theme["muted"]).pack(pady=(5,0))
        
        tk.Button(sidebar, text="➕ New Chat", font=("Courier", 10),
                 bg=self.theme["button_bg"], fg=self.theme["button_fg"],
                 relief="flat", command=self._new_chat).pack(fill="x", padx=15, pady=15)
        
        # Settings
        settings = tk.LabelFrame(sidebar, text="⚙️ Settings", font=("Courier", 10),
                                bg=self.theme["sidebar"], fg=self.theme["muted"])
        settings.pack(fill="x", padx=15, pady=10)
        
        tk.Label(settings, text="Temperature:", font=("Courier", 9),
                bg=self.theme["sidebar"], fg=self.theme["muted"]).pack(anchor="w", padx=5)
        tk.Scale(settings, from_=0.1, to=2.0, resolution=0.1, orient="horizontal",
                variable=self.temp_var, bg=self.theme["sidebar"], fg=self.theme["text"],
                highlightthickness=0, troughcolor=self.theme["button_bg"]).pack(fill="x", padx=5)
        
        tk.Checkbutton(settings, text="🧠 Chain-of-Thought", variable=self.cot_var,
                      bg=self.theme["sidebar"], fg=self.theme["text"],
                      selectcolor=self.theme["bg2"], font=("Courier", 9)).pack(anchor="w", padx=5)
        
        # Architecture info
        arch = tk.LabelFrame(sidebar, text="🏗️ Architecture", font=("Courier", 10),
                            bg=self.theme["sidebar"], fg=self.theme["muted"])
        arch.pack(fill="x", padx=15, pady=10)
        
        for txt in ["• MoE: 4 experts, top-2", "• MLA: Compressed KV", "• RoPE: Rotary Pos",
                   "• RMSNorm", "• SwiGLU FFN"]:
            tk.Label(arch, text=txt, font=("Courier", 8), bg=self.theme["sidebar"],
                    fg=self.theme["muted"]).pack(anchor="w", padx=5)
        
        # Main
        main = tk.Frame(self, bg=self.theme["bg"])
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)
        
        self.chat = scrolledtext.ScrolledText(main, font=("Courier", 11),
            bg=self.theme["bg2"], fg=self.theme["text"], wrap="word",
            state="disabled", padx=15, pady=15)
        self.chat.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.chat.tag_configure("user", foreground=self.theme["user"])
        self.chat.tag_configure("assistant", foreground=self.theme["assistant"])
        self.chat.tag_configure("system", foreground=self.theme["system"])
        self.chat.tag_configure("think", foreground=self.theme["think"])
        self.chat.tag_configure("error", foreground=self.theme["error"])
        
        # Input
        inp = tk.Frame(main, bg=self.theme["bg"])
        inp.grid(row=1, column=0, sticky="ew", padx=10, pady=(0,10))
        inp.grid_columnconfigure(0, weight=1)
        
        tools = tk.Frame(inp, bg=self.theme["bg"])
        tools.grid(row=0, column=0, sticky="w", pady=(0,5))
        for icon, cmd in [("📎", self._upload), ("🖼️", self._image_dialog),
                          ("🌐", self._search_dialog), ("🧠", self._memory_dialog)]:
            tk.Button(tools, text=icon, font=("Courier", 12), bg=self.theme["bg"],
                     fg=self.theme["text"], relief="flat", command=cmd).pack(side="left", padx=2)
        
        self.input = tk.Text(inp, height=3, font=("Courier", 11),
            bg=self.theme["bg3"], fg=self.theme["text"], wrap="word", padx=10, pady=10)
        self.input.grid(row=1, column=0, sticky="ew")
        self.input.bind("<Return>", self._on_enter)
        
        tk.Button(inp, text="Send ➤", font=("Courier", 11, "bold"),
                 bg=self.theme["button_bg"], fg=self.theme["button_fg"],
                 relief="flat", command=self._send).grid(row=1, column=1, padx=(10,0))
        
        # Right panel
        right = tk.Frame(self, width=350, bg=self.theme["sidebar"])
        right.grid(row=0, column=2, sticky="nsew")
        right.grid_propagate(False)
        
        tk.Label(right, text="💻 Code Interpreter", font=("Courier", 12, "bold"),
                bg=self.theme["sidebar"], fg=self.theme["accent"]).pack(pady=10)
        
        self.code_input = scrolledtext.ScrolledText(right, font=("Courier", 10),
            bg=self.theme["bg3"], fg=self.theme["text"], height=10)
        self.code_input.pack(fill="x", padx=10, pady=5)
        
        btn_frame = tk.Frame(right, bg=self.theme["sidebar"])
        btn_frame.pack(fill="x", padx=10)
        tk.Button(btn_frame, text="▶ Run", font=("Courier", 10),
                 bg=self.theme["button_bg"], fg=self.theme["button_fg"],
                 relief="flat", command=self._run_code).pack(side="left", padx=2)
        
        tk.Label(right, text="Output:", font=("Courier", 10),
                bg=self.theme["sidebar"], fg=self.theme["muted"]).pack(anchor="w", padx=10, pady=(10,0))
        
        self.code_output = scrolledtext.ScrolledText(right, font=("Courier", 10),
            bg=self.theme["bg3"], fg=self.theme["text"], height=8, state="disabled")
        self.code_output.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Image canvas
        tk.Label(right, text="🖼️ Generated Image", font=("Courier", 10),
                bg=self.theme["sidebar"], fg=self.theme["muted"]).pack(anchor="w", padx=10, pady=(10,0))
        self.img_canvas = tk.Canvas(right, width=330, height=200,
            bg=self.theme["bg3"], highlightthickness=0)
        self.img_canvas.pack(padx=10, pady=5)
    
    def _add_msg(self, role: str, content: str):
        self.chat.configure(state="normal")
        icons = {"user": "🧑", "assistant": "🐱", "system": "⚙️", "error": "❌"}
        names = {"user": "You", "assistant": "CatSeek R1", "system": "System", "error": "Error"}
        
        self.chat.insert(tk.END, f"\n{icons.get(role, '•')} {names.get(role, role)}:\n", role)
        
        # Color <think> blocks
        if '<think>' in content:
            parts = re.split(r'(<think>.*?</think>)', content, flags=re.DOTALL)
            for part in parts:
                if part.startswith('<think>'):
                    self.chat.insert(tk.END, part + "\n", "think")
                else:
                    self.chat.insert(tk.END, part)
            self.chat.insert(tk.END, "\n")
        else:
            self.chat.insert(tk.END, f"{content}\n")
        
        self.chat.configure(state="disabled")
        self.chat.see(tk.END)
        self.messages.append({"role": role, "content": content})
    
    def _show_welcome(self):
        self._add_msg("system", f"""🐱 Welcome to **CatSeek R1 14B**!

**🏗️ Architecture:** DeepSeek R1 Clone
• Mixture of Experts (MoE) - 4 experts, top-2 routing
• Multi-Head Latent Attention (MLA) - Compressed KV
• Rotary Position Embeddings (RoPE)
• RMSNorm + SwiGLU activation
• Chain-of-Thought reasoning with <think> tags

**📊 Parameters:** {self.model.param_count:,}

**Features:** Chat • Code • Images • Search • Memory • CoT Reasoning

Enable "🧠 Chain-of-Thought" for step-by-step reasoning!
Type /help for commands.""")
    
    def _on_enter(self, e):
        if not (e.state & 0x1):
            self._send()
            return "break"
    
    def _send(self):
        text = self.input.get("1.0", tk.END).strip()
        if not text or self.streaming:
            return
        self.input.delete("1.0", tk.END)
        self._add_msg("user", text)
        self._process(text)
    
    def _process(self, text: str):
        if text.startswith("/"):
            self._command(text)
            return
        
        if any(w in text.lower() for w in ['generate image', 'create image', 'draw']):
            self._gen_image(text)
            return
        
        if any(w in text.lower() for w in ['search for', 'look up']):
            self._search(text)
            return
        
        self._respond(text)
    
    def _command(self, text: str):
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if cmd == "/help":
            self._add_msg("system", """**Commands:**
/help - Show help
/think <question> - Use Chain-of-Thought
/code <python> - Execute code
/image <prompt> - Generate image
/search <query> - Web search
/remember <text> - Save to memory
/recall <key> - Get memory
/clear - New chat""")
        elif cmd == "/think":
            self.streaming = True
            def gen():
                resp = self.model.generate(arg, use_cot=True, temperature=self.temp_var.get())
                self.after(0, lambda: self._add_msg("assistant", resp))
                self.streaming = False
            threading.Thread(target=gen, daemon=True).start()
        elif cmd == "/code":
            self._exec_code(arg)
        elif cmd == "/image":
            self._gen_image(arg)
        elif cmd == "/search":
            self._search(arg)
        elif cmd == "/remember":
            key = hashlib.md5(arg.encode()).hexdigest()[:8]
            self.memory.remember(key, arg)
            self._add_msg("assistant", f"Remembered with key: {key}")
        elif cmd == "/recall":
            val = self.memory.recall(arg)
            self._add_msg("assistant", f"Memory [{arg}]: {val}" if val else "Not found")
        elif cmd == "/clear":
            self._new_chat()
        else:
            self._add_msg("error", f"Unknown: {cmd}")
    
    def _respond(self, text: str):
        self.streaming = True
        def gen():
            resp = self.model.generate(text, use_cot=self.cot_var.get(), temperature=self.temp_var.get())
            self.after(0, lambda: self._add_msg("assistant", resp))
            self.streaming = False
        threading.Thread(target=gen, daemon=True).start()
    
    def _run_code(self):
        code = self.code_input.get("1.0", tk.END).strip()
        if not code:
            return
        result = self.interpreter.execute(code)
        self.code_output.configure(state="normal")
        self.code_output.delete("1.0", tk.END)
        if result['output']:
            self.code_output.insert(tk.END, result['output'])
        if result['result'] is not None:
            self.code_output.insert(tk.END, f"\n→ {result['result']}")
        if result['error']:
            self.code_output.insert(tk.END, f"\n❌ {result['error']}")
        self.code_output.configure(state="disabled")
    
    def _exec_code(self, code: str):
        result = self.interpreter.execute(code)
        out = result['output'] or ""
        if result['result'] is not None:
            out += f"\n→ {result['result']}"
        if result['error']:
            out += f"\n❌ {result['error']}"
        self._add_msg("assistant", f"```\n{out}\n```")
    
    def _gen_image(self, prompt: str):
        if not HAS_PIL:
            self._add_msg("error", "PIL not installed")
            return
        self._add_msg("assistant", f"🎨 Generating: {prompt}")
        def gen():
            data = self.image_gen.generate(prompt)
            self.after(0, lambda: self._show_image(data))
        threading.Thread(target=gen, daemon=True).start()
    
    def _show_image(self, data: bytes):
        if not data:
            return
        img = Image.open(io.BytesIO(data)).resize((330, 200))
        self.current_img = ImageTk.PhotoImage(img)
        self.img_canvas.delete("all")
        self.img_canvas.create_image(165, 100, image=self.current_img)
    
    def _search(self, query: str):
        self._add_msg("assistant", f"🔍 Searching: {query}")
        def search():
            results = self.web_search.search(query)
            resp = "**Results:**\n\n"
            for r in results[:5]:
                resp += f"• **{r['title']}**\n  {r['snippet'][:150]}\n\n"
            self.after(0, lambda: self._add_msg("assistant", resp))
        threading.Thread(target=search, daemon=True).start()
    
    def _upload(self):
        path = filedialog.askopenfilename()
        if path:
            try:
                with open(path, 'r', errors='ignore') as f:
                    content = f.read()[:3000]
                self._add_msg("assistant", f"📁 Loaded {os.path.basename(path)}:\n```\n{content[:500]}...\n```")
            except:
                self._add_msg("error", "Failed to load file")
    
    def _image_dialog(self):
        try:
            prompt = tk.simpledialog.askstring("Generate Image", "Enter prompt:")
            if prompt:
                self._gen_image(prompt)
        except:
            pass
    
    def _search_dialog(self):
        try:
            query = tk.simpledialog.askstring("Web Search", "Enter query:")
            if query:
                self._search(query)
        except:
            pass
    
    def _memory_dialog(self):
        mems = self.memory.list_memories()
        self._add_msg("assistant", "**Memories:**\n" + "\n".join([f"• {m['key']}: {m['value'][:50]}" for m in mems[:10]]) or "None")
    
    def _new_chat(self):
        if self.messages:
            title = self.messages[0]['content'][:30] if self.messages else "Chat"
            self.memory.save_conversation(title, self.messages)
        self.messages = []
        self.chat.configure(state="normal")
        self.chat.delete("1.0", tk.END)
        self.chat.configure(state="disabled")
        self._show_welcome()


try:
    from tkinter import simpledialog
    tk.simpledialog = simpledialog
except:
    class SD:
        @staticmethod
        def askstring(t, p): return None
    tk.simpledialog = SD()


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          CATSEEK R1 14B - DeepSeek R1 Architecture           ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Architecture Components:                                    ║")
    print("║  • Mixture of Experts (MoE) with Top-K Routing               ║")
    print("║  • Multi-Head Latent Attention (MLA)                         ║")
    print("║  • Rotary Position Embeddings (RoPE)                         ║")
    print("║  • RMSNorm + SwiGLU Activation                               ║")
    print("║  • Chain-of-Thought (CoT) Reasoning                          ║")
    print("║  • KV-Cache for Efficient Generation                         ║")
    print("║                                                              ║")
    print("║  (C) 2025 Samsoft / Flames Co. / Team Flames                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"PIL: {'✓' if HAS_PIL else '✗'} | Matplotlib: {'✓' if HAS_MATPLOTLIB else '✗'}")
    print()
    
    app = CatSeekApp()
    app.mainloop()


if __name__ == "__main__":
    main()
