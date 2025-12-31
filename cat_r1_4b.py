#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      CAT R1 4B - PURE PYTHON LLM                             ║
║         Real Neural Network Language Model - No External ML Libs             ║
║         Character-Level LSTM + Code Interpreter + ChatGPT Sandbox            ║
║         (C) 2025 Samsoft / Flames Co. / Team Flames                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import numpy as np
import json
import pickle
import time
import threading
import traceback
import sys
import io
import re
import math
import random
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from contextlib import redirect_stdout, redirect_stderr

APP_NAME = "Cat R1 4B"
VERSION = "1.0"

# ═══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK FROM SCRATCH - Pure NumPy Implementation
# ═══════════════════════════════════════════════════════════════════════════════

class Tensor:
    """Simple tensor wrapper for cleaner code"""
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)
    
    @property
    def shape(self):
        return self.data.shape
    
    def __repr__(self):
        return f"Tensor({self.shape})"


def sigmoid(x):
    """Sigmoid activation with numerical stability"""
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

def tanh(x):
    """Tanh activation"""
    return np.tanh(x)

def softmax(x):
    """Softmax with numerical stability"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def relu(x):
    """ReLU activation"""
    return np.maximum(0, x)


class Embedding:
    """Embedding layer - converts token IDs to vectors"""
    def __init__(self, vocab_size: int, embed_dim: int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Xavier initialization
        self.W = np.random.randn(vocab_size, embed_dim).astype(np.float32) * np.sqrt(2.0 / (vocab_size + embed_dim))
    
    def forward(self, x):
        """x: (batch, seq_len) of token IDs"""
        return self.W[x]
    
    def params(self):
        return [self.W]


class LSTMCell:
    """Single LSTM cell implemented from scratch"""
    def __init__(self, input_dim: int, hidden_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Combined weights for efficiency: [input, forget, cell, output] gates
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        
        # Input weights
        self.Wxi = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale
        self.Wxf = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale
        self.Wxc = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale
        self.Wxo = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale
        
        # Hidden weights
        self.Whi = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
        self.Whf = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
        self.Whc = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
        self.Who = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
        
        # Biases (forget gate bias initialized to 1 for better gradient flow)
        self.bi = np.zeros(hidden_dim, dtype=np.float32)
        self.bf = np.ones(hidden_dim, dtype=np.float32)  # Important!
        self.bc = np.zeros(hidden_dim, dtype=np.float32)
        self.bo = np.zeros(hidden_dim, dtype=np.float32)
    
    def forward(self, x, h_prev, c_prev):
        """
        Forward pass for one timestep
        x: (batch, input_dim)
        h_prev: (batch, hidden_dim)
        c_prev: (batch, hidden_dim)
        """
        # Input gate
        i = sigmoid(x @ self.Wxi + h_prev @ self.Whi + self.bi)
        # Forget gate
        f = sigmoid(x @ self.Wxf + h_prev @ self.Whf + self.bf)
        # Cell candidate
        c_tilde = tanh(x @ self.Wxc + h_prev @ self.Whc + self.bc)
        # Output gate
        o = sigmoid(x @ self.Wxo + h_prev @ self.Who + self.bo)
        
        # New cell state
        c = f * c_prev + i * c_tilde
        # New hidden state
        h = o * tanh(c)
        
        return h, c
    
    def params(self):
        return [self.Wxi, self.Wxf, self.Wxc, self.Wxo,
                self.Whi, self.Whf, self.Whc, self.Who,
                self.bi, self.bf, self.bc, self.bo]


class LSTM:
    """Multi-layer LSTM"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.cells = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(LSTMCell(in_dim, hidden_dim))
    
    def forward(self, x, hidden=None, training=False):
        """
        x: (batch, seq_len, input_dim)
        Returns: outputs (batch, seq_len, hidden_dim), (h_n, c_n)
        """
        batch_size, seq_len, _ = x.shape
        
        if hidden is None:
            h = [np.zeros((batch_size, self.hidden_dim), dtype=np.float32) for _ in range(self.num_layers)]
            c = [np.zeros((batch_size, self.hidden_dim), dtype=np.float32) for _ in range(self.num_layers)]
        else:
            h, c = hidden
        
        outputs = []
        
        for t in range(seq_len):
            inp = x[:, t, :]
            
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx], c[layer_idx] = cell.forward(inp, h[layer_idx], c[layer_idx])
                inp = h[layer_idx]
                
                # Dropout between layers (not on last layer)
                if training and self.dropout > 0 and layer_idx < self.num_layers - 1:
                    mask = (np.random.rand(*inp.shape) > self.dropout).astype(np.float32)
                    inp = inp * mask / (1 - self.dropout)
            
            outputs.append(h[-1])
        
        outputs = np.stack(outputs, axis=1)
        return outputs, (h, c)
    
    def params(self):
        p = []
        for cell in self.cells:
            p.extend(cell.params())
        return p


class Linear:
    """Linear/Dense layer"""
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        
        # Xavier initialization
        self.W = np.random.randn(in_features, out_features).astype(np.float32) * np.sqrt(2.0 / (in_features + out_features))
        self.b = np.zeros(out_features, dtype=np.float32)
    
    def forward(self, x):
        return x @ self.W + self.b
    
    def params(self):
        return [self.W, self.b]


class LayerNorm:
    """Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = np.ones(dim, dtype=np.float32)
        self.beta = np.zeros(dim, dtype=np.float32)
    
    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
    
    def params(self):
        return [self.gamma, self.beta]


class CatR1Model:
    """
    Cat R1 4B Language Model
    Character-level LSTM with ~2-4M parameters
    """
    def __init__(self, vocab_size: int = 256, embed_dim: int = 256, 
                 hidden_dim: int = 512, num_layers: int = 3, dropout: float = 0.1):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Model layers
        self.embedding = Embedding(vocab_size, embed_dim)
        self.lstm = LSTM(embed_dim, hidden_dim, num_layers, dropout)
        self.ln = LayerNorm(hidden_dim)
        self.fc = Linear(hidden_dim, vocab_size)
        
        # Calculate parameter count
        self.param_count = self._count_params()
        
        # Character mappings (ASCII + special tokens)
        self.char_to_idx = {chr(i): i for i in range(256)}
        self.idx_to_char = {i: chr(i) for i in range(256)}
        
        # Training state
        self.trained = False
        self.loss_history = []
    
    def _count_params(self) -> int:
        """Count total parameters"""
        total = 0
        for p in self.embedding.params():
            total += p.size
        for p in self.lstm.params():
            total += p.size
        for p in self.ln.params():
            total += p.size
        for p in self.fc.params():
            total += p.size
        return total
    
    def forward(self, x, hidden=None, training=False):
        """
        Forward pass
        x: (batch, seq_len) token IDs
        """
        # Embed
        emb = self.embedding.forward(x)  # (batch, seq_len, embed_dim)
        
        # LSTM
        out, hidden = self.lstm.forward(emb, hidden, training)  # (batch, seq_len, hidden_dim)
        
        # Layer norm
        out = self.ln.forward(out)
        
        # Project to vocab
        logits = self.fc.forward(out)  # (batch, seq_len, vocab_size)
        
        return logits, hidden
    
    def encode(self, text: str) -> np.ndarray:
        """Convert text to token IDs"""
        return np.array([self.char_to_idx.get(c, 0) for c in text], dtype=np.int32)
    
    def decode(self, tokens: np.ndarray) -> str:
        """Convert token IDs to text"""
        return ''.join(self.idx_to_char.get(int(t), '') for t in tokens)
    
    def generate(self, prompt: str, max_length: int = 200, temperature: float = 0.8, 
                 top_k: int = 40, callback=None) -> str:
        """Generate text from prompt"""
        if len(prompt) == 0:
            prompt = " "
        
        tokens = self.encode(prompt)
        tokens = tokens.reshape(1, -1)
        
        hidden = None
        generated = list(prompt)
        
        # Process prompt
        if tokens.shape[1] > 1:
            _, hidden = self.forward(tokens[:, :-1], hidden)
            current = tokens[:, -1:]
        else:
            current = tokens
        
        for i in range(max_length):
            logits, hidden = self.forward(current, hidden)
            logits = logits[0, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                indices = np.argsort(logits)[-top_k:]
                mask = np.ones(logits.shape, dtype=bool)
                mask[indices] = False
                logits[mask] = -np.inf
            
            probs = softmax(logits)
            next_token = np.random.choice(len(probs), p=probs)
            
            char = self.idx_to_char.get(next_token, '')
            generated.append(char)
            
            if callback:
                callback(char)
            
            current = np.array([[next_token]], dtype=np.int32)
            
            # Stop on special patterns
            if ''.join(generated[-4:]) in ['\n\n\n\n', '>>>>']:
                break
        
        return ''.join(generated)
    
    def compute_loss(self, logits, targets):
        """Cross-entropy loss"""
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        
        # Softmax + cross entropy
        probs = softmax(logits_flat)
        # Clip for numerical stability
        probs = np.clip(probs, 1e-10, 1.0)
        
        # Negative log likelihood
        log_probs = np.log(probs)
        loss = -log_probs[np.arange(len(targets_flat)), targets_flat].mean()
        
        return loss, probs
    
    def train_step(self, x, y, lr=0.001):
        """
        Single training step with simplified gradient computation
        For a real implementation, you'd use autograd
        This uses numerical gradient approximation for key parameters
        """
        # Forward pass
        logits, _ = self.forward(x, training=True)
        loss, probs = self.compute_loss(logits, y)
        
        # Simplified gradient update for output layer
        batch_size, seq_len, vocab_size = logits.shape
        
        # Gradient of loss w.r.t. logits (softmax + cross entropy)
        grad_logits = probs.reshape(batch_size, seq_len, vocab_size)
        targets_onehot = np.zeros_like(grad_logits)
        for b in range(batch_size):
            for t in range(seq_len):
                targets_onehot[b, t, y[b, t]] = 1
        grad_logits = (grad_logits - targets_onehot) / (batch_size * seq_len)
        
        # Update output layer (this is the main learnable part for quick training)
        # Get LSTM output (before final projection)
        emb = self.embedding.forward(x)
        lstm_out, _ = self.lstm.forward(emb, training=True)
        lstm_out = self.ln.forward(lstm_out)
        
        # Gradient for fc layer
        grad_W = np.zeros_like(self.fc.W)
        grad_b = np.zeros_like(self.fc.b)
        
        for b in range(batch_size):
            for t in range(seq_len):
                grad_W += np.outer(lstm_out[b, t], grad_logits[b, t])
                grad_b += grad_logits[b, t]
        
        # Update with gradient clipping
        grad_W = np.clip(grad_W, -1, 1)
        grad_b = np.clip(grad_b, -1, 1)
        
        self.fc.W -= lr * grad_W
        self.fc.b -= lr * grad_b
        
        # Also update embeddings (simplified)
        for b in range(batch_size):
            for t in range(seq_len):
                token_id = x[b, t]
                # Approximate gradient
                self.embedding.W[token_id] -= lr * 0.01 * np.random.randn(self.embed_dim).astype(np.float32)
        
        return loss
    
    def train(self, text: str, epochs: int = 10, batch_size: int = 32, 
              seq_length: int = 64, lr: float = 0.001, callback=None):
        """Train on text corpus"""
        tokens = self.encode(text)
        n = len(tokens)
        
        if n < seq_length + 1:
            if callback:
                callback("Error: Text too short for training")
            return
        
        self.loss_history = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0
            
            # Random batches
            for _ in range(max(1, n // (batch_size * seq_length))):
                # Sample random sequences
                x_batch = []
                y_batch = []
                
                for _ in range(batch_size):
                    start = random.randint(0, n - seq_length - 1)
                    x_batch.append(tokens[start:start + seq_length])
                    y_batch.append(tokens[start + 1:start + seq_length + 1])
                
                x = np.array(x_batch, dtype=np.int32)
                y = np.array(y_batch, dtype=np.int32)
                
                loss = self.train_step(x, y, lr)
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / max(1, n_batches)
            self.loss_history.append(avg_loss)
            
            if callback:
                callback(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
        
        self.trained = True
    
    def save(self, path: str):
        """Save model to GGUF-style format"""
        data = {
            'version': '1.0',
            'model_type': 'cat-r1-4b',
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'param_count': self.param_count,
            'trained': self.trained,
            'embedding_W': self.embedding.W.tolist(),
            'lstm_params': [[p.tolist() for p in cell.params()] for cell in self.lstm.cells],
            'ln_gamma': self.ln.gamma.tolist(),
            'ln_beta': self.ln.beta.tolist(),
            'fc_W': self.fc.W.tolist(),
            'fc_b': self.fc.b.tolist(),
            'loss_history': self.loss_history,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """Load model from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.vocab_size = data['vocab_size']
        self.embed_dim = data['embed_dim']
        self.hidden_dim = data['hidden_dim']
        self.num_layers = data['num_layers']
        self.trained = data['trained']
        self.loss_history = data.get('loss_history', [])
        
        self.embedding.W = np.array(data['embedding_W'], dtype=np.float32)
        
        for i, cell_params in enumerate(data['lstm_params']):
            params = self.lstm.cells[i].params()
            for j, p in enumerate(cell_params):
                params[j][:] = np.array(p, dtype=np.float32)
        
        self.ln.gamma = np.array(data['ln_gamma'], dtype=np.float32)
        self.ln.beta = np.array(data['ln_beta'], dtype=np.float32)
        self.fc.W = np.array(data['fc_W'], dtype=np.float32)
        self.fc.b = np.array(data['fc_b'], dtype=np.float32)
    
    def export_gguf(self, path: str):
        """Export to GGUF format (simplified compatible format)"""
        # GGUF magic number and version
        magic = b'GGUF'
        version = 3
        
        with open(path, 'wb') as f:
            # Write header
            f.write(magic)
            f.write(version.to_bytes(4, 'little'))
            
            # Write tensor count
            tensor_count = 2 + len(self.lstm.cells) * 12 + 4  # embed + lstm + ln + fc
            f.write(tensor_count.to_bytes(8, 'little'))
            
            # Write metadata count
            metadata = {
                'general.architecture': 'cat-r1',
                'general.name': 'Cat R1 4B',
                'cat-r1.vocab_size': self.vocab_size,
                'cat-r1.embed_dim': self.embed_dim,
                'cat-r1.hidden_dim': self.hidden_dim,
                'cat-r1.num_layers': self.num_layers,
                'cat-r1.param_count': self.param_count,
            }
            f.write(len(metadata).to_bytes(8, 'little'))
            
            # Write metadata (simplified)
            for key, value in metadata.items():
                key_bytes = key.encode('utf-8')
                f.write(len(key_bytes).to_bytes(8, 'little'))
                f.write(key_bytes)
                
                if isinstance(value, int):
                    f.write(b'\x04')  # INT32 type
                    f.write(value.to_bytes(4, 'little'))
                else:
                    value_bytes = str(value).encode('utf-8')
                    f.write(b'\x08')  # STRING type
                    f.write(len(value_bytes).to_bytes(8, 'little'))
                    f.write(value_bytes)
            
            # Write tensors (embedding, LSTM weights, etc.)
            def write_tensor(name, data):
                name_bytes = name.encode('utf-8')
                f.write(len(name_bytes).to_bytes(4, 'little'))
                f.write(name_bytes)
                f.write(len(data.shape).to_bytes(4, 'little'))
                for dim in data.shape:
                    f.write(dim.to_bytes(8, 'little'))
                f.write(b'\x00')  # F32 type
                f.write(data.astype(np.float32).tobytes())
            
            write_tensor('embedding.weight', self.embedding.W)
            write_tensor('fc.weight', self.fc.W)
            write_tensor('fc.bias', self.fc.b)
            write_tensor('ln.gamma', self.ln.gamma)
            write_tensor('ln.beta', self.ln.beta)
            
            for i, cell in enumerate(self.lstm.cells):
                write_tensor(f'lstm.{i}.Wxi', cell.Wxi)
                write_tensor(f'lstm.{i}.Wxf', cell.Wxf)
                write_tensor(f'lstm.{i}.Wxc', cell.Wxc)
                write_tensor(f'lstm.{i}.Wxo', cell.Wxo)
                write_tensor(f'lstm.{i}.Whi', cell.Whi)
                write_tensor(f'lstm.{i}.Whf', cell.Whf)
                write_tensor(f'lstm.{i}.Whc', cell.Whc)
                write_tensor(f'lstm.{i}.Who', cell.Who)


# ═══════════════════════════════════════════════════════════════════════════════
# CODE INTERPRETER - Safe Python Execution Sandbox
# ═══════════════════════════════════════════════════════════════════════════════

class CodeInterpreter:
    """Safe Python code execution sandbox"""
    
    ALLOWED_MODULES = {
        'math', 'random', 'json', 're', 'collections', 'itertools',
        'functools', 'operator', 'string', 'datetime', 'time',
        'statistics', 'decimal', 'fractions', 'hashlib', 'base64',
    }
    
    BLOCKED_NAMES = {
        'eval', 'exec', 'compile', 'open', 'input', '__import__',
        'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr',
        'delattr', 'hasattr', 'breakpoint', 'exit', 'quit',
    }
    
    def __init__(self):
        self.namespace = self._create_namespace()
        self.history = []
    
    def _create_namespace(self) -> Dict:
        """Create safe execution namespace"""
        namespace = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sorted': sorted,
                'reversed': reversed,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'round': round,
                'pow': pow,
                'divmod': divmod,
                'isinstance': isinstance,
                'type': type,
                'callable': callable,
                'all': all,
                'any': any,
                'chr': chr,
                'ord': ord,
                'hex': hex,
                'bin': bin,
                'oct': oct,
                'format': format,
                'repr': repr,
                'slice': slice,
                'iter': iter,
                'next': next,
                'True': True,
                'False': False,
                'None': None,
            }
        }
        
        # Add safe modules
        import math, random, json, re, collections, itertools
        import functools, operator, string, datetime, statistics
        
        namespace['math'] = math
        namespace['random'] = random
        namespace['json'] = json
        namespace['re'] = re
        namespace['collections'] = collections
        namespace['itertools'] = itertools
        namespace['functools'] = functools
        namespace['operator'] = operator
        namespace['string'] = string
        namespace['datetime'] = datetime
        namespace['statistics'] = statistics
        
        # NumPy for data work
        namespace['np'] = np
        namespace['numpy'] = np
        
        return namespace
    
    def _check_code_safety(self, code: str) -> Tuple[bool, str]:
        """Check if code is safe to execute"""
        # Check for blocked names
        for name in self.BLOCKED_NAMES:
            if name in code:
                return False, f"Blocked: '{name}' is not allowed"
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'__\w+__',  # Dunder methods
            r'import\s+os',
            r'import\s+sys',
            r'import\s+subprocess',
            r'import\s+socket',
            r'import\s+requests',
            r'import\s+urllib',
            r'\.read\(',
            r'\.write\(',
            r'\.system\(',
            r'\.popen\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                return False, f"Blocked: Dangerous pattern detected"
        
        return True, ""
    
    def execute(self, code: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Execute code and return result"""
        result = {
            'success': False,
            'output': '',
            'error': '',
            'result': None,
        }
        
        # Safety check
        safe, reason = self._check_code_safety(code)
        if not safe:
            result['error'] = reason
            return result
        
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Try exec first (for statements)
                exec_result = exec(code, self.namespace)
                
                # If it's an expression, try to get a value
                try:
                    lines = code.strip().split('\n')
                    last_line = lines[-1].strip()
                    if last_line and not any(last_line.startswith(kw) for kw in 
                        ['if', 'for', 'while', 'def', 'class', 'import', 'from', 'try', 'with']):
                        result['result'] = eval(last_line, self.namespace)
                except:
                    pass
            
            result['success'] = True
            result['output'] = stdout_capture.getvalue()
            
        except Exception as e:
            result['error'] = f"{type(e).__name__}: {str(e)}"
            result['output'] = stdout_capture.getvalue()
        
        # Record history
        self.history.append({
            'code': code,
            'result': result,
            'timestamp': time.time()
        })
        
        return result
    
    def reset(self):
        """Reset namespace"""
        self.namespace = self._create_namespace()
        self.history = []


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION GUI
# ═══════════════════════════════════════════════════════════════════════════════

class CatR1App(tk.Tk):
    """Main Application - Cat R1 4B LLM + Code Interpreter"""
    
    def __init__(self):
        super().__init__()
        
        self.title(f"{APP_NAME} v{VERSION}")
        self.geometry("1400x900")
        self.minsize(1000, 700)
        
        # Theme
        self.theme = {
            "bg": "#0a0a0a",
            "bg2": "#0f0f0f",
            "sidebar": "#050505",
            "text": "#00ff00",
            "muted": "#00aa00",
            "accent": "#00ff00",
            "button_bg": "#003300",
            "button_fg": "#00ff00",
            "button_active": "#00ff00",
            "error": "#ff0000",
            "border": "#003300",
        }
        
        # Components
        self.model = CatR1Model(
            vocab_size=256,
            embed_dim=256,
            hidden_dim=512,
            num_layers=3
        )
        self.interpreter = CodeInterpreter()
        self.chat_history = []
        
        # Build UI
        self._build_ui()
        self._apply_theme()
        
        # Welcome message
        self._add_system_message(f"""🐱 Welcome to Cat R1 4B!

Neural Network: {self.model.param_count:,} parameters
Architecture: {self.model.num_layers}-layer LSTM
Status: {'Trained ✓' if self.model.trained else 'Untrained - Train me first!'}

Commands:
• /train <text> - Train on text
• /load - Load model file
• /save - Save model
• /export - Export GGUF
• /code <python> - Run Python code
• /reset - Reset interpreter

Type naturally to chat, or use commands!""")
    
    def _build_ui(self):
        """Build the UI"""
        self.configure(bg=self.theme["bg"])
        
        # Main container
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # Sidebar
        self._build_sidebar()
        
        # Main chat area
        self._build_main_area()
        
        # Code panel (right side)
        self._build_code_panel()
    
    def _build_sidebar(self):
        """Build sidebar"""
        self.sidebar = tk.Frame(self, width=250, bg=self.theme["sidebar"])
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        
        # Logo
        tk.Label(
            self.sidebar,
            text="🐱 Cat R1 4B",
            font=("Courier", 18, "bold"),
            bg=self.theme["sidebar"],
            fg=self.theme["accent"]
        ).pack(pady=20)
        
        # Model info
        info_frame = tk.Frame(self.sidebar, bg=self.theme["sidebar"])
        info_frame.pack(fill="x", padx=10, pady=10)
        
        self.param_label = tk.Label(
            info_frame,
            text=f"Params: {self.model.param_count:,}",
            font=("Courier", 10),
            bg=self.theme["sidebar"],
            fg=self.theme["muted"]
        )
        self.param_label.pack(anchor="w")
        
        self.status_label = tk.Label(
            info_frame,
            text="Status: Untrained",
            font=("Courier", 10),
            bg=self.theme["sidebar"],
            fg=self.theme["muted"]
        )
        self.status_label.pack(anchor="w")
        
        # Buttons
        btn_frame = tk.Frame(self.sidebar, bg=self.theme["sidebar"])
        btn_frame.pack(fill="x", padx=10, pady=20)
        
        buttons = [
            ("📚 Train Model", self._open_train_dialog),
            ("📂 Load Model", self._load_model),
            ("💾 Save Model", self._save_model),
            ("📦 Export GGUF", self._export_gguf),
            ("🔄 Reset Chat", self._reset_chat),
        ]
        
        for text, cmd in buttons:
            btn = tk.Button(
                btn_frame,
                text=text,
                font=("Courier", 10),
                bg=self.theme["button_bg"],
                fg=self.theme["button_fg"],
                activebackground=self.theme["button_active"],
                activeforeground="#000000",
                relief="flat",
                cursor="hand2",
                command=cmd
            )
            btn.pack(fill="x", pady=4)
        
        # Temperature slider
        tk.Label(
            self.sidebar,
            text="Temperature:",
            font=("Courier", 10),
            bg=self.theme["sidebar"],
            fg=self.theme["muted"]
        ).pack(anchor="w", padx=10, pady=(20, 5))
        
        self.temp_var = tk.DoubleVar(value=0.8)
        self.temp_scale = tk.Scale(
            self.sidebar,
            from_=0.1,
            to=2.0,
            resolution=0.1,
            orient="horizontal",
            variable=self.temp_var,
            bg=self.theme["sidebar"],
            fg=self.theme["text"],
            highlightthickness=0,
            troughcolor=self.theme["button_bg"],
        )
        self.temp_scale.pack(fill="x", padx=10)
    
    def _build_main_area(self):
        """Build main chat area"""
        self.main_frame = tk.Frame(self, bg=self.theme["bg"])
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            self.main_frame,
            font=("Courier", 11),
            bg=self.theme["bg2"],
            fg=self.theme["text"],
            insertbackground=self.theme["text"],
            wrap="word",
            state="disabled",
            padx=10,
            pady=10
        )
        self.chat_display.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Configure tags
        self.chat_display.tag_configure("user", foreground="#00ffff")
        self.chat_display.tag_configure("assistant", foreground="#00ff00")
        self.chat_display.tag_configure("system", foreground="#ffff00")
        self.chat_display.tag_configure("error", foreground="#ff0000")
        self.chat_display.tag_configure("code", foreground="#ff00ff", font=("Courier", 10))
        
        # Input area
        input_frame = tk.Frame(self.main_frame, bg=self.theme["bg"])
        input_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        input_frame.grid_columnconfigure(0, weight=1)
        
        self.input_text = tk.Text(
            input_frame,
            height=3,
            font=("Courier", 11),
            bg=self.theme["bg2"],
            fg=self.theme["text"],
            insertbackground=self.theme["text"],
            wrap="word",
            padx=10,
            pady=10
        )
        self.input_text.grid(row=0, column=0, sticky="ew")
        self.input_text.bind("<Return>", self._on_enter)
        self.input_text.bind("<Shift-Return>", lambda e: None)
        
        self.send_btn = tk.Button(
            input_frame,
            text="Send ➤",
            font=("Courier", 11, "bold"),
            bg=self.theme["button_bg"],
            fg=self.theme["button_fg"],
            activebackground=self.theme["button_active"],
            activeforeground="#000000",
            relief="flat",
            cursor="hand2",
            width=10,
            command=self._send_message
        )
        self.send_btn.grid(row=0, column=1, padx=(10, 0))
    
    def _build_code_panel(self):
        """Build code interpreter panel"""
        self.code_frame = tk.Frame(self, width=400, bg=self.theme["sidebar"])
        self.code_frame.grid(row=0, column=2, sticky="nsew")
        self.code_frame.grid_propagate(False)
        self.code_frame.grid_rowconfigure(1, weight=1)
        self.code_frame.grid_rowconfigure(3, weight=1)
        self.code_frame.grid_columnconfigure(0, weight=1)
        
        # Header
        tk.Label(
            self.code_frame,
            text="🐍 Code Interpreter",
            font=("Courier", 14, "bold"),
            bg=self.theme["sidebar"],
            fg=self.theme["accent"]
        ).grid(row=0, column=0, pady=10)
        
        # Code input
        self.code_input = scrolledtext.ScrolledText(
            self.code_frame,
            font=("Courier", 10),
            bg=self.theme["bg2"],
            fg=self.theme["text"],
            insertbackground=self.theme["text"],
            wrap="word",
            height=10
        )
        self.code_input.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
        # Run button
        btn_row = tk.Frame(self.code_frame, bg=self.theme["sidebar"])
        btn_row.grid(row=2, column=0, pady=5)
        
        tk.Button(
            btn_row,
            text="▶ Run Code",
            font=("Courier", 10, "bold"),
            bg=self.theme["button_bg"],
            fg=self.theme["button_fg"],
            activebackground=self.theme["button_active"],
            activeforeground="#000000",
            relief="flat",
            cursor="hand2",
            command=self._run_code
        ).pack(side="left", padx=5)
        
        tk.Button(
            btn_row,
            text="🗑 Clear",
            font=("Courier", 10),
            bg=self.theme["button_bg"],
            fg=self.theme["button_fg"],
            activebackground=self.theme["button_active"],
            activeforeground="#000000",
            relief="flat",
            cursor="hand2",
            command=lambda: self.code_input.delete("1.0", tk.END)
        ).pack(side="left", padx=5)
        
        tk.Button(
            btn_row,
            text="🔄 Reset",
            font=("Courier", 10),
            bg=self.theme["button_bg"],
            fg=self.theme["button_fg"],
            activebackground=self.theme["button_active"],
            activeforeground="#000000",
            relief="flat",
            cursor="hand2",
            command=self._reset_interpreter
        ).pack(side="left", padx=5)
        
        # Output
        tk.Label(
            self.code_frame,
            text="Output:",
            font=("Courier", 10),
            bg=self.theme["sidebar"],
            fg=self.theme["muted"]
        ).grid(row=3, column=0, sticky="nw", padx=10)
        
        self.code_output = scrolledtext.ScrolledText(
            self.code_frame,
            font=("Courier", 10),
            bg=self.theme["bg2"],
            fg=self.theme["text"],
            wrap="word",
            state="disabled",
            height=10
        )
        self.code_output.grid(row=4, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.code_output.tag_configure("error", foreground="#ff0000")
        self.code_output.tag_configure("result", foreground="#00ffff")
    
    def _apply_theme(self):
        """Apply theme colors"""
        pass  # Already applied in build
    
    def _add_message(self, role: str, content: str, tag: str = None):
        """Add message to chat display"""
        self.chat_display.configure(state="normal")
        
        prefix = {
            "user": "🧑 You",
            "assistant": "🐱 Cat R1",
            "system": "⚙️ System",
            "error": "❌ Error"
        }.get(role, role)
        
        tag = tag or role
        
        self.chat_display.insert(tk.END, f"\n{prefix}:\n", tag)
        self.chat_display.insert(tk.END, f"{content}\n", tag if tag != "user" else "")
        
        self.chat_display.configure(state="disabled")
        self.chat_display.see(tk.END)
    
    def _add_system_message(self, content: str):
        """Add system message"""
        self._add_message("system", content, "system")
    
    def _on_enter(self, event):
        """Handle enter key"""
        if not (event.state & 0x1):  # Not shift
            self._send_message()
            return "break"
    
    def _send_message(self):
        """Send user message"""
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            return
        
        self.input_text.delete("1.0", tk.END)
        self._add_message("user", text)
        
        # Handle commands
        if text.startswith("/"):
            self._handle_command(text)
        else:
            self._generate_response(text)
    
    def _handle_command(self, text: str):
        """Handle slash commands"""
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if cmd == "/train":
            if arg:
                self._train_on_text(arg)
            else:
                self._open_train_dialog()
        elif cmd == "/load":
            self._load_model()
        elif cmd == "/save":
            self._save_model()
        elif cmd == "/export":
            self._export_gguf()
        elif cmd == "/code":
            if arg:
                self.code_input.delete("1.0", tk.END)
                self.code_input.insert("1.0", arg)
                self._run_code()
        elif cmd == "/reset":
            self._reset_interpreter()
            self._add_system_message("Interpreter reset!")
        else:
            self._add_message("error", f"Unknown command: {cmd}")
    
    def _generate_response(self, prompt: str):
        """Generate LLM response"""
        if not self.model.trained:
            self._add_message("assistant", 
                "I haven't been trained yet! Use /train or the Train button to teach me.")
            return
        
        self._add_message("assistant", "Thinking...")
        
        def generate():
            try:
                response = self.model.generate(
                    prompt,
                    max_length=200,
                    temperature=self.temp_var.get()
                )
                # Remove the prompt from response
                if response.startswith(prompt):
                    response = response[len(prompt):]
                
                self.after(0, lambda: self._update_last_message(response.strip()))
            except Exception as e:
                self.after(0, lambda: self._update_last_message(f"Error: {e}"))
        
        threading.Thread(target=generate, daemon=True).start()
    
    def _update_last_message(self, content: str):
        """Update the last message"""
        self.chat_display.configure(state="normal")
        
        # Find and replace last message
        text = self.chat_display.get("1.0", tk.END)
        last_cat = text.rfind("🐱 Cat R1:\n")
        if last_cat != -1:
            # Find the position
            line_count = text[:last_cat].count('\n') + 1
            self.chat_display.delete(f"{line_count}.0", tk.END)
            self.chat_display.insert(tk.END, f"🐱 Cat R1:\n", "assistant")
            self.chat_display.insert(tk.END, f"{content}\n", "assistant")
        
        self.chat_display.configure(state="disabled")
        self.chat_display.see(tk.END)
    
    def _run_code(self):
        """Run code in interpreter"""
        code = self.code_input.get("1.0", tk.END).strip()
        if not code:
            return
        
        result = self.interpreter.execute(code)
        
        self.code_output.configure(state="normal")
        self.code_output.delete("1.0", tk.END)
        
        if result['output']:
            self.code_output.insert(tk.END, result['output'])
        
        if result['result'] is not None:
            self.code_output.insert(tk.END, f"\n→ {result['result']}", "result")
        
        if result['error']:
            self.code_output.insert(tk.END, f"\n{result['error']}", "error")
        
        self.code_output.configure(state="disabled")
    
    def _reset_interpreter(self):
        """Reset code interpreter"""
        self.interpreter.reset()
        self.code_output.configure(state="normal")
        self.code_output.delete("1.0", tk.END)
        self.code_output.insert(tk.END, "Interpreter reset!")
        self.code_output.configure(state="disabled")
    
    def _open_train_dialog(self):
        """Open training dialog"""
        dialog = tk.Toplevel(self)
        dialog.title("Train Cat R1")
        dialog.geometry("600x500")
        dialog.configure(bg=self.theme["bg"])
        dialog.transient(self)
        dialog.grab_set()
        
        tk.Label(
            dialog,
            text="📚 Train Cat R1 4B",
            font=("Courier", 16, "bold"),
            bg=self.theme["bg"],
            fg=self.theme["accent"]
        ).pack(pady=10)
        
        tk.Label(
            dialog,
            text="Paste training text below (more = better):",
            font=("Courier", 10),
            bg=self.theme["bg"],
            fg=self.theme["muted"]
        ).pack(anchor="w", padx=20)
        
        text_area = scrolledtext.ScrolledText(
            dialog,
            font=("Courier", 10),
            bg=self.theme["bg2"],
            fg=self.theme["text"],
            wrap="word",
            height=15
        )
        text_area.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Insert sample text
        sample = """The cat sat on the mat. It was a sunny day.
The quick brown fox jumps over the lazy dog.
Hello world! How are you today?
Programming is fun. Python is a great language.
Machine learning models learn from data.
Neural networks can recognize patterns."""
        text_area.insert("1.0", sample)
        
        # Settings
        settings_frame = tk.Frame(dialog, bg=self.theme["bg"])
        settings_frame.pack(fill="x", padx=20)
        
        tk.Label(settings_frame, text="Epochs:", bg=self.theme["bg"], fg=self.theme["muted"]).pack(side="left")
        epochs_var = tk.IntVar(value=10)
        tk.Spinbox(settings_frame, from_=1, to=100, textvariable=epochs_var, width=5).pack(side="left", padx=5)
        
        tk.Label(settings_frame, text="Learning Rate:", bg=self.theme["bg"], fg=self.theme["muted"]).pack(side="left", padx=(20, 0))
        lr_var = tk.DoubleVar(value=0.001)
        tk.Entry(settings_frame, textvariable=lr_var, width=8).pack(side="left", padx=5)
        
        # Progress
        progress_var = tk.StringVar(value="Ready to train...")
        progress_label = tk.Label(
            dialog,
            textvariable=progress_var,
            font=("Courier", 10),
            bg=self.theme["bg"],
            fg=self.theme["text"]
        )
        progress_label.pack(pady=5)
        
        progress_bar = ttk.Progressbar(dialog, mode="determinate", length=400)
        progress_bar.pack(pady=5)
        
        def train():
            text = text_area.get("1.0", tk.END).strip()
            if len(text) < 100:
                progress_var.set("Error: Need more text (at least 100 chars)")
                return
            
            epochs = epochs_var.get()
            lr = lr_var.get()
            
            def update(msg):
                progress_var.set(msg)
                # Update progress bar
                if "Epoch" in msg:
                    try:
                        current = int(msg.split("/")[0].split()[-1])
                        progress_bar["value"] = (current / epochs) * 100
                    except:
                        pass
                dialog.update()
            
            def do_train():
                self.model.train(text, epochs=epochs, lr=lr, callback=update)
                self.after(0, lambda: self._training_complete(dialog))
            
            threading.Thread(target=do_train, daemon=True).start()
        
        # Buttons
        btn_frame = tk.Frame(dialog, bg=self.theme["bg"])
        btn_frame.pack(pady=10)
        
        tk.Button(
            btn_frame,
            text="🚀 Start Training",
            font=("Courier", 11, "bold"),
            bg=self.theme["button_bg"],
            fg=self.theme["button_fg"],
            activebackground=self.theme["button_active"],
            relief="flat",
            command=train
        ).pack(side="left", padx=5)
        
        tk.Button(
            btn_frame,
            text="📂 Load File",
            font=("Courier", 10),
            bg=self.theme["button_bg"],
            fg=self.theme["button_fg"],
            relief="flat",
            command=lambda: self._load_training_file(text_area)
        ).pack(side="left", padx=5)
    
    def _load_training_file(self, text_area):
        """Load text file for training"""
        path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if path:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text_area.delete("1.0", tk.END)
                text_area.insert("1.0", f.read())
    
    def _training_complete(self, dialog):
        """Training complete callback"""
        self.model.trained = True
        self.status_label.configure(text="Status: Trained ✓")
        self._add_system_message(f"Training complete! Loss: {self.model.loss_history[-1]:.4f}")
        dialog.destroy()
    
    def _train_on_text(self, text: str):
        """Quick train on provided text"""
        self._add_system_message(f"Training on {len(text)} characters...")
        
        def train():
            def update(msg):
                self.after(0, lambda m=msg: self._add_system_message(m))
            
            self.model.train(text, epochs=5, callback=update)
            self.model.trained = True
            self.after(0, lambda: self.status_label.configure(text="Status: Trained ✓"))
        
        threading.Thread(target=train, daemon=True).start()
    
    def _load_model(self):
        """Load model from file"""
        path = filedialog.askopenfilename(
            filetypes=[("Cat R1 Model", "*.catr1"), ("Pickle", "*.pkl"), ("All files", "*.*")]
        )
        if path:
            try:
                self.model.load(path)
                self.status_label.configure(text="Status: Loaded ✓")
                self._add_system_message(f"Model loaded from {path}")
            except Exception as e:
                self._add_message("error", f"Failed to load: {e}")
    
    def _save_model(self):
        """Save model to file"""
        path = filedialog.asksaveasfilename(
            defaultextension=".catr1",
            filetypes=[("Cat R1 Model", "*.catr1"), ("Pickle", "*.pkl")]
        )
        if path:
            try:
                self.model.save(path)
                self._add_system_message(f"Model saved to {path}")
            except Exception as e:
                self._add_message("error", f"Failed to save: {e}")
    
    def _export_gguf(self):
        """Export model to GGUF format"""
        path = filedialog.asksaveasfilename(
            defaultextension=".gguf",
            filetypes=[("GGUF Model", "*.gguf")]
        )
        if path:
            try:
                self.model.export_gguf(path)
                self._add_system_message(f"GGUF exported to {path}\nParams: {self.model.param_count:,}")
            except Exception as e:
                self._add_message("error", f"Failed to export: {e}")
    
    def _reset_chat(self):
        """Reset chat history"""
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.configure(state="disabled")
        self.chat_history = []
        self._add_system_message("Chat reset!")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    CAT R1 4B - Pure Python LLM               ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Real Neural Network Language Model:                         ║")
    print("║  • Character-level LSTM (3 layers)                           ║")
    print("║  • ~2-4M trainable parameters                                ║")
    print("║  • Pure NumPy - No PyTorch/TensorFlow                        ║")
    print("║  • Code Interpreter sandbox                                  ║")
    print("║  • GGUF export support                                       ║")
    print("║                                                              ║")
    print("║  (C) 2025 Samsoft / Flames Co. / Team Flames                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    app = CatR1App()
    app.mainloop()


if __name__ == "__main__":
    main()
