#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     CAT R1 4B ULTIMATE - ChatGPT Clone                       ║
║         Full-Featured AI Assistant - Pure Python Implementation              ║
║                                                                              ║
║  Features:                                                                   ║
║  • Real Neural Network LLM (LSTM)      • Code Interpreter Sandbox            ║
║  • File Upload & Analysis              • Image Generation (Procedural)       ║
║  • Web Search                          • Data Analysis & Charts              ║
║  • Memory System                       • Canvas/Artifacts                    ║
║  • Voice I/O (TTS)                     • Plugin System                       ║
║  • Multi-Conversation                  • Export (PDF, MD, JSON)              ║
║  • DALL-E Style Prompts               • File Creation & Download            ║
║                                                                              ║
║  (C) 2025 Samsoft / Flames Co. / Team Flames                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext, colorchooser
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
import base64
import struct
import wave
import os
import tempfile
import webbrowser
import urllib.request
import urllib.parse
import http.client
import ssl
import html
import csv
import sqlite3
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Callable
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta
from collections import defaultdict
import textwrap

# Optional imports - graceful fallback
try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

APP_NAME = "Cat R1 4B Ultimate"
VERSION = "2.0"

# ═══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK LLM - Pure NumPy Implementation
# ═══════════════════════════════════════════════════════════════════════════════

def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def relu(x):
    return np.maximum(0, x)


class Embedding:
    def __init__(self, vocab_size: int, embed_dim: int):
        self.W = np.random.randn(vocab_size, embed_dim).astype(np.float32) * np.sqrt(2.0 / (vocab_size + embed_dim))
    
    def forward(self, x):
        return self.W[x]


class LSTMCell:
    def __init__(self, input_dim: int, hidden_dim: int):
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.Wxi = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale
        self.Wxf = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale
        self.Wxc = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale
        self.Wxo = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale
        self.Whi = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
        self.Whf = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
        self.Whc = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
        self.Who = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
        self.bi = np.zeros(hidden_dim, dtype=np.float32)
        self.bf = np.ones(hidden_dim, dtype=np.float32)
        self.bc = np.zeros(hidden_dim, dtype=np.float32)
        self.bo = np.zeros(hidden_dim, dtype=np.float32)
    
    def forward(self, x, h_prev, c_prev):
        i = sigmoid(x @ self.Wxi + h_prev @ self.Whi + self.bi)
        f = sigmoid(x @ self.Wxf + h_prev @ self.Whf + self.bf)
        c_tilde = tanh(x @ self.Wxc + h_prev @ self.Whc + self.bc)
        o = sigmoid(x @ self.Wxo + h_prev @ self.Who + self.bo)
        c = f * c_prev + i * c_tilde
        h = o * tanh(c)
        return h, c


class LSTM:
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cells = [LSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]
    
    def forward(self, x, hidden=None):
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
            outputs.append(h[-1])
        
        return np.stack(outputs, axis=1), (h, c)


class LayerNorm:
    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = np.ones(dim, dtype=np.float32)
        self.beta = np.zeros(dim, dtype=np.float32)
    
    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta


class Linear:
    def __init__(self, in_features: int, out_features: int):
        self.W = np.random.randn(in_features, out_features).astype(np.float32) * np.sqrt(2.0 / (in_features + out_features))
        self.b = np.zeros(out_features, dtype=np.float32)
    
    def forward(self, x):
        return x @ self.W + self.b


class CatR1Model:
    """Cat R1 4B Language Model - Character-level LSTM"""
    
    def __init__(self, vocab_size: int = 256, embed_dim: int = 256, hidden_dim: int = 512, num_layers: int = 3):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = Embedding(vocab_size, embed_dim)
        self.lstm = LSTM(embed_dim, hidden_dim, num_layers)
        self.ln = LayerNorm(hidden_dim)
        self.fc = Linear(hidden_dim, vocab_size)
        
        self.char_to_idx = {chr(i): i for i in range(256)}
        self.idx_to_char = {i: chr(i) for i in range(256)}
        self.trained = False
        self.param_count = self._count_params()
    
    def _count_params(self) -> int:
        total = self.embedding.W.size + self.fc.W.size + self.fc.b.size
        total += self.ln.gamma.size + self.ln.beta.size
        for cell in self.lstm.cells:
            for attr in ['Wxi', 'Wxf', 'Wxc', 'Wxo', 'Whi', 'Whf', 'Whc', 'Who', 'bi', 'bf', 'bc', 'bo']:
                total += getattr(cell, attr).size
        return total
    
    def forward(self, x, hidden=None):
        emb = self.embedding.forward(x)
        out, hidden = self.lstm.forward(emb, hidden)
        out = self.ln.forward(out)
        logits = self.fc.forward(out)
        return logits, hidden
    
    def encode(self, text: str) -> np.ndarray:
        return np.array([self.char_to_idx.get(c, 0) for c in text], dtype=np.int32)
    
    def decode(self, tokens: np.ndarray) -> str:
        return ''.join(self.idx_to_char.get(int(t), '') for t in tokens)
    
    def generate(self, prompt: str, max_length: int = 200, temperature: float = 0.8, top_k: int = 40) -> str:
        if not prompt:
            prompt = " "
        
        tokens = self.encode(prompt).reshape(1, -1)
        hidden = None
        generated = list(prompt)
        
        if tokens.shape[1] > 1:
            _, hidden = self.forward(tokens[:, :-1], hidden)
            current = tokens[:, -1:]
        else:
            current = tokens
        
        for _ in range(max_length):
            logits, hidden = self.forward(current, hidden)
            logits = logits[0, -1, :] / temperature
            
            if top_k > 0:
                indices = np.argsort(logits)[-top_k:]
                mask = np.ones(logits.shape, dtype=bool)
                mask[indices] = False
                logits[mask] = -np.inf
            
            probs = softmax(logits)
            next_token = np.random.choice(len(probs), p=probs)
            generated.append(self.idx_to_char.get(next_token, ''))
            current = np.array([[next_token]], dtype=np.int32)
            
            if ''.join(generated[-4:]) in ['\n\n\n\n', '>>>>']:
                break
        
        return ''.join(generated)
    
    def train(self, text: str, epochs: int = 10, batch_size: int = 32, seq_length: int = 64, lr: float = 0.001, callback=None):
        tokens = self.encode(text)
        n = len(tokens)
        if n < seq_length + 1:
            return
        
        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0
            
            for _ in range(max(1, n // (batch_size * seq_length))):
                x_batch, y_batch = [], []
                for _ in range(batch_size):
                    start = random.randint(0, n - seq_length - 1)
                    x_batch.append(tokens[start:start + seq_length])
                    y_batch.append(tokens[start + 1:start + seq_length + 1])
                
                x = np.array(x_batch, dtype=np.int32)
                y = np.array(y_batch, dtype=np.int32)
                
                # Forward
                logits, _ = self.forward(x)
                probs = softmax(logits.reshape(-1, self.vocab_size))
                targets = y.reshape(-1)
                loss = -np.log(np.clip(probs[np.arange(len(targets)), targets], 1e-10, 1.0)).mean()
                
                # Simplified gradient update
                grad_logits = probs.reshape(batch_size, seq_length, self.vocab_size)
                for b in range(batch_size):
                    for t in range(seq_length):
                        grad_logits[b, t, y[b, t]] -= 1
                grad_logits /= (batch_size * seq_length)
                
                emb = self.embedding.forward(x)
                lstm_out, _ = self.lstm.forward(emb)
                lstm_out = self.ln.forward(lstm_out)
                
                grad_W = np.zeros_like(self.fc.W)
                grad_b = np.zeros_like(self.fc.b)
                for b in range(batch_size):
                    for t in range(seq_length):
                        grad_W += np.outer(lstm_out[b, t], grad_logits[b, t])
                        grad_b += grad_logits[b, t]
                
                self.fc.W -= lr * np.clip(grad_W, -1, 1)
                self.fc.b -= lr * np.clip(grad_b, -1, 1)
                
                epoch_loss += loss
                n_batches += 1
            
            if callback:
                callback(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / max(1, n_batches):.4f}")
        
        self.trained = True
    
    def save(self, path: str):
        data = {
            'vocab_size': self.vocab_size, 'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim, 'num_layers': self.num_layers,
            'trained': self.trained, 'embedding_W': self.embedding.W,
            'fc_W': self.fc.W, 'fc_b': self.fc.b,
            'ln_gamma': self.ln.gamma, 'ln_beta': self.ln.beta,
            'lstm_cells': [(cell.Wxi, cell.Wxf, cell.Wxc, cell.Wxo,
                           cell.Whi, cell.Whf, cell.Whc, cell.Who,
                           cell.bi, cell.bf, cell.bc, cell.bo) for cell in self.lstm.cells]
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.embedding.W = data['embedding_W']
        self.fc.W = data['fc_W']
        self.fc.b = data['fc_b']
        self.ln.gamma = data['ln_gamma']
        self.ln.beta = data['ln_beta']
        for i, params in enumerate(data['lstm_cells']):
            cell = self.lstm.cells[i]
            cell.Wxi, cell.Wxf, cell.Wxc, cell.Wxo = params[:4]
            cell.Whi, cell.Whf, cell.Whc, cell.Who = params[4:8]
            cell.bi, cell.bf, cell.bc, cell.bo = params[8:12]
        self.trained = data['trained']


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE & SMART RESPONSES
# ═══════════════════════════════════════════════════════════════════════════════

class KnowledgeBase:
    """Built-in knowledge for intelligent responses"""
    
    def __init__(self):
        self.knowledge = self._build_knowledge()
        self.context_memory = []
        self.user_memory = {}
    
    def _build_knowledge(self) -> Dict:
        return {
            'greeting': {
                'triggers': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening', 'howdy'],
                'responses': [
                    "Hello! 🐱 I'm Cat R1, your AI assistant. How can I help you today?",
                    "Hi there! Ready to assist with anything you need.",
                    "Hey! What can I do for you?",
                ]
            },
            'capabilities': {
                'triggers': ['what can you do', 'your capabilities', 'features', 'help me', 'what are you capable of'],
                'responses': [
                    """🐱 **Cat R1 4B Ultimate** can help you with:

📝 **Chat & Conversation** - Natural language understanding
💻 **Code Interpreter** - Execute Python code safely
📊 **Data Analysis** - Analyze CSV, JSON, create charts
🖼️ **Image Generation** - Create images from descriptions
🌐 **Web Search** - Find information online
📁 **File Handling** - Upload, analyze, create files
🧠 **Memory** - Remember context across conversations
🎨 **Canvas** - Create documents and artifacts
🔊 **Voice** - Text-to-speech output
🔌 **Plugins** - Extensible functionality

Just ask naturally!""",
                ]
            },
            'coding': {
                'triggers': ['write code', 'python', 'javascript', 'programming', 'function', 'debug', 'code'],
                'responses': ["I can help with coding! Use the Code Interpreter panel on the right, or ask me to write code."]
            },
            'math': {
                'triggers': ['calculate', 'math', 'equation', 'solve', 'compute'],
                'responses': ["I can do math! Try the code interpreter for complex calculations, or ask me directly."]
            },
            'image': {
                'triggers': ['generate image', 'create image', 'draw', 'picture', 'illustration', 'dall-e', 'image of'],
                'responses': ["I'll generate an image for you! Let me create that..."]
            },
            'search': {
                'triggers': ['search for', 'look up', 'find information', 'google', 'what is', 'who is', 'when did'],
                'responses': ["I'll search for that information..."]
            },
            'thanks': {
                'triggers': ['thank', 'thanks', 'appreciate', 'helpful'],
                'responses': ["You're welcome! 😊", "Happy to help!", "Anytime!"]
            },
            'goodbye': {
                'triggers': ['bye', 'goodbye', 'see you', 'quit', 'exit'],
                'responses': ["Goodbye! Come back anytime! 🐱", "See you later!", "Take care!"]
            },
        }
    
    def find_response(self, query: str) -> Tuple[Optional[str], str]:
        """Find best matching response"""
        query_lower = query.lower()
        
        for category, data in self.knowledge.items():
            for trigger in data['triggers']:
                if trigger in query_lower:
                    return category, random.choice(data['responses'])
        
        return None, ""
    
    def add_memory(self, key: str, value: str):
        self.user_memory[key] = value
    
    def get_memory(self, key: str) -> Optional[str]:
        return self.user_memory.get(key)


# ═══════════════════════════════════════════════════════════════════════════════
# CODE INTERPRETER - Safe Python Sandbox
# ═══════════════════════════════════════════════════════════════════════════════

class CodeInterpreter:
    """Safe Python execution sandbox"""
    
    BLOCKED = {'eval', 'exec', 'compile', 'open', '__import__', 'globals', 'locals', 'exit', 'quit'}
    
    def __init__(self):
        self.namespace = self._create_namespace()
        self.history = []
        self.files = {}  # Virtual filesystem
    
    def _create_namespace(self) -> Dict:
        ns = {
            '__builtins__': {
                'print': print, 'len': len, 'range': range, 'enumerate': enumerate,
                'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted,
                'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
                'str': str, 'int': int, 'float': float, 'bool': bool,
                'abs': abs, 'min': min, 'max': max, 'sum': sum, 'round': round,
                'pow': pow, 'divmod': divmod, 'isinstance': isinstance, 'type': type,
                'all': all, 'any': any, 'chr': chr, 'ord': ord, 'hex': hex, 'bin': bin,
                'True': True, 'False': False, 'None': None,
            }
        }
        
        import math, random, json, re, datetime, statistics, itertools, functools
        ns.update({
            'math': math, 'random': random, 'json': json, 're': re,
            'datetime': datetime, 'statistics': statistics,
            'itertools': itertools, 'functools': functools,
            'np': np, 'numpy': np,
        })
        
        if HAS_MATPLOTLIB:
            ns['plt'] = plt
            ns['matplotlib'] = matplotlib
        
        return ns
    
    def execute(self, code: str) -> Dict[str, Any]:
        result = {'success': False, 'output': '', 'error': '', 'result': None, 'figures': []}
        
        for blocked in self.BLOCKED:
            if blocked in code:
                result['error'] = f"Blocked: '{blocked}' not allowed"
                return result
        
        stdout_capture = io.StringIO()
        
        try:
            with redirect_stdout(stdout_capture):
                exec(code, self.namespace)
                
                # Check for matplotlib figures
                if HAS_MATPLOTLIB:
                    figs = [plt.figure(i) for i in plt.get_fignums()]
                    for fig in figs:
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#0a0a0a')
                        buf.seek(0)
                        result['figures'].append(buf.getvalue())
                    plt.close('all')
                
                # Try to get last expression value
                try:
                    lines = code.strip().split('\n')
                    last = lines[-1].strip()
                    if last and not any(last.startswith(k) for k in ['if', 'for', 'while', 'def', 'class', 'import', 'from', 'try', 'with', '#']):
                        result['result'] = eval(last, self.namespace)
                except:
                    pass
            
            result['success'] = True
            result['output'] = stdout_capture.getvalue()
            
        except Exception as e:
            result['error'] = f"{type(e).__name__}: {str(e)}"
            result['output'] = stdout_capture.getvalue()
        
        self.history.append({'code': code, 'result': result})
        return result
    
    def reset(self):
        self.namespace = self._create_namespace()
        self.history = []


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE GENERATOR - Procedural Art
# ═══════════════════════════════════════════════════════════════════════════════

class ImageGenerator:
    """Generate images from text prompts"""
    
    def __init__(self):
        self.width = 512
        self.height = 512
    
    def generate(self, prompt: str) -> bytes:
        """Generate image from prompt"""
        if HAS_PIL:
            return self._generate_pil(prompt)
        else:
            return self._generate_ascii(prompt)
    
    def _generate_pil(self, prompt: str) -> bytes:
        """Generate with PIL"""
        # Parse prompt for colors and shapes
        colors = self._extract_colors(prompt)
        
        img = Image.new('RGB', (self.width, self.height), colors['background'])
        draw = ImageDraw.Draw(img)
        
        # Generate based on keywords
        prompt_lower = prompt.lower()
        
        if any(w in prompt_lower for w in ['sunset', 'sunrise', 'sky']):
            self._draw_sunset(draw, colors)
        elif any(w in prompt_lower for w in ['mountain', 'landscape']):
            self._draw_mountains(draw, colors)
        elif any(w in prompt_lower for w in ['ocean', 'sea', 'water', 'wave']):
            self._draw_ocean(draw, colors)
        elif any(w in prompt_lower for w in ['forest', 'tree', 'nature']):
            self._draw_forest(draw, colors)
        elif any(w in prompt_lower for w in ['city', 'building', 'urban']):
            self._draw_city(draw, colors)
        elif any(w in prompt_lower for w in ['space', 'star', 'galaxy', 'cosmic']):
            self._draw_space(draw, colors)
        elif any(w in prompt_lower for w in ['abstract', 'art', 'pattern']):
            self._draw_abstract(draw, colors)
        elif any(w in prompt_lower for w in ['cat', 'kitten', 'feline']):
            self._draw_cat(draw, colors)
        else:
            self._draw_abstract(draw, colors)
        
        # Add some noise/texture
        img = img.filter(ImageFilter.SMOOTH)
        
        # Convert to bytes
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    
    def _extract_colors(self, prompt: str) -> Dict:
        prompt_lower = prompt.lower()
        
        color_map = {
            'red': (220, 60, 60), 'blue': (60, 120, 220), 'green': (60, 180, 80),
            'yellow': (240, 220, 60), 'purple': (160, 60, 200), 'orange': (240, 140, 40),
            'pink': (255, 150, 180), 'black': (20, 20, 20), 'white': (240, 240, 240),
            'cyan': (60, 220, 220), 'gold': (255, 200, 50),
        }
        
        colors = {
            'primary': (100, 150, 200),
            'secondary': (200, 100, 150),
            'background': (30, 30, 50),
            'accent': (255, 200, 100),
        }
        
        for name, rgb in color_map.items():
            if name in prompt_lower:
                colors['primary'] = rgb
                break
        
        if 'dark' in prompt_lower:
            colors['background'] = (15, 15, 25)
        elif 'light' in prompt_lower or 'bright' in prompt_lower:
            colors['background'] = (200, 210, 220)
        
        return colors
    
    def _draw_sunset(self, draw, colors):
        h = self.height
        w = self.width
        
        # Gradient sky
        for y in range(h // 2):
            ratio = y / (h // 2)
            r = int(255 * (1 - ratio) + 100 * ratio)
            g = int(100 * (1 - ratio) + 50 * ratio)
            b = int(50 * (1 - ratio) + 100 * ratio)
            draw.line([(0, y), (w, y)], fill=(r, g, b))
        
        # Sun
        sun_y = h // 3
        draw.ellipse([w//2-60, sun_y-60, w//2+60, sun_y+60], fill=(255, 200, 50))
        
        # Ground
        draw.rectangle([0, h//2, w, h], fill=(30, 30, 40))
    
    def _draw_mountains(self, draw, colors):
        h, w = self.height, self.width
        
        # Sky gradient
        for y in range(h * 2 // 3):
            ratio = y / (h * 2 // 3)
            draw.line([(0, y), (w, y)], fill=(
                int(135 - 80 * ratio),
                int(180 - 100 * ratio),
                int(220 - 80 * ratio)
            ))
        
        # Mountains
        for layer in range(3):
            points = [(0, h)]
            peaks = random.randint(3, 6)
            for i in range(peaks + 1):
                x = i * w // peaks
                y = h * (0.4 + layer * 0.15) + random.randint(-50, 50)
                points.append((x, y))
            points.append((w, h))
            
            shade = 60 + layer * 40
            draw.polygon(points, fill=(shade, shade + 10, shade + 20))
    
    def _draw_ocean(self, draw, colors):
        h, w = self.height, self.width
        
        # Sky
        for y in range(h // 2):
            ratio = y / (h // 2)
            draw.line([(0, y), (w, y)], fill=(
                int(135 + 50 * ratio),
                int(180 + 20 * ratio),
                int(220)
            ))
        
        # Ocean waves
        for y in range(h // 2, h):
            wave = math.sin((y + random.random() * 10) * 0.1) * 20
            ratio = (y - h // 2) / (h // 2)
            draw.line([(0, y), (w, y)], fill=(
                int(20 + 40 * ratio),
                int(80 + 40 * ratio),
                int(150 - 30 * ratio)
            ))
    
    def _draw_forest(self, draw, colors):
        h, w = self.height, self.width
        
        # Sky
        draw.rectangle([0, 0, w, h], fill=(180, 200, 180))
        
        # Trees
        for _ in range(20):
            x = random.randint(0, w)
            tree_h = random.randint(100, 300)
            tree_w = tree_h // 3
            y = h - random.randint(0, 100)
            
            # Trunk
            draw.rectangle([x - 10, y - tree_h // 3, x + 10, y], fill=(80, 50, 30))
            
            # Foliage
            for i in range(3):
                offset = i * tree_h // 6
                draw.polygon([
                    (x, y - tree_h + offset),
                    (x - tree_w + i * 20, y - tree_h // 3 + offset),
                    (x + tree_w - i * 20, y - tree_h // 3 + offset),
                ], fill=(30 + random.randint(0, 40), 100 + random.randint(0, 50), 30))
    
    def _draw_city(self, draw, colors):
        h, w = self.height, self.width
        
        # Night sky
        draw.rectangle([0, 0, w, h], fill=(20, 25, 40))
        
        # Stars
        for _ in range(100):
            x, y = random.randint(0, w), random.randint(0, h // 2)
            draw.point((x, y), fill=(255, 255, 255))
        
        # Buildings
        for i in range(15):
            bw = random.randint(40, 100)
            bh = random.randint(100, 350)
            x = i * w // 15 + random.randint(-20, 20)
            
            shade = random.randint(40, 80)
            draw.rectangle([x, h - bh, x + bw, h], fill=(shade, shade, shade + 10))
            
            # Windows
            for wy in range(h - bh + 10, h - 10, 20):
                for wx in range(x + 5, x + bw - 5, 15):
                    if random.random() > 0.3:
                        draw.rectangle([wx, wy, wx + 8, wy + 12], fill=(255, 240, 150))
    
    def _draw_space(self, draw, colors):
        h, w = self.height, self.width
        
        # Deep space
        draw.rectangle([0, 0, w, h], fill=(5, 5, 15))
        
        # Stars
        for _ in range(300):
            x, y = random.randint(0, w), random.randint(0, h)
            brightness = random.randint(100, 255)
            size = random.randint(1, 3)
            draw.ellipse([x, y, x + size, y + size], fill=(brightness, brightness, brightness))
        
        # Nebula
        for _ in range(5):
            cx, cy = random.randint(0, w), random.randint(0, h)
            for r in range(100, 10, -10):
                alpha = (100 - r) // 10
                color = (
                    random.randint(100, 200),
                    random.randint(50, 150),
                    random.randint(150, 255)
                )
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=None)
        
        # Planet
        px, py = w // 3 * 2, h // 3
        draw.ellipse([px - 80, py - 80, px + 80, py + 80], fill=(180, 120, 80))
    
    def _draw_abstract(self, draw, colors):
        h, w = self.height, self.width
        
        draw.rectangle([0, 0, w, h], fill=colors['background'])
        
        # Shapes
        for _ in range(30):
            shape = random.choice(['circle', 'rect', 'line'])
            color = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )
            
            if shape == 'circle':
                x, y = random.randint(0, w), random.randint(0, h)
                r = random.randint(20, 100)
                draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
            elif shape == 'rect':
                x1, y1 = random.randint(0, w), random.randint(0, h)
                x2, y2 = x1 + random.randint(50, 200), y1 + random.randint(50, 200)
                draw.rectangle([x1, y1, x2, y2], fill=color)
            else:
                draw.line([
                    random.randint(0, w), random.randint(0, h),
                    random.randint(0, w), random.randint(0, h)
                ], fill=color, width=random.randint(2, 10))
    
    def _draw_cat(self, draw, colors):
        h, w = self.height, self.width
        cx, cy = w // 2, h // 2
        
        draw.rectangle([0, 0, w, h], fill=colors['background'])
        
        # Body
        draw.ellipse([cx - 100, cy - 50, cx + 100, cy + 120], fill=(80, 80, 80))
        
        # Head
        draw.ellipse([cx - 80, cy - 130, cx + 80, cy + 20], fill=(100, 100, 100))
        
        # Ears
        draw.polygon([(cx - 70, cy - 100), (cx - 40, cy - 180), (cx - 10, cy - 100)], fill=(100, 100, 100))
        draw.polygon([(cx + 70, cy - 100), (cx + 40, cy - 180), (cx + 10, cy - 100)], fill=(100, 100, 100))
        draw.polygon([(cx - 60, cy - 100), (cx - 40, cy - 160), (cx - 20, cy - 100)], fill=(255, 150, 150))
        draw.polygon([(cx + 60, cy - 100), (cx + 40, cy - 160), (cx + 20, cy - 100)], fill=(255, 150, 150))
        
        # Eyes
        draw.ellipse([cx - 50, cy - 80, cx - 20, cy - 40], fill=(0, 255, 0))
        draw.ellipse([cx + 20, cy - 80, cx + 50, cy - 40], fill=(0, 255, 0))
        draw.ellipse([cx - 40, cy - 70, cx - 30, cy - 50], fill=(0, 0, 0))
        draw.ellipse([cx + 30, cy - 70, cx + 40, cy - 50], fill=(0, 0, 0))
        
        # Nose
        draw.polygon([(cx, cy - 20), (cx - 15, cy), (cx + 15, cy)], fill=(255, 150, 150))
        
        # Whiskers
        for i in range(3):
            y_off = i * 15 - 15
            draw.line([cx - 80, cy + y_off, cx - 150, cy + y_off - 10], fill=(200, 200, 200), width=2)
            draw.line([cx + 80, cy + y_off, cx + 150, cy + y_off - 10], fill=(200, 200, 200), width=2)
    
    def _generate_ascii(self, prompt: str) -> str:
        """Fallback ASCII art"""
        art = """
    /\\_/\\  
   ( o.o ) 
    > ^ <
   /|   |\\
  (_|   |_)
        
  [Cat R1 Generated Image]
  Prompt: {}
        """.format(prompt[:50])
        return art.encode('utf-8')


# ═══════════════════════════════════════════════════════════════════════════════
# WEB SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

class WebSearch:
    """Simple web search functionality"""
    
    def __init__(self):
        self.cache = {}
    
    def search(self, query: str) -> List[Dict]:
        """Search the web"""
        if query in self.cache:
            return self.cache[query]
        
        results = []
        
        try:
            # Try DuckDuckGo instant answers
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            req = urllib.request.Request(url, headers={'User-Agent': 'Cat R1/2.0'})
            with urllib.request.urlopen(req, timeout=5, context=ctx) as response:
                data = json.loads(response.read().decode('utf-8'))
                
                if data.get('Abstract'):
                    results.append({
                        'title': data.get('Heading', query),
                        'snippet': data.get('Abstract', ''),
                        'url': data.get('AbstractURL', ''),
                        'source': data.get('AbstractSource', 'DuckDuckGo')
                    })
                
                for topic in data.get('RelatedTopics', [])[:5]:
                    if isinstance(topic, dict) and 'Text' in topic:
                        results.append({
                            'title': topic.get('Text', '')[:50],
                            'snippet': topic.get('Text', ''),
                            'url': topic.get('FirstURL', ''),
                            'source': 'DuckDuckGo'
                        })
        
        except Exception as e:
            # Fallback to simulated results
            results = [
                {'title': f'Search result for: {query}', 'snippet': 'Web search is currently unavailable. Try again later.', 'url': '', 'source': 'Offline'}
            ]
        
        self.cache[query] = results
        return results
    
    def fetch_url(self, url: str) -> str:
        """Fetch URL content"""
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            req = urllib.request.Request(url, headers={'User-Agent': 'Cat R1/2.0'})
            with urllib.request.urlopen(req, timeout=10, context=ctx) as response:
                content = response.read().decode('utf-8', errors='ignore')
                # Strip HTML
                content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
                content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL)
                content = re.sub(r'<[^>]+>', ' ', content)
                content = html.unescape(content)
                content = re.sub(r'\s+', ' ', content)
                return content[:5000]
        except Exception as e:
            return f"Error fetching URL: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class DataAnalyzer:
    """Analyze data and create visualizations"""
    
    def __init__(self):
        pass
    
    def analyze_csv(self, content: str) -> Dict:
        """Analyze CSV data"""
        lines = content.strip().split('\n')
        reader = csv.reader(lines)
        rows = list(reader)
        
        if not rows:
            return {'error': 'Empty CSV'}
        
        headers = rows[0]
        data = rows[1:]
        
        result = {
            'rows': len(data),
            'columns': len(headers),
            'headers': headers,
            'preview': data[:5],
            'stats': {}
        }
        
        # Try to compute stats for numeric columns
        for i, header in enumerate(headers):
            try:
                values = [float(row[i]) for row in data if row[i].replace('.', '').replace('-', '').isdigit()]
                if values:
                    result['stats'][header] = {
                        'min': min(values),
                        'max': max(values),
                        'mean': sum(values) / len(values),
                        'count': len(values)
                    }
            except:
                pass
        
        return result
    
    def analyze_json(self, content: str) -> Dict:
        """Analyze JSON data"""
        try:
            data = json.loads(content)
            
            if isinstance(data, list):
                return {
                    'type': 'array',
                    'length': len(data),
                    'preview': data[:5] if len(data) > 5 else data,
                    'item_type': type(data[0]).__name__ if data else 'unknown'
                }
            elif isinstance(data, dict):
                return {
                    'type': 'object',
                    'keys': list(data.keys()),
                    'preview': {k: str(v)[:100] for k, v in list(data.items())[:10]}
                }
            else:
                return {'type': type(data).__name__, 'value': str(data)[:1000]}
        
        except json.JSONDecodeError as e:
            return {'error': f'Invalid JSON: {e}'}
    
    def create_chart(self, data: Dict, chart_type: str = 'bar') -> Optional[bytes]:
        """Create chart from data"""
        if not HAS_MATPLOTLIB:
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0a0a0a')
            ax.set_facecolor('#0a0a0a')
            
            if 'labels' in data and 'values' in data:
                labels = data['labels']
                values = data['values']
                
                if chart_type == 'bar':
                    bars = ax.bar(labels, values, color='#00ff00')
                elif chart_type == 'line':
                    ax.plot(labels, values, color='#00ff00', marker='o')
                elif chart_type == 'pie':
                    ax.pie(values, labels=labels, colors=plt.cm.Greens(np.linspace(0.3, 0.9, len(values))))
            
            ax.tick_params(colors='#00ff00')
            ax.spines['bottom'].set_color('#00ff00')
            ax.spines['left'].set_color('#00ff00')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            if 'title' in data:
                ax.set_title(data['title'], color='#00ff00')
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#0a0a0a')
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()
        
        except Exception as e:
            print(f"Chart error: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# FILE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class FileManager:
    """Handle file operations"""
    
    def __init__(self):
        self.files = {}  # Virtual file storage
        self.upload_dir = tempfile.mkdtemp()
    
    def upload(self, path: str) -> Dict:
        """Upload and analyze file"""
        try:
            with open(path, 'rb') as f:
                content = f.read()
            
            name = os.path.basename(path)
            ext = os.path.splitext(name)[1].lower()
            
            file_info = {
                'name': name,
                'path': path,
                'size': len(content),
                'ext': ext,
                'content': content,
            }
            
            # Text analysis
            if ext in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.csv', '.xml']:
                text = content.decode('utf-8', errors='ignore')
                file_info['text'] = text
                file_info['lines'] = len(text.split('\n'))
                file_info['words'] = len(text.split())
            
            # Image info
            elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                file_info['type'] = 'image'
                if HAS_PIL:
                    img = Image.open(io.BytesIO(content))
                    file_info['dimensions'] = img.size
                    file_info['mode'] = img.mode
            
            self.files[name] = file_info
            return file_info
        
        except Exception as e:
            return {'error': str(e)}
    
    def create_file(self, name: str, content: str) -> str:
        """Create a new file"""
        path = os.path.join(self.upload_dir, name)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return path
    
    def list_files(self) -> List[str]:
        return list(self.files.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# VOICE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class VoiceSystem:
    """Text-to-speech and voice handling"""
    
    def __init__(self):
        self.enabled = False
        self.engine = None
        
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.enabled = True
        except:
            pass
    
    def speak(self, text: str):
        """Speak text"""
        if self.enabled and self.engine:
            def run():
                self.engine.say(text)
                self.engine.runAndWait()
            threading.Thread(target=run, daemon=True).start()
    
    def set_rate(self, rate: int = 150):
        if self.engine:
            self.engine.setProperty('rate', rate)


# ═══════════════════════════════════════════════════════════════════════════════
# PLUGIN SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class PluginManager:
    """Manage plugins/extensions"""
    
    def __init__(self):
        self.plugins = {}
        self._load_default_plugins()
    
    def _load_default_plugins(self):
        """Load built-in plugins"""
        self.plugins['calculator'] = {
            'name': 'Calculator',
            'description': 'Advanced math calculations',
            'handler': self._calc_handler
        }
        self.plugins['translator'] = {
            'name': 'Translator',
            'description': 'Text translation (simulated)',
            'handler': self._translate_handler
        }
        self.plugins['timer'] = {
            'name': 'Timer',
            'description': 'Set timers and reminders',
            'handler': self._timer_handler
        }
        self.plugins['weather'] = {
            'name': 'Weather',
            'description': 'Weather information',
            'handler': self._weather_handler
        }
    
    def _calc_handler(self, args: str) -> str:
        try:
            # Safe eval for math
            allowed = {'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'sqrt': np.sqrt,
                      'log': np.log, 'exp': np.exp, 'pi': np.pi, 'e': np.e,
                      'abs': abs, 'pow': pow}
            result = eval(args, {"__builtins__": {}}, allowed)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"
    
    def _translate_handler(self, args: str) -> str:
        # Simulated translation
        return f"Translation (simulated): {args[::-1]}"
    
    def _timer_handler(self, args: str) -> str:
        try:
            seconds = int(args)
            return f"Timer set for {seconds} seconds"
        except:
            return "Usage: timer <seconds>"
    
    def _weather_handler(self, args: str) -> str:
        # Simulated weather
        conditions = ['Sunny', 'Cloudy', 'Rainy', 'Partly Cloudy', 'Clear']
        temp = random.randint(15, 35)
        return f"Weather in {args}: {random.choice(conditions)}, {temp}°C"
    
    def run_plugin(self, name: str, args: str) -> str:
        if name in self.plugins:
            return self.plugins[name]['handler'](args)
        return f"Plugin '{name}' not found"
    
    def list_plugins(self) -> List[Dict]:
        return [{'name': k, **v} for k, v in self.plugins.items()]


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class MemorySystem:
    """Persistent memory across conversations"""
    
    def __init__(self, db_path: str = "cat_r1_memory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
    
    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                key TEXT UNIQUE,
                value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                title TEXT,
                messages TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()
    
    def remember(self, key: str, value: str):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO memories (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (key, value))
        self.conn.commit()
    
    def recall(self, key: str) -> Optional[str]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT value FROM memories WHERE key = ?', (key,))
        row = cursor.fetchone()
        return row[0] if row else None
    
    def forget(self, key: str):
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM memories WHERE key = ?', (key,))
        self.conn.commit()
    
    def list_memories(self) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT key, value, updated_at FROM memories ORDER BY updated_at DESC')
        return [{'key': r[0], 'value': r[1], 'updated': r[2]} for r in cursor.fetchall()]
    
    def save_conversation(self, title: str, messages: List[Dict]) -> int:
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (title, messages) VALUES (?, ?)
        ''', (title, json.dumps(messages)))
        self.conn.commit()
        return cursor.lastrowid
    
    def load_conversations(self) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, title, created_at FROM conversations ORDER BY created_at DESC')
        return [{'id': r[0], 'title': r[1], 'created': r[2]} for r in cursor.fetchall()]
    
    def load_conversation(self, id: int) -> Optional[List[Dict]]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT messages FROM conversations WHERE id = ?', (id,))
        row = cursor.fetchone()
        return json.loads(row[0]) if row else None


# ═══════════════════════════════════════════════════════════════════════════════
# CANVAS / ARTIFACTS SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class CanvasSystem:
    """Create and manage document artifacts"""
    
    def __init__(self):
        self.artifacts = {}
    
    def create(self, name: str, content: str, artifact_type: str = 'text') -> Dict:
        artifact = {
            'name': name,
            'content': content,
            'type': artifact_type,
            'created': datetime.now().isoformat(),
            'modified': datetime.now().isoformat()
        }
        self.artifacts[name] = artifact
        return artifact
    
    def update(self, name: str, content: str):
        if name in self.artifacts:
            self.artifacts[name]['content'] = content
            self.artifacts[name]['modified'] = datetime.now().isoformat()
    
    def get(self, name: str) -> Optional[Dict]:
        return self.artifacts.get(name)
    
    def list(self) -> List[str]:
        return list(self.artifacts.keys())
    
    def export(self, name: str, format: str = 'txt') -> Optional[str]:
        if name not in self.artifacts:
            return None
        
        content = self.artifacts[name]['content']
        
        if format == 'md':
            return content
        elif format == 'html':
            # Simple markdown to HTML
            html_content = f"<html><body><pre>{html.escape(content)}</pre></body></html>"
            return html_content
        elif format == 'json':
            return json.dumps(self.artifacts[name], indent=2)
        else:
            return content


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class CatR1UltimateApp(tk.Tk):
    """Cat R1 4B Ultimate - Full ChatGPT Clone"""
    
    def __init__(self):
        super().__init__()
        
        self.title(f"{APP_NAME} v{VERSION}")
        self.geometry("1600x1000")
        self.minsize(1200, 800)
        
        # Theme
        self.theme = {
            "bg": "#0a0a0a",
            "bg2": "#111111",
            "bg3": "#1a1a1a",
            "sidebar": "#050505",
            "text": "#00ff00",
            "text2": "#ffffff",
            "muted": "#00aa00",
            "accent": "#00ff00",
            "button_bg": "#003300",
            "button_fg": "#00ff00",
            "button_active": "#00ff00",
            "error": "#ff0000",
            "warning": "#ffaa00",
            "border": "#003300",
            "user": "#00ffff",
            "assistant": "#00ff00",
            "system": "#ffff00",
        }
        
        # Initialize components
        self.model = CatR1Model()
        self.knowledge = KnowledgeBase()
        self.interpreter = CodeInterpreter()
        self.image_gen = ImageGenerator()
        self.web_search = WebSearch()
        self.data_analyzer = DataAnalyzer()
        self.file_manager = FileManager()
        self.voice = VoiceSystem()
        self.plugins = PluginManager()
        self.memory = MemorySystem()
        self.canvas = CanvasSystem()
        
        # State
        self.messages = []
        self.current_conversation_id = None
        self.streaming = False
        self.voice_enabled = tk.BooleanVar(value=False)
        self.web_enabled = tk.BooleanVar(value=True)
        self.temp_var = tk.DoubleVar(value=0.8)
        
        # Build UI
        self._build_ui()
        self._apply_theme()
        self._show_welcome()
    
    def _build_ui(self):
        """Build the complete UI"""
        self.configure(bg=self.theme["bg"])
        
        # Grid config
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # Left sidebar
        self._build_sidebar()
        
        # Main area
        self._build_main_area()
        
        # Right panel (code + tools)
        self._build_right_panel()
        
        # Bottom toolbar
        self._build_toolbar()
    
    def _build_sidebar(self):
        """Build left sidebar with conversations and settings"""
        self.sidebar = tk.Frame(self, width=280, bg=self.theme["sidebar"])
        self.sidebar.grid(row=0, column=0, sticky="nsew", rowspan=2)
        self.sidebar.grid_propagate(False)
        
        # Logo
        logo_frame = tk.Frame(self.sidebar, bg=self.theme["sidebar"])
        logo_frame.pack(fill="x", pady=15)
        
        tk.Label(
            logo_frame,
            text="🐱 Cat R1 4B",
            font=("Courier", 20, "bold"),
            bg=self.theme["sidebar"],
            fg=self.theme["accent"]
        ).pack()
        
        tk.Label(
            logo_frame,
            text="Ultimate Edition",
            font=("Courier", 10),
            bg=self.theme["sidebar"],
            fg=self.theme["muted"]
        ).pack()
        
        # New chat button
        tk.Button(
            self.sidebar,
            text="➕ New Chat",
            font=("Courier", 11),
            bg=self.theme["button_bg"],
            fg=self.theme["button_fg"],
            activebackground=self.theme["button_active"],
            relief="flat",
            cursor="hand2",
            command=self._new_chat
        ).pack(fill="x", padx=15, pady=10)
        
        # Conversations list
        tk.Label(
            self.sidebar,
            text="💬 Conversations",
            font=("Courier", 10, "bold"),
            bg=self.theme["sidebar"],
            fg=self.theme["muted"]
        ).pack(anchor="w", padx=15, pady=(10, 5))
        
        conv_frame = tk.Frame(self.sidebar, bg=self.theme["sidebar"])
        conv_frame.pack(fill="both", expand=True, padx=10)
        
        self.conv_listbox = tk.Listbox(
            conv_frame,
            font=("Courier", 10),
            bg=self.theme["bg2"],
            fg=self.theme["text"],
            selectbackground=self.theme["button_bg"],
            relief="flat",
            highlightthickness=0
        )
        self.conv_listbox.pack(fill="both", expand=True)
        self.conv_listbox.bind('<<ListboxSelect>>', self._on_conversation_select)
        
        self._refresh_conversations()
        
        # Settings section
        settings_frame = tk.Frame(self.sidebar, bg=self.theme["sidebar"])
        settings_frame.pack(fill="x", padx=15, pady=10)
        
        tk.Label(
            settings_frame,
            text="⚙️ Settings",
            font=("Courier", 10, "bold"),
            bg=self.theme["sidebar"],
            fg=self.theme["muted"]
        ).pack(anchor="w")
        
        # Temperature
        temp_frame = tk.Frame(settings_frame, bg=self.theme["sidebar"])
        temp_frame.pack(fill="x", pady=5)
        tk.Label(temp_frame, text="Temp:", bg=self.theme["sidebar"], fg=self.theme["muted"], font=("Courier", 9)).pack(side="left")
        tk.Scale(
            temp_frame, from_=0.1, to=2.0, resolution=0.1, orient="horizontal",
            variable=self.temp_var, bg=self.theme["sidebar"], fg=self.theme["text"],
            highlightthickness=0, troughcolor=self.theme["button_bg"], length=150
        ).pack(side="left", fill="x", expand=True)
        
        # Toggles
        tk.Checkbutton(
            settings_frame, text="🔊 Voice", variable=self.voice_enabled,
            bg=self.theme["sidebar"], fg=self.theme["text"],
            selectcolor=self.theme["bg2"], font=("Courier", 9),
            activebackground=self.theme["sidebar"]
        ).pack(anchor="w")
        
        tk.Checkbutton(
            settings_frame, text="🌐 Web Search", variable=self.web_enabled,
            bg=self.theme["sidebar"], fg=self.theme["text"],
            selectcolor=self.theme["bg2"], font=("Courier", 9),
            activebackground=self.theme["sidebar"]
        ).pack(anchor="w")
        
        # Model info
        info_frame = tk.Frame(self.sidebar, bg=self.theme["sidebar"])
        info_frame.pack(fill="x", padx=15, pady=10)
        
        tk.Label(
            info_frame,
            text=f"Params: {self.model.param_count:,}",
            font=("Courier", 9),
            bg=self.theme["sidebar"],
            fg=self.theme["muted"]
        ).pack(anchor="w")
        
        self.status_label = tk.Label(
            info_frame,
            text=f"Status: {'Trained ✓' if self.model.trained else 'Untrained'}",
            font=("Courier", 9),
            bg=self.theme["sidebar"],
            fg=self.theme["muted"]
        )
        self.status_label.pack(anchor="w")
    
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
            wrap="word",
            state="disabled",
            padx=15,
            pady=15,
            insertbackground=self.theme["text"]
        )
        self.chat_display.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Tags
        self.chat_display.tag_configure("user", foreground=self.theme["user"])
        self.chat_display.tag_configure("assistant", foreground=self.theme["assistant"])
        self.chat_display.tag_configure("system", foreground=self.theme["system"])
        self.chat_display.tag_configure("error", foreground=self.theme["error"])
        self.chat_display.tag_configure("code", foreground="#ff00ff", font=("Courier", 10))
        self.chat_display.tag_configure("bold", font=("Courier", 11, "bold"))
        
        # Input area
        input_frame = tk.Frame(self.main_frame, bg=self.theme["bg"])
        input_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        input_frame.grid_columnconfigure(0, weight=1)
        
        # Attachment buttons
        attach_frame = tk.Frame(input_frame, bg=self.theme["bg"])
        attach_frame.grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        for icon, cmd, tip in [
            ("📎", self._upload_file, "Upload file"),
            ("🖼️", self._generate_image_dialog, "Generate image"),
            ("📊", self._analyze_data_dialog, "Analyze data"),
            ("🌐", self._web_search_dialog, "Web search"),
            ("🧠", self._memory_dialog, "Memory"),
            ("📝", self._canvas_dialog, "Canvas"),
            ("🔌", self._plugins_dialog, "Plugins"),
        ]:
            btn = tk.Button(
                attach_frame, text=icon, font=("Courier", 12),
                bg=self.theme["bg"], fg=self.theme["text"],
                relief="flat", cursor="hand2", command=cmd
            )
            btn.pack(side="left", padx=2)
        
        # Text input
        self.input_text = tk.Text(
            input_frame,
            height=3,
            font=("Courier", 11),
            bg=self.theme["bg3"],
            fg=self.theme["text"],
            insertbackground=self.theme["text"],
            wrap="word",
            padx=10,
            pady=10
        )
        self.input_text.grid(row=1, column=0, sticky="ew")
        self.input_text.bind("<Return>", self._on_enter)
        self.input_text.bind("<Shift-Return>", lambda e: None)
        
        # Send button
        btn_frame = tk.Frame(input_frame, bg=self.theme["bg"])
        btn_frame.grid(row=1, column=1, padx=(10, 0))
        
        self.send_btn = tk.Button(
            btn_frame,
            text="Send ➤",
            font=("Courier", 11, "bold"),
            bg=self.theme["button_bg"],
            fg=self.theme["button_fg"],
            activebackground=self.theme["button_active"],
            relief="flat",
            cursor="hand2",
            width=10,
            command=self._send_message
        )
        self.send_btn.pack(pady=2)
        
        self.stop_btn = tk.Button(
            btn_frame,
            text="⏹ Stop",
            font=("Courier", 10),
            bg="#330000",
            fg="#ff0000",
            relief="flat",
            cursor="hand2",
            width=10,
            command=self._stop_generation
        )
        self.stop_btn.pack(pady=2)
    
    def _build_right_panel(self):
        """Build right panel with code interpreter and tools"""
        self.right_panel = tk.Frame(self, width=450, bg=self.theme["sidebar"])
        self.right_panel.grid(row=0, column=2, sticky="nsew", rowspan=2)
        self.right_panel.grid_propagate(False)
        
        # Notebook for tabs
        style = ttk.Style()
        style.configure('TNotebook', background=self.theme["sidebar"])
        style.configure('TNotebook.Tab', background=self.theme["button_bg"], foreground=self.theme["text"])
        
        self.notebook = ttk.Notebook(self.right_panel)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Code tab
        code_frame = tk.Frame(self.notebook, bg=self.theme["bg2"])
        self.notebook.add(code_frame, text="💻 Code")
        
        tk.Label(
            code_frame,
            text="Python Interpreter",
            font=("Courier", 12, "bold"),
            bg=self.theme["bg2"],
            fg=self.theme["accent"]
        ).pack(pady=10)
        
        self.code_input = scrolledtext.ScrolledText(
            code_frame,
            font=("Courier", 10),
            bg=self.theme["bg3"],
            fg=self.theme["text"],
            insertbackground=self.theme["text"],
            height=12
        )
        self.code_input.pack(fill="x", padx=10, pady=5)
        
        code_btn_frame = tk.Frame(code_frame, bg=self.theme["bg2"])
        code_btn_frame.pack(fill="x", padx=10)
        
        tk.Button(
            code_btn_frame, text="▶ Run", font=("Courier", 10, "bold"),
            bg=self.theme["button_bg"], fg=self.theme["button_fg"],
            relief="flat", command=self._run_code
        ).pack(side="left", padx=2)
        
        tk.Button(
            code_btn_frame, text="🗑 Clear", font=("Courier", 10),
            bg=self.theme["button_bg"], fg=self.theme["button_fg"],
            relief="flat", command=lambda: self.code_input.delete("1.0", tk.END)
        ).pack(side="left", padx=2)
        
        tk.Button(
            code_btn_frame, text="🔄 Reset", font=("Courier", 10),
            bg=self.theme["button_bg"], fg=self.theme["button_fg"],
            relief="flat", command=self._reset_interpreter
        ).pack(side="left", padx=2)
        
        tk.Label(code_frame, text="Output:", font=("Courier", 10),
                bg=self.theme["bg2"], fg=self.theme["muted"]).pack(anchor="w", padx=10, pady=(10, 0))
        
        self.code_output = scrolledtext.ScrolledText(
            code_frame,
            font=("Courier", 10),
            bg=self.theme["bg3"],
            fg=self.theme["text"],
            state="disabled",
            height=10
        )
        self.code_output.pack(fill="both", expand=True, padx=10, pady=5)
        self.code_output.tag_configure("error", foreground="#ff0000")
        self.code_output.tag_configure("result", foreground="#00ffff")
        
        # Image tab
        image_frame = tk.Frame(self.notebook, bg=self.theme["bg2"])
        self.notebook.add(image_frame, text="🖼️ Image")
        
        tk.Label(
            image_frame,
            text="Image Generator",
            font=("Courier", 12, "bold"),
            bg=self.theme["bg2"],
            fg=self.theme["accent"]
        ).pack(pady=10)
        
        self.image_canvas = tk.Canvas(
            image_frame,
            width=400,
            height=400,
            bg=self.theme["bg3"],
            highlightthickness=0
        )
        self.image_canvas.pack(pady=10)
        
        # Files tab
        files_frame = tk.Frame(self.notebook, bg=self.theme["bg2"])
        self.notebook.add(files_frame, text="📁 Files")
        
        tk.Label(
            files_frame,
            text="File Manager",
            font=("Courier", 12, "bold"),
            bg=self.theme["bg2"],
            fg=self.theme["accent"]
        ).pack(pady=10)
        
        self.files_listbox = tk.Listbox(
            files_frame,
            font=("Courier", 10),
            bg=self.theme["bg3"],
            fg=self.theme["text"],
            height=10
        )
        self.files_listbox.pack(fill="x", padx=10, pady=5)
        
        files_btn_frame = tk.Frame(files_frame, bg=self.theme["bg2"])
        files_btn_frame.pack(fill="x", padx=10)
        
        tk.Button(
            files_btn_frame, text="📤 Upload", font=("Courier", 10),
            bg=self.theme["button_bg"], fg=self.theme["button_fg"],
            relief="flat", command=self._upload_file
        ).pack(side="left", padx=2)
        
        tk.Button(
            files_btn_frame, text="📥 Download", font=("Courier", 10),
            bg=self.theme["button_bg"], fg=self.theme["button_fg"],
            relief="flat", command=self._download_file
        ).pack(side="left", padx=2)
        
        self.file_preview = scrolledtext.ScrolledText(
            files_frame,
            font=("Courier", 9),
            bg=self.theme["bg3"],
            fg=self.theme["text"],
            height=8
        )
        self.file_preview.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Canvas tab
        canvas_frame = tk.Frame(self.notebook, bg=self.theme["bg2"])
        self.notebook.add(canvas_frame, text="📝 Canvas")
        
        tk.Label(
            canvas_frame,
            text="Document Canvas",
            font=("Courier", 12, "bold"),
            bg=self.theme["bg2"],
            fg=self.theme["accent"]
        ).pack(pady=10)
        
        self.canvas_text = scrolledtext.ScrolledText(
            canvas_frame,
            font=("Courier", 11),
            bg=self.theme["bg3"],
            fg=self.theme["text"],
            wrap="word"
        )
        self.canvas_text.pack(fill="both", expand=True, padx=10, pady=5)
        
        canvas_btn_frame = tk.Frame(canvas_frame, bg=self.theme["bg2"])
        canvas_btn_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Button(
            canvas_btn_frame, text="💾 Save", font=("Courier", 10),
            bg=self.theme["button_bg"], fg=self.theme["button_fg"],
            relief="flat", command=self._save_canvas
        ).pack(side="left", padx=2)
        
        tk.Button(
            canvas_btn_frame, text="📤 Export", font=("Courier", 10),
            bg=self.theme["button_bg"], fg=self.theme["button_fg"],
            relief="flat", command=self._export_canvas
        ).pack(side="left", padx=2)
    
    def _build_toolbar(self):
        """Build bottom toolbar"""
        self.toolbar = tk.Frame(self, height=30, bg=self.theme["sidebar"])
        self.toolbar.grid(row=1, column=1, sticky="ew")
        
        # Status
        self.toolbar_status = tk.Label(
            self.toolbar,
            text="Ready",
            font=("Courier", 9),
            bg=self.theme["sidebar"],
            fg=self.theme["muted"]
        )
        self.toolbar_status.pack(side="left", padx=10)
        
        # Quick actions
        for text, cmd in [
            ("🔧 Train", self._train_dialog),
            ("💾 Save Model", self._save_model),
            ("📂 Load Model", self._load_model),
            ("📦 Export GGUF", self._export_gguf),
        ]:
            tk.Button(
                self.toolbar, text=text, font=("Courier", 9),
                bg=self.theme["sidebar"], fg=self.theme["muted"],
                relief="flat", command=cmd
            ).pack(side="right", padx=5)
    
    def _apply_theme(self):
        pass
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MESSAGE HANDLING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _add_message(self, role: str, content: str, tag: str = None):
        """Add message to chat"""
        self.chat_display.configure(state="normal")
        
        icons = {"user": "🧑", "assistant": "🐱", "system": "⚙️", "error": "❌"}
        names = {"user": "You", "assistant": "Cat R1", "system": "System", "error": "Error"}
        
        self.chat_display.insert(tk.END, f"\n{icons.get(role, '•')} {names.get(role, role)}:\n", tag or role)
        self.chat_display.insert(tk.END, f"{content}\n")
        
        self.chat_display.configure(state="disabled")
        self.chat_display.see(tk.END)
        
        self.messages.append({"role": role, "content": content})
        
        if self.voice_enabled.get() and role == "assistant":
            self.voice.speak(content[:500])
    
    def _show_welcome(self):
        """Show welcome message"""
        self._add_message("system", f"""🐱 Welcome to Cat R1 4B Ultimate!

Neural Network: {self.model.param_count:,} parameters
Architecture: {self.model.num_layers}-layer LSTM
PIL/Images: {'✓' if HAS_PIL else '✗'}
Matplotlib: {'✓' if HAS_MATPLOTLIB else '✗'}

**Features:**
💬 Chat • 💻 Code Interpreter • 🖼️ Image Generation
🌐 Web Search • 📊 Data Analysis • 📁 File Handling
🧠 Memory • 📝 Canvas • 🔌 Plugins • 🔊 Voice

Type naturally or use /commands!
/help - Show all commands""")
    
    def _on_enter(self, event):
        if not (event.state & 0x1):
            self._send_message()
            return "break"
    
    def _send_message(self):
        """Send user message"""
        text = self.input_text.get("1.0", tk.END).strip()
        if not text or self.streaming:
            return
        
        self.input_text.delete("1.0", tk.END)
        self._add_message("user", text)
        
        # Process message
        self._process_message(text)
    
    def _process_message(self, text: str):
        """Process user message"""
        text_lower = text.lower()
        
        # Commands
        if text.startswith("/"):
            self._handle_command(text)
            return
        
        # Check for special intents
        if any(w in text_lower for w in ['generate image', 'create image', 'draw', 'make a picture', 'dall-e']):
            self._generate_image(text)
            return
        
        if any(w in text_lower for w in ['search for', 'look up', 'google', 'find info']):
            if self.web_enabled.get():
                self._do_web_search(text)
                return
        
        if any(w in text_lower for w in ['run code', 'execute', 'python:']):
            code = text.replace('run code', '').replace('execute', '').replace('python:', '').strip()
            self._execute_code(code)
            return
        
        if any(w in text_lower for w in ['remember that', 'remember my', 'save to memory']):
            key = hashlib.md5(text.encode()).hexdigest()[:8]
            self.memory.remember(key, text)
            self._add_message("assistant", f"I'll remember that! (Key: {key})")
            return
        
        # Knowledge base response
        category, response = self.knowledge.find_response(text)
        if category:
            self._add_message("assistant", response)
            return
        
        # Use LLM
        self._generate_llm_response(text)
    
    def _handle_command(self, text: str):
        """Handle slash commands"""
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        commands = {
            "/help": self._show_help,
            "/train": lambda: self._train_dialog() if not arg else self._quick_train(arg),
            "/code": lambda: self._execute_code(arg),
            "/image": lambda: self._generate_image(arg),
            "/search": lambda: self._do_web_search(arg),
            "/remember": lambda: self._remember(arg),
            "/recall": lambda: self._recall(arg),
            "/forget": lambda: self._forget(arg),
            "/memories": self._list_memories,
            "/canvas": lambda: self._create_canvas(arg),
            "/plugin": lambda: self._run_plugin(arg),
            "/plugins": self._list_plugins,
            "/save": self._save_model,
            "/load": self._load_model,
            "/export": self._export_gguf,
            "/clear": self._new_chat,
            "/voice": self._toggle_voice,
        }
        
        if cmd in commands:
            commands[cmd]()
        else:
            self._add_message("error", f"Unknown command: {cmd}\nUse /help for list of commands")
    
    def _show_help(self):
        """Show help"""
        self._add_message("system", """**Commands:**
/help - Show this help
/train <text> - Train the model
/code <python> - Execute Python code
/image <prompt> - Generate image
/search <query> - Web search
/remember <text> - Save to memory
/recall <key> - Recall memory
/memories - List all memories
/canvas <name> - Create canvas document
/plugin <name> <args> - Run plugin
/plugins - List plugins
/save - Save model
/load - Load model
/export - Export GGUF
/clear - New chat
/voice - Toggle voice""")
    
    def _generate_llm_response(self, prompt: str):
        """Generate response using LLM"""
        self.streaming = True
        self.toolbar_status.configure(text="Generating...")
        
        def generate():
            if self.model.trained:
                response = self.model.generate(
                    prompt,
                    max_length=200,
                    temperature=self.temp_var.get()
                )
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()
            else:
                response = "I haven't been trained yet! Use /train or click 🔧 Train to teach me."
            
            self.after(0, lambda: self._add_message("assistant", response))
            self.after(0, lambda: self.toolbar_status.configure(text="Ready"))
            self.streaming = False
        
        threading.Thread(target=generate, daemon=True).start()
    
    def _stop_generation(self):
        self.streaming = False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FEATURES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _run_code(self):
        """Run code from interpreter panel"""
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
        
        # Handle figures
        if result['figures'] and HAS_PIL:
            for fig_data in result['figures']:
                self._display_image(fig_data)
        
        self.code_output.configure(state="disabled")
    
    def _execute_code(self, code: str):
        """Execute code and show in chat"""
        if not code:
            return
        
        result = self.interpreter.execute(code)
        
        output = ""
        if result['output']:
            output += result['output']
        if result['result'] is not None:
            output += f"\n→ {result['result']}"
        if result['error']:
            output += f"\n❌ {result['error']}"
        
        self._add_message("assistant", f"```\n{output}\n```")
    
    def _reset_interpreter(self):
        self.interpreter.reset()
        self.code_output.configure(state="normal")
        self.code_output.delete("1.0", tk.END)
        self.code_output.insert(tk.END, "Interpreter reset!")
        self.code_output.configure(state="disabled")
    
    def _generate_image(self, prompt: str):
        """Generate image from prompt"""
        if not prompt:
            prompt = "abstract art"
        
        self._add_message("assistant", f"🎨 Generating image: {prompt}")
        self.toolbar_status.configure(text="Generating image...")
        
        def generate():
            try:
                img_data = self.image_gen.generate(prompt)
                self.after(0, lambda: self._display_image(img_data))
                self.after(0, lambda: self._add_message("assistant", "Image generated! Check the Image tab."))
            except Exception as e:
                self.after(0, lambda: self._add_message("error", f"Image generation failed: {e}"))
            self.after(0, lambda: self.toolbar_status.configure(text="Ready"))
        
        threading.Thread(target=generate, daemon=True).start()
    
    def _display_image(self, img_data: bytes):
        """Display image on canvas"""
        if not HAS_PIL:
            return
        
        try:
            img = Image.open(io.BytesIO(img_data))
            img = img.resize((400, 400), Image.Resampling.LANCZOS)
            self.current_image = ImageTk.PhotoImage(img)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(200, 200, image=self.current_image)
            self.notebook.select(1)  # Switch to image tab
        except Exception as e:
            print(f"Image display error: {e}")
    
    def _do_web_search(self, query: str):
        """Perform web search"""
        if not query:
            return
        
        self._add_message("assistant", f"🔍 Searching: {query}")
        self.toolbar_status.configure(text="Searching...")
        
        def search():
            results = self.web_search.search(query)
            
            response = "**Search Results:**\n\n"
            for r in results[:5]:
                response += f"• **{r['title']}**\n  {r['snippet'][:200]}\n\n"
            
            self.after(0, lambda: self._add_message("assistant", response))
            self.after(0, lambda: self.toolbar_status.configure(text="Ready"))
        
        threading.Thread(target=search, daemon=True).start()
    
    def _upload_file(self):
        """Upload file"""
        path = filedialog.askopenfilename()
        if path:
            info = self.file_manager.upload(path)
            
            self.files_listbox.insert(tk.END, info.get('name', 'unknown'))
            
            preview = f"Name: {info.get('name')}\nSize: {info.get('size', 0)} bytes\n"
            if 'lines' in info:
                preview += f"Lines: {info['lines']}\nWords: {info['words']}\n"
            if 'dimensions' in info:
                preview += f"Dimensions: {info['dimensions']}\n"
            if 'text' in info:
                preview += f"\nPreview:\n{info['text'][:500]}"
            
            self.file_preview.delete("1.0", tk.END)
            self.file_preview.insert(tk.END, preview)
            
            self._add_message("assistant", f"📁 Uploaded: {info.get('name')}")
    
    def _download_file(self):
        """Download/save file"""
        selection = self.files_listbox.curselection()
        if not selection:
            return
        
        name = self.files_listbox.get(selection[0])
        if name in self.file_manager.files:
            path = filedialog.asksaveasfilename(initialfile=name)
            if path:
                with open(path, 'wb') as f:
                    f.write(self.file_manager.files[name]['content'])
                self._add_message("system", f"Saved: {path}")
    
    def _remember(self, text: str):
        if text:
            key = hashlib.md5(text.encode()).hexdigest()[:8]
            self.memory.remember(key, text)
            self._add_message("assistant", f"✓ Remembered with key: {key}")
    
    def _recall(self, key: str):
        value = self.memory.recall(key)
        if value:
            self._add_message("assistant", f"Memory [{key}]: {value}")
        else:
            self._add_message("assistant", f"No memory found for key: {key}")
    
    def _forget(self, key: str):
        self.memory.forget(key)
        self._add_message("assistant", f"Forgot memory: {key}")
    
    def _list_memories(self):
        memories = self.memory.list_memories()
        if memories:
            text = "**Memories:**\n\n"
            for m in memories[:10]:
                text += f"• [{m['key']}] {m['value'][:50]}...\n"
            self._add_message("assistant", text)
        else:
            self._add_message("assistant", "No memories stored.")
    
    def _create_canvas(self, name: str):
        if not name:
            name = f"doc_{int(time.time())}"
        self.canvas.create(name, "", "text")
        self.canvas_text.delete("1.0", tk.END)
        self._add_message("assistant", f"📝 Created canvas: {name}")
        self.notebook.select(3)  # Canvas tab
    
    def _save_canvas(self):
        content = self.canvas_text.get("1.0", tk.END)
        name = f"canvas_{int(time.time())}"
        self.canvas.create(name, content)
        self._add_message("system", f"Canvas saved: {name}")
    
    def _export_canvas(self):
        content = self.canvas_text.get("1.0", tk.END)
        path = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt"), ("HTML", "*.html")]
        )
        if path:
            with open(path, 'w') as f:
                f.write(content)
            self._add_message("system", f"Exported: {path}")
    
    def _run_plugin(self, args: str):
        parts = args.split(maxsplit=1)
        if not parts:
            return
        name = parts[0]
        plugin_args = parts[1] if len(parts) > 1 else ""
        result = self.plugins.run_plugin(name, plugin_args)
        self._add_message("assistant", f"🔌 {result}")
    
    def _list_plugins(self):
        plugins = self.plugins.list_plugins()
        text = "**Available Plugins:**\n\n"
        for p in plugins:
            text += f"• **{p['name']}** - {p['description']}\n"
        self._add_message("assistant", text)
    
    def _toggle_voice(self):
        self.voice_enabled.set(not self.voice_enabled.get())
        status = "enabled" if self.voice_enabled.get() else "disabled"
        self._add_message("system", f"Voice {status}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DIALOGS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _generate_image_dialog(self):
        prompt = tk.simpledialog.askstring("Generate Image", "Enter image prompt:")
        if prompt:
            self._generate_image(prompt)
    
    def _web_search_dialog(self):
        query = tk.simpledialog.askstring("Web Search", "Enter search query:")
        if query:
            self._do_web_search(query)
    
    def _analyze_data_dialog(self):
        self._upload_file()
    
    def _memory_dialog(self):
        self._list_memories()
    
    def _canvas_dialog(self):
        name = tk.simpledialog.askstring("New Canvas", "Document name:")
        if name:
            self._create_canvas(name)
    
    def _plugins_dialog(self):
        self._list_plugins()
    
    def _train_dialog(self):
        """Training dialog"""
        dialog = tk.Toplevel(self)
        dialog.title("Train Cat R1")
        dialog.geometry("600x500")
        dialog.configure(bg=self.theme["bg"])
        
        tk.Label(dialog, text="📚 Train Model", font=("Courier", 16, "bold"),
                bg=self.theme["bg"], fg=self.theme["accent"]).pack(pady=10)
        
        text_area = scrolledtext.ScrolledText(dialog, font=("Courier", 10),
            bg=self.theme["bg2"], fg=self.theme["text"], height=15)
        text_area.pack(fill="both", expand=True, padx=20, pady=10)
        text_area.insert("1.0", "The quick brown fox jumps over the lazy dog.\nHello world! How are you today?\nProgramming is fun. Python is great.")
        
        progress_var = tk.StringVar(value="Ready")
        tk.Label(dialog, textvariable=progress_var, font=("Courier", 10),
                bg=self.theme["bg"], fg=self.theme["text"]).pack()
        
        def train():
            text = text_area.get("1.0", tk.END)
            def update(msg):
                progress_var.set(msg)
                dialog.update()
            self.model.train(text, epochs=10, callback=update)
            self.status_label.configure(text="Status: Trained ✓")
            self._add_message("system", "Training complete!")
            dialog.destroy()
        
        tk.Button(dialog, text="🚀 Train", font=("Courier", 11, "bold"),
            bg=self.theme["button_bg"], fg=self.theme["button_fg"],
            command=lambda: threading.Thread(target=train, daemon=True).start()
        ).pack(pady=10)
    
    def _quick_train(self, text: str):
        def train():
            self.model.train(text, epochs=5)
            self.after(0, lambda: self._add_message("system", "Quick training complete!"))
            self.after(0, lambda: self.status_label.configure(text="Status: Trained ✓"))
        threading.Thread(target=train, daemon=True).start()
    
    def _save_model(self):
        path = filedialog.asksaveasfilename(defaultextension=".catr1",
            filetypes=[("Cat R1 Model", "*.catr1")])
        if path:
            self.model.save(path)
            self._add_message("system", f"Model saved: {path}")
    
    def _load_model(self):
        path = filedialog.askopenfilename(filetypes=[("Cat R1 Model", "*.catr1")])
        if path:
            self.model.load(path)
            self.status_label.configure(text="Status: Loaded ✓")
            self._add_message("system", f"Model loaded: {path}")
    
    def _export_gguf(self):
        path = filedialog.asksaveasfilename(defaultextension=".gguf",
            filetypes=[("GGUF", "*.gguf")])
        if path:
            # Simplified GGUF export
            with open(path, 'wb') as f:
                f.write(b'GGUF')
                pickle.dump({
                    'model_type': 'cat-r1-4b',
                    'params': self.model.param_count,
                    'weights': {
                        'embedding': self.model.embedding.W,
                        'fc_w': self.model.fc.W,
                        'fc_b': self.model.fc.b,
                    }
                }, f)
            self._add_message("system", f"GGUF exported: {path}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONVERSATION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _new_chat(self):
        """Start new chat"""
        if self.messages:
            title = self.messages[0]['content'][:30] if self.messages else "New Chat"
            self.memory.save_conversation(title, self.messages)
            self._refresh_conversations()
        
        self.messages = []
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.configure(state="disabled")
        self._show_welcome()
    
    def _refresh_conversations(self):
        """Refresh conversation list"""
        self.conv_listbox.delete(0, tk.END)
        for conv in self.memory.load_conversations()[:20]:
            self.conv_listbox.insert(tk.END, f"💬 {conv['title'][:25]}")
    
    def _on_conversation_select(self, event):
        """Load selected conversation"""
        selection = self.conv_listbox.curselection()
        if not selection:
            return
        
        convs = self.memory.load_conversations()
        if selection[0] < len(convs):
            messages = self.memory.load_conversation(convs[selection[0]]['id'])
            if messages:
                self.messages = []
                self.chat_display.configure(state="normal")
                self.chat_display.delete("1.0", tk.END)
                self.chat_display.configure(state="disabled")
                
                for msg in messages:
                    self._add_message(msg['role'], msg['content'])


# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE DIALOG (if not available)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from tkinter import simpledialog
    tk.simpledialog = simpledialog
except:
    class SimpleDialog:
        @staticmethod
        def askstring(title, prompt):
            return None
    tk.simpledialog = SimpleDialog()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              CAT R1 4B ULTIMATE - ChatGPT Clone              ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Features:                                                   ║")
    print("║  • Real Neural Network LLM (LSTM)                            ║")
    print("║  • Code Interpreter Sandbox                                  ║")
    print("║  • Image Generation                                          ║")
    print("║  • Web Search                                                ║")
    print("║  • Data Analysis & Charts                                    ║")
    print("║  • File Upload & Management                                  ║")
    print("║  • Memory System (SQLite)                                    ║")
    print("║  • Canvas/Artifacts                                          ║")
    print("║  • Plugin System                                             ║")
    print("║  • Voice Output                                              ║")
    print("║  • Multi-Conversation                                        ║")
    print("║                                                              ║")
    print("║  (C) 2025 Samsoft / Flames Co. / Team Flames                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"PIL/Pillow: {'Available ✓' if HAS_PIL else 'Not installed'}")
    print(f"Matplotlib: {'Available ✓' if HAS_MATPLOTLIB else 'Not installed'}")
    print()
    
    app = CatR1UltimateApp()
    app.mainloop()


if __name__ == "__main__":
    main()
