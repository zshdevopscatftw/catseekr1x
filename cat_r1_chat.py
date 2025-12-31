#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       CAT R1 - PURE PYTHON EDITION                           ║
║              Custom AI Algorithm - No External Dependencies                  ║
║              Markov Chain + Pattern Matching + Knowledge Base                ║
║              (C) 2025 Samsoft / Flames Co.                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import time
import json
import random
import hashlib
import re
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Generator
import threading

APP_NAME = "Cat R1"
VERSION = "1.X"


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM AI ENGINE - Pure Python Implementation
# ═══════════════════════════════════════════════════════════════════════════════

class MarkovChain:
    """
    N-gram Markov Chain for text generation.
    Learns patterns from training corpus and generates coherent text.
    """
    def __init__(self, n: int = 3):
        self.n = n
        self.chain: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
        self.starters: List[Tuple[str, ...]] = []
    
    def train(self, text: str):
        """Train on a text corpus"""
        words = text.split()
        if len(words) < self.n + 1:
            return
        
        for i in range(len(words) - self.n):
            key = tuple(words[i:i + self.n])
            next_word = words[i + self.n]
            self.chain[key].append(next_word)
            
            if i == 0 or words[i - 1].endswith(('.', '!', '?')):
                self.starters.append(key)
    
    def generate(self, max_words: int = 50, seed: Optional[Tuple[str, ...]] = None) -> str:
        """Generate text using the Markov chain"""
        if not self.chain:
            return ""
        
        if seed and seed in self.chain:
            current = seed
        elif self.starters:
            current = random.choice(self.starters)
        else:
            current = random.choice(list(self.chain.keys()))
        
        result = list(current)
        
        for _ in range(max_words - self.n):
            if current not in self.chain:
                break
            next_word = random.choice(self.chain[current])
            result.append(next_word)
            current = tuple(result[-self.n:])
            
            if next_word.endswith(('.', '!', '?')) and len(result) > 10:
                if random.random() < 0.3:
                    break
        
        return ' '.join(result)


class NeuralTextGenerator:
    """
    Simple neural-inspired text generator using weighted probability distributions.
    Mimics attention-like behavior without actual neural networks.
    """
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self.co_occurrence: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.word_freq: Dict[int, int] = defaultdict(int)
        self.total_words = 0
    
    def tokenize(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        words = re.findall(r'\b\w+\b|[.,!?;:]', text.lower())
        tokens = []
        for word in words:
            if word not in self.vocab:
                idx = len(self.vocab)
                self.vocab[word] = idx
                self.reverse_vocab[idx] = word
            tokens.append(self.vocab[word])
        return tokens
    
    def train(self, text: str, window: int = 5):
        """Train co-occurrence matrix from text"""
        tokens = self.tokenize(text)
        self.total_words += len(tokens)
        
        for i, token in enumerate(tokens):
            self.word_freq[token] += 1
            
            for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    weight = 1.0 / distance
                    self.co_occurrence[token][tokens[j]] += weight
    
    def get_context_vector(self, tokens: List[int], decay: float = 0.8) -> Dict[int, float]:
        """Create weighted context vector from recent tokens"""
        context = defaultdict(float)
        weight = 1.0
        
        for token in reversed(tokens[-10:]):
            if token in self.co_occurrence:
                for related, score in self.co_occurrence[token].items():
                    context[related] += score * weight
            weight *= decay
        
        return context
    
    def sample_next(self, context: Dict[int, float], temperature: float = 0.8) -> Optional[int]:
        """Sample next token based on context with temperature"""
        if not context:
            if self.word_freq:
                return random.choice(list(self.word_freq.keys()))
            return None
        
        candidates = list(context.items())
        if not candidates:
            return None
        
        scores = [score ** (1.0 / temperature) for _, score in candidates]
        total = sum(scores)
        if total == 0:
            return random.choice([t for t, _ in candidates])
        
        probs = [s / total for s in scores]
        r = random.random()
        cumulative = 0
        
        for (token, _), prob in zip(candidates, probs):
            cumulative += prob
            if r <= cumulative:
                return token
        
        return candidates[-1][0]
    
    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.8) -> str:
        """Generate text continuation"""
        tokens = self.tokenize(prompt)
        result_tokens = tokens.copy()
        
        for _ in range(max_tokens):
            context = self.get_context_vector(result_tokens)
            next_token = self.sample_next(context, temperature)
            
            if next_token is None:
                break
            
            result_tokens.append(next_token)
            word = self.reverse_vocab.get(next_token, '')
            
            if word in '.!?' and len(result_tokens) > len(tokens) + 5:
                if random.random() < 0.4:
                    break
        
        generated = result_tokens[len(tokens):]
        return ' '.join(self.reverse_vocab.get(t, '') for t in generated)


class ReasoningEngine:
    """
    Chain-of-thought reasoning engine that shows thinking process.
    Mimics Cat R1's reasoning capabilities.
    """
    def __init__(self):
        self.thinking_steps = []
    
    def analyze_query(self, query: str) -> Dict:
        """Analyze query to determine type and required reasoning"""
        query_lower = query.lower()
        
        analysis = {
            'type': 'general',
            'complexity': 'simple',
            'topics': [],
            'requires_math': False,
            'requires_code': False,
            'requires_explanation': False,
            'is_question': '?' in query,
            'sentiment': 'neutral'
        }
        
        if any(w in query_lower for w in ['calculate', 'compute', 'math', 'equation', '+', '-', '*', '/', '=']):
            analysis['requires_math'] = True
            analysis['type'] = 'math'
            analysis['complexity'] = 'medium'
        
        if any(w in query_lower for w in ['code', 'program', 'function', 'algorithm', 'python', 'javascript']):
            analysis['requires_code'] = True
            analysis['type'] = 'coding'
            analysis['complexity'] = 'medium'
        
        if any(w in query_lower for w in ['explain', 'what is', 'how does', 'why', 'describe']):
            analysis['requires_explanation'] = True
            analysis['type'] = 'explanation'
        
        if any(w in query_lower for w in ['hello', 'hi', 'hey', 'greetings']):
            analysis['type'] = 'greeting'
            analysis['sentiment'] = 'positive'
        
        if any(w in query_lower for w in ['thank', 'thanks', 'appreciate']):
            analysis['sentiment'] = 'grateful'
        
        topic_keywords = {
            'science': ['science', 'physics', 'chemistry', 'biology', 'experiment'],
            'technology': ['computer', 'software', 'hardware', 'internet', 'ai', 'machine learning'],
            'history': ['history', 'ancient', 'war', 'civilization', 'century'],
            'philosophy': ['philosophy', 'meaning', 'existence', 'ethics', 'morality'],
            'creative': ['story', 'poem', 'write', 'creative', 'imagine'],
        }
        
        for topic, keywords in topic_keywords.items():
            if any(k in query_lower for k in keywords):
                analysis['topics'].append(topic)
        
        word_count = len(query.split())
        if word_count > 20:
            analysis['complexity'] = 'complex'
        elif word_count > 10:
            analysis['complexity'] = 'medium'
        
        return analysis
    
    def generate_thinking(self, query: str, analysis: Dict) -> List[str]:
        """Generate chain-of-thought reasoning steps"""
        steps = []
        
        steps.append(f"Analyzing the query: \"{query[:50]}{'...' if len(query) > 50 else ''}\"")
        steps.append(f"Query type identified: {analysis['type']}")
        steps.append(f"Complexity level: {analysis['complexity']}")
        
        if analysis['topics']:
            steps.append(f"Relevant topics: {', '.join(analysis['topics'])}")
        
        if analysis['requires_math']:
            steps.append("This requires mathematical reasoning...")
            steps.append("Breaking down the mathematical components...")
        
        if analysis['requires_code']:
            steps.append("This involves programming concepts...")
            steps.append("Considering best practices and patterns...")
        
        if analysis['requires_explanation']:
            steps.append("User is seeking an explanation...")
            steps.append("Structuring response for clarity...")
        
        if analysis['is_question']:
            steps.append("Formulating a comprehensive answer...")
        
        steps.append("Synthesizing response based on analysis...")
        
        return steps


class KnowledgeBase:
    """
    Built-in knowledge base with common topics and responses.
    Uses semantic similarity for matching.
    """
    def __init__(self):
        self.knowledge = self._build_knowledge()
        self.word_vectors = self._build_simple_vectors()
    
    def _build_knowledge(self) -> Dict[str, Dict]:
        """Build knowledge base entries"""
        return {
            'greeting': {
                'patterns': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good evening'],
                'responses': [
                    "Hello! 🐱 I'm Cat R1, your AI assistant. How can I help you today?",
                    "Hi there! Great to see you. What would you like to explore?",
                    "Hey! I'm ready to assist with any questions or tasks you have.",
                    "Greetings! I'm here to help. What's on your mind?",
                ]
            },
            'capabilities': {
                'patterns': ['what can you do', 'your capabilities', 'help me', 'what are you'],
                'responses': [
                    """I'm Cat R1, an AI assistant powered by a custom algorithm built entirely in Python.

**What I can help with:**
• 💡 Answering questions on various topics
• 💻 Writing & editing - essays, code, creative content
• 🔍 Analysis - breaking down complex problems
• 🔢 Math & logic - calculations and reasoning
• 🐱 Coding help - debugging, explanations, examples
• ✨ Creative tasks - stories, ideas, brainstorming

What would you like help with?""",
                ]
            },
            'coding': {
                'patterns': ['code', 'program', 'python', 'javascript', 'function', 'algorithm'],
                'responses': [
                    """I'd love to help with coding! Here's what I can do:

**Languages I can help with:**
• Python, JavaScript, TypeScript
• C, C++, Java, Go, Rust
• HTML/CSS, SQL, and more

**How I can assist:**
• Write code from descriptions
• Debug and fix errors
• Explain code concepts
• Optimize performance
• Review and improve code

What programming challenge are you working on?""",
                ]
            },
            'math': {
                'patterns': ['math', 'calculate', 'equation', 'formula', 'solve'],
                'responses': [
                    """I can help with mathematics! My capabilities include:

• **Arithmetic** - basic to complex calculations
• **Algebra** - equations, inequalities, polynomials
• **Calculus** - derivatives, integrals, limits
• **Statistics** - probability, distributions, analysis
• **Logic** - proofs, set theory, boolean algebra

Share your math problem and I'll work through it step by step!""",
                ]
            },
            'explanation': {
                'patterns': ['explain', 'what is', 'how does', 'why is', 'describe'],
                'responses': [
                    "I'll break this down clearly for you. Let me explain the key concepts step by step...",
                    "Great question! Here's a comprehensive explanation...",
                    "Let me walk you through this topic systematically...",
                ]
            },
            'thanks': {
                'patterns': ['thank', 'thanks', 'thx', 'appreciate'],
                'responses': [
                    "You're welcome! 🐱 Feel free to ask if you have more questions.",
                    "Happy to help! Let me know if there's anything else.",
                    "My pleasure! I'm here whenever you need assistance.",
                ]
            },
            'goodbye': {
                'patterns': ['bye', 'goodbye', 'see you', 'later', 'exit'],
                'responses': [
                    "Goodbye! 🐱 It was great chatting. Come back anytime!",
                    "See you later! Feel free to start a new conversation whenever.",
                    "Take care! I'll be here when you need me.",
                ]
            },
            'creative': {
                'patterns': ['story', 'poem', 'creative', 'write', 'imagine'],
                'responses': [
                    """I love creative writing! I can help with:

• **Stories** - short fiction, narratives, plot development
• **Poetry** - various styles and forms
• **Dialogue** - character conversations
• **Worldbuilding** - settings, lore, details
• **Brainstorming** - ideas and inspiration

What creative project would you like to explore?""",
                ]
            },
        }
    
    def _build_simple_vectors(self) -> Dict[str, List[float]]:
        """Build simple word vectors based on character n-grams"""
        vectors = {}
        all_words = set()
        
        for entry in self.knowledge.values():
            for pattern in entry['patterns']:
                all_words.update(pattern.lower().split())
        
        for word in all_words:
            vector = [0.0] * 26
            for char in word.lower():
                if 'a' <= char <= 'z':
                    vector[ord(char) - ord('a')] += 1
            magnitude = math.sqrt(sum(v * v for v in vector)) or 1
            vectors[word] = [v / magnitude for v in vector]
        
        return vectors
    
    def _word_similarity(self, word1: str, word2: str) -> float:
        """Calculate simple similarity between two words"""
        w1 = word1.lower()
        w2 = word2.lower()
        
        if w1 == w2:
            return 1.0
        
        if w1 in w2 or w2 in w1:
            return 0.7
        
        set1 = set(w1)
        set2 = set(w2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def find_best_match(self, query: str) -> Tuple[Optional[str], float]:
        """Find best matching knowledge entry"""
        query_words = set(query.lower().split())
        best_category = None
        best_score = 0.0
        
        for category, entry in self.knowledge.items():
            score = 0.0
            for pattern in entry['patterns']:
                pattern_words = pattern.lower().split()
                
                for qw in query_words:
                    for pw in pattern_words:
                        sim = self._word_similarity(qw, pw)
                        if sim > 0.5:
                            score += sim
            
            if score > best_score:
                best_score = score
                best_category = category
        
        return best_category, best_score
    
    def get_response(self, category: str) -> str:
        """Get a response for a category"""
        if category in self.knowledge:
            return random.choice(self.knowledge[category]['responses'])
        return ""


class CatR1AI:
    """
    Main AI engine combining all components.
    Provides streaming responses with thinking visualization.
    """
    def __init__(self, temperature: float = 0.8):
        self.temperature = temperature
        self.markov = MarkovChain(n=2)
        self.neural = NeuralTextGenerator()
        self.reasoning = ReasoningEngine()
        self.knowledge = KnowledgeBase()
        
        self._train_models()
    
    def _train_models(self):
        """Train models on built-in corpus"""
        corpus = """
        Artificial intelligence is transforming how we interact with technology. 
        Machine learning algorithms can identify patterns in data and make predictions.
        Deep learning uses neural networks with many layers to process complex information.
        Natural language processing enables computers to understand human language.
        Computer vision allows machines to interpret and analyze visual information.
        
        Programming is the process of creating instructions for computers to follow.
        Python is a versatile language known for its readability and extensive libraries.
        JavaScript powers interactive web applications and runs in browsers.
        Algorithms are step-by-step procedures for solving problems efficiently.
        Data structures organize and store data for efficient access and modification.
        
        Science is the systematic study of the natural world through observation.
        Physics explores the fundamental laws governing matter and energy.
        Chemistry investigates the composition and properties of substances.
        Biology studies living organisms and their interactions with environments.
        Mathematics provides the language and tools for scientific discovery.
        
        The universe is vast and contains billions of galaxies each with billions of stars.
        Our solar system orbits in the Milky Way galaxy about 26000 light years from the center.
        Earth is the only known planet to harbor life with diverse ecosystems.
        Climate change is affecting weather patterns and ecosystems worldwide.
        Renewable energy sources like solar and wind are becoming more prevalent.
        
        Philosophy examines fundamental questions about existence knowledge and ethics.
        Critical thinking involves analyzing information objectively and making reasoned judgments.
        Logic is the study of valid reasoning and argumentation.
        Ethics explores concepts of right and wrong in human behavior.
        Epistemology investigates the nature and scope of knowledge.
        
        Communication is essential for sharing ideas and building relationships.
        Writing clearly helps convey complex information effectively.
        Reading expands vocabulary and improves comprehension skills.
        Listening actively demonstrates respect and improves understanding.
        Public speaking requires practice and confidence to deliver messages effectively.
        
        I understand your question and will provide a helpful response.
        Let me think about this carefully to give you accurate information.
        This is an interesting topic that deserves thorough exploration.
        I can help you understand this concept better with examples.
        Here is my analysis based on the information provided.
        """
        
        self.markov.train(corpus)
        self.neural.train(corpus)
    
    def generate_response(self, 
                          query: str, 
                          history: List[Dict] = None,
                          show_thinking: bool = True) -> Generator[Tuple[str, str], None, None]:
        """
        Generate streaming response with optional thinking visualization.
        Yields tuples of (type, content) where type is 'thinking' or 'response'.
        """
        analysis = self.reasoning.analyze_query(query)
        
        if show_thinking:
            thinking_steps = self.reasoning.generate_thinking(query, analysis)
            for step in thinking_steps:
                yield ('thinking', step)
                time.sleep(0.1)
        
        category, score = self.knowledge.find_best_match(query)
        
        if score > 1.0 and category:
            response = self.knowledge.get_response(category)
        else:
            response = self._generate_custom_response(query, analysis)
        
        for i, char in enumerate(response):
            yield ('response', char)
            if char in '.!?\n':
                time.sleep(0.03)
            elif char == ' ':
                time.sleep(0.015)
            else:
                time.sleep(0.008)
    
    def _generate_custom_response(self, query: str, analysis: Dict) -> str:
        """Generate custom response using neural and Markov models"""
        intro_phrases = [
            "That's an interesting question! ",
            "Let me help you with that. ",
            "I'd be happy to assist. ",
            "Great question! ",
            "Here's what I think: ",
        ]
        
        response_parts = [random.choice(intro_phrases)]
        
        if analysis['type'] == 'math':
            response_parts.append(self._handle_math(query))
        elif analysis['type'] == 'coding':
            response_parts.append(self._handle_coding(query))
        elif analysis['requires_explanation']:
            response_parts.append(self._handle_explanation(query, analysis))
        else:
            markov_text = self.markov.generate(max_words=30)
            neural_text = self.neural.generate(query, max_tokens=20, temperature=self.temperature)
            
            if markov_text:
                response_parts.append(markov_text)
            if neural_text:
                response_parts.append(neural_text)
        
        response_parts.append("\n\nIs there anything specific you'd like me to elaborate on?")
        
        return ''.join(response_parts)
    
    def _handle_math(self, query: str) -> str:
        """Handle math-related queries"""
        numbers = re.findall(r'-?\d+\.?\d*', query)
        
        if len(numbers) >= 2:
            nums = [float(n) for n in numbers[:2]]
            results = []
            
            if '+' in query or 'add' in query.lower() or 'sum' in query.lower():
                results.append(f"{nums[0]} + {nums[1]} = {nums[0] + nums[1]}")
            if '-' in query or 'subtract' in query.lower() or 'minus' in query.lower():
                results.append(f"{nums[0]} - {nums[1]} = {nums[0] - nums[1]}")
            if '*' in query or 'x' in query or 'multiply' in query.lower() or 'times' in query.lower():
                results.append(f"{nums[0]} × {nums[1]} = {nums[0] * nums[1]}")
            if '/' in query or 'divide' in query.lower():
                if nums[1] != 0:
                    results.append(f"{nums[0]} ÷ {nums[1]} = {nums[0] / nums[1]:.4f}")
            
            if results:
                return "Here's the calculation:\n\n" + "\n".join(results)
        
        return """I can help with mathematics! Please provide:
• Numbers to calculate
• The operation you need (add, subtract, multiply, divide)
• Or describe the math problem in detail

Example: "What is 25 + 17?" or "Calculate 144 divided by 12" """
    
    def _handle_coding(self, query: str) -> str:
        """Handle coding-related queries"""
        query_lower = query.lower()
        
        if 'python' in query_lower:
            if 'hello' in query_lower or 'print' in query_lower:
                return '''Here's a simple Python example:

```python
# Hello World in Python
print("Hello, World!")

# With a function
def greet(name):
    return f"Hello, {name}!"

print(greet("User"))
```

Python is known for its clean, readable syntax!'''
            
            if 'loop' in query_lower or 'for' in query_lower:
                return '''Here's how loops work in Python:

```python
# For loop
for i in range(5):
    print(f"Iteration {i}")

# While loop
count = 0
while count < 5:
    print(f"Count: {count}")
    count += 1

# Loop through a list
items = ["apple", "banana", "cherry"]
for item in items:
    print(item)
```'''
        
        return """I can help with programming! Tell me:
• What language you're using
• What you're trying to accomplish
• Any errors you're encountering

I'll provide code examples and explanations!"""
    
    def _handle_explanation(self, query: str, analysis: Dict) -> str:
        """Handle explanation requests"""
        topics = analysis.get('topics', [])
        
        if 'science' in topics:
            return """Science is the systematic pursuit of knowledge about the natural world through observation, experimentation, and evidence-based reasoning.

**Key aspects:**
• **Observation** - Carefully watching and recording phenomena
• **Hypothesis** - Forming testable explanations
• **Experimentation** - Testing hypotheses through controlled tests
• **Analysis** - Examining data to draw conclusions
• **Peer Review** - Having findings verified by other scientists

What specific scientific topic interests you?"""
        
        if 'technology' in topics:
            return """Technology encompasses the tools, systems, and methods we use to solve problems and extend human capabilities.

**Modern tech includes:**
• **Computing** - Hardware, software, algorithms
• **Internet** - Global connectivity and communication
• **AI** - Machines that can learn and reason
• **Mobile** - Smartphones and portable devices
• **Cloud** - Remote computing and storage

What technology topic would you like to explore?"""
        
        return """I'll do my best to explain clearly!

To give you the most helpful explanation, could you tell me:
• What specific concept or topic you want explained?
• Your current level of familiarity with it?
• Any particular aspect that confuses you?

I'll tailor my explanation to your needs!"""


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Message:
    role: str
    content: str
    thinking: str = ""
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])


@dataclass  
class ChatThread:
    title: str
    messages: List[Message] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])
    
    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "id": self.id,
            "created": self.created,
            "messages": [
                {"role": m.role, "content": m.content, "thinking": m.thinking, "timestamp": m.timestamp}
                for m in self.messages
            ]
        }


class ChatController:
    """Manages chat threads and AI interactions"""
    def __init__(self):
        self.threads: List[ChatThread] = []
        self.current_index: int = -1
        self.ai = CatR1AI(temperature=0.8)
        self.temperature = 0.8
        self.show_thinking = True
        
        self.new_thread()
    
    @property
    def current(self) -> Optional[ChatThread]:
        if 0 <= self.current_index < len(self.threads):
            return self.threads[self.current_index]
        return None
    
    def new_thread(self, title: str = "New Chat") -> ChatThread:
        thread = ChatThread(title=title)
        self.threads.insert(0, thread)
        self.current_index = 0
        return thread
    
    def delete_thread(self, index: int):
        if 0 <= index < len(self.threads):
            del self.threads[index]
            if self.current_index >= len(self.threads):
                self.current_index = max(0, len(self.threads) - 1)
            if not self.threads:
                self.new_thread()
    
    def set_temperature(self, temp: float):
        self.temperature = max(0.1, min(1.5, temp))
        self.ai.temperature = self.temperature


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION GUI
# ═══════════════════════════════════════════════════════════════════════════════

class CatR1ChatApp(tk.Tk):
    """Main application window - Cat R1 Chat interface"""
    
    def __init__(self):
        super().__init__()
        
        self.title(f"{APP_NAME} v{VERSION}")
        self.geometry("1200x800")
        self.minsize(900, 600)
        
        self.theme = tk.StringVar(value="dark")
        self.themes = {
            "dark": {
                "bg": "#0a0a0a",
                "bg2": "#0f0f0f",
                "sidebar": "#050505",
                "panel": "#0a0a0a",
                "text": "#00ff00",
                "muted": "#00aa00",
                "accent": "#00ff00",
                "accent2": "#00cc00",
                "user_bubble": "#0a1a0a",
                "ai_bubble": "#001a00",
                "thinking": "#0a150a",
                "border": "#003300",
                "success": "#00ff00",
                "warning": "#00ff00",
                "error": "#ff0000",
                "button_bg": "#003300",
                "button_fg": "#00ff00",
                "button_hover": "#004400",
                "button_active": "#00ff00",
            },
            "light": {
                "bg": "#f0fff0",
                "bg2": "#ffffff",
                "sidebar": "#e0ffe0",
                "panel": "#ffffff",
                "text": "#003300",
                "muted": "#006600",
                "accent": "#00aa00",
                "accent2": "#008800",
                "user_bubble": "#d0ffd0",
                "ai_bubble": "#e8ffe8",
                "thinking": "#f0fff0",
                "border": "#00aa00",
                "success": "#00aa00",
                "warning": "#00aa00",
                "error": "#cc0000",
                "button_bg": "#00aa00",
                "button_fg": "#ffffff",
                "button_hover": "#00cc00",
                "button_active": "#008800",
            }
        }
        
        self.controller = ChatController()
        self._streaming = False
        self._stop_flag = False
        self._stream_thread = None
        
        self._build_ui()
        self._apply_theme()
        
        self._add_welcome_message()
    
    def _p(self) -> Dict:
        """Get current theme palette"""
        return self.themes[self.theme.get()]
    
    def _build_ui(self):
        """Build the main UI"""
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        self._build_sidebar()
        self._build_main_area()
        self._build_input_area()
    
    def _build_sidebar(self):
        """Build the sidebar with chat list"""
        pal = self._p()
        
        self.sidebar = tk.Frame(self, width=280)
        self.sidebar.grid(row=0, column=0, sticky="nsew", rowspan=2)
        self.sidebar.grid_propagate(False)
        
        header = tk.Frame(self.sidebar)
        header.pack(fill="x", padx=12, pady=12)
        
        self.logo_label = tk.Label(
            header,
            text="🐱 Cat R1",
            font=("Helvetica", 18, "bold"),
        )
        self.logo_label.pack(side="left")
        
        self.new_chat_btn = tk.Button(
            header,
            text="➕ New",
            font=("Helvetica", 10),
            relief="flat",
            cursor="hand2",
            command=self._new_chat
        )
        self.new_chat_btn.pack(side="right")
        
        search_frame = tk.Frame(self.sidebar)
        search_frame.pack(fill="x", padx=12, pady=(0, 12))
        
        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(
            search_frame,
            textvariable=self.search_var,
            font=("Helvetica", 11),
            relief="flat",
        )
        self.search_entry.pack(fill="x", ipady=8, ipadx=8)
        self.search_entry.insert(0, "🔍 Search chats...")
        self.search_entry.bind("<FocusIn>", lambda e: self._clear_placeholder())
        self.search_entry.bind("<FocusOut>", lambda e: self._restore_placeholder())
        self.search_entry.bind("<KeyRelease>", lambda e: self._filter_chats())
        
        list_frame = tk.Frame(self.sidebar)
        list_frame.pack(fill="both", expand=True, padx=12)
        
        self.chat_listbox = tk.Listbox(
            list_frame,
            font=("Helvetica", 11),
            relief="flat",
            highlightthickness=0,
            selectmode="single",
            activestyle="none",
        )
        self.chat_listbox.pack(fill="both", expand=True)
        self.chat_listbox.bind("<<ListboxSelect>>", self._on_chat_select)
        self.chat_listbox.bind("<Button-3>", self._show_context_menu)
        
        bottom_frame = tk.Frame(self.sidebar)
        bottom_frame.pack(fill="x", padx=12, pady=12)
        
        self.theme_btn = tk.Button(
            bottom_frame,
            text="🌙 Dark",
            font=("Helvetica", 10),
            relief="flat",
            cursor="hand2",
            command=self._toggle_theme
        )
        self.theme_btn.pack(side="left")
        
        self.settings_btn = tk.Button(
            bottom_frame,
            text="⚙️ Settings",
            font=("Helvetica", 10),
            relief="flat",
            cursor="hand2",
            command=self._open_settings
        )
        self.settings_btn.pack(side="right")
        
        self._refresh_chat_list()
    
    def _build_main_area(self):
        """Build the main chat area"""
        self.main_frame = tk.Frame(self)
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        header = tk.Frame(self.main_frame, height=50)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        self.chat_title_label = tk.Label(
            header,
            text="New Chat",
            font=("Helvetica", 14, "bold"),
        )
        self.chat_title_label.pack(side="left", padx=16, pady=12)
        
        self.export_btn = tk.Button(
            header,
            text="📥 Export",
            font=("Helvetica", 10),
            relief="flat",
            cursor="hand2",
            command=self._export_chat
        )
        self.export_btn.pack(side="right", padx=8, pady=12)
        
        self.canvas = tk.Canvas(self.main_frame, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.messages_frame = tk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.messages_frame, anchor="nw")
        
        self.messages_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)
    
    def _build_input_area(self):
        """Build the input area"""
        self.input_frame = tk.Frame(self)
        self.input_frame.grid(row=1, column=1, sticky="ew", padx=16, pady=16)
        self.input_frame.grid_columnconfigure(0, weight=1)
        
        input_container = tk.Frame(self.input_frame)
        input_container.pack(fill="x")
        input_container.grid_columnconfigure(0, weight=1)
        
        self.input_text = tk.Text(
            input_container,
            height=3,
            font=("Helvetica", 12),
            wrap="word",
            relief="flat",
            padx=12,
            pady=10,
        )
        self.input_text.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.input_text.bind("<Return>", self._on_enter)
        self.input_text.bind("<Shift-Return>", lambda e: None)
        
        btn_frame = tk.Frame(input_container)
        btn_frame.grid(row=0, column=1)
        
        self.send_btn = tk.Button(
            btn_frame,
            text="Send ➤",
            font=("Helvetica", 11, "bold"),
            relief="flat",
            cursor="hand2",
            width=10,
            command=self._send_message
        )
        self.send_btn.pack(pady=(0, 4))
        
        self.stop_btn = tk.Button(
            btn_frame,
            text="⏹ Stop",
            font=("Helvetica", 10),
            relief="flat",
            cursor="hand2",
            width=10,
            command=self._stop_generation
        )
        self.stop_btn.pack()
        
        hint_label = tk.Label(
            self.input_frame,
            text="Press Enter to send, Shift+Enter for new line",
            font=("Helvetica", 9),
        )
        hint_label.pack(anchor="w", pady=(4, 0))
    
    def _apply_theme(self):
        """Apply current theme colors"""
        pal = self._p()
        
        self.configure(bg=pal["bg"])
        
        self.sidebar.configure(bg=pal["sidebar"])
        for widget in self.sidebar.winfo_children():
            self._apply_theme_recursive(widget, pal, "sidebar")
        
        self.logo_label.configure(bg=pal["sidebar"], fg=pal["accent"])
        self.new_chat_btn.configure(
            bg=pal["button_bg"], 
            fg=pal["button_fg"], 
            activebackground=pal["button_active"],
            activeforeground="#000000"
        )
        
        self.search_entry.configure(bg=pal["bg2"], fg=pal["text"], insertbackground=pal["text"])
        
        self.chat_listbox.configure(
            bg=pal["sidebar"],
            fg=pal["text"],
            selectbackground=pal["button_bg"],
            selectforeground=pal["button_fg"]
        )
        
        self.theme_btn.configure(
            bg=pal["button_bg"], 
            fg=pal["button_fg"], 
            activebackground=pal["button_active"],
            activeforeground="#000000"
        )
        self.settings_btn.configure(
            bg=pal["button_bg"], 
            fg=pal["button_fg"], 
            activebackground=pal["button_active"],
            activeforeground="#000000"
        )
        self.theme_btn.configure(text="☀️ Light" if self.theme.get() == "dark" else "🌙 Dark")
        
        self.main_frame.configure(bg=pal["bg"])
        self.canvas.configure(bg=pal["bg"])
        self.messages_frame.configure(bg=pal["bg"])
        
        self.chat_title_label.configure(bg=pal["bg"], fg=pal["text"])
        self.export_btn.configure(
            bg=pal["button_bg"], 
            fg=pal["button_fg"], 
            activebackground=pal["button_active"],
            activeforeground="#000000"
        )
        
        self.input_frame.configure(bg=pal["bg"])
        self.input_text.configure(
            bg=pal["bg2"],
            fg=pal["text"],
            insertbackground=pal["text"],
            highlightbackground=pal["border"],
            highlightcolor=pal["accent"],
            highlightthickness=2
        )
        
        self.send_btn.configure(
            bg=pal["button_bg"], 
            fg=pal["button_fg"], 
            activebackground=pal["button_active"],
            activeforeground="#000000"
        )
        self.stop_btn.configure(
            bg="#330000", 
            fg="#ff0000", 
            activebackground="#ff0000",
            activeforeground="#000000"
        )
        
        for child in self.messages_frame.winfo_children():
            self._style_message_widget(child)
    
    def _apply_theme_recursive(self, widget, pal, area="bg"):
        """Recursively apply theme to widgets"""
        bg_color = pal["sidebar"] if area == "sidebar" else pal["bg"]
        
        try:
            widget.configure(bg=bg_color)
        except:
            pass
        
        for child in widget.winfo_children():
            self._apply_theme_recursive(child, pal, area)
    
    def _style_message_widget(self, widget):
        """Style a message widget for current theme"""
        pal = self._p()
        
        try:
            role = getattr(widget, '_role', 'user')
            bubble_bg = pal["user_bubble"] if role == "user" else pal["ai_bubble"]
            
            widget.configure(bg=pal["bg"])
            
            for child in widget.winfo_children():
                try:
                    if hasattr(child, '_is_bubble') and child._is_bubble:
                        child.configure(bg=bubble_bg)
                        for label in child.winfo_children():
                            if isinstance(label, tk.Label):
                                label.configure(bg=bubble_bg, fg=pal["text"])
                    elif hasattr(child, '_is_thinking') and child._is_thinking:
                        child.configure(bg=pal["thinking"])
                        for label in child.winfo_children():
                            if isinstance(label, tk.Label):
                                label.configure(bg=pal["thinking"], fg=pal["muted"])
                    else:
                        child.configure(bg=pal["bg"])
                except:
                    pass
        except:
            pass
    
    def _toggle_theme(self):
        """Toggle between dark and light themes"""
        self.theme.set("light" if self.theme.get() == "dark" else "dark")
        self._apply_theme()
    
    def _clear_placeholder(self):
        """Clear search placeholder"""
        if self.search_var.get() == "🔍 Search chats...":
            self.search_entry.delete(0, tk.END)
    
    def _restore_placeholder(self):
        """Restore search placeholder"""
        if not self.search_var.get():
            self.search_entry.insert(0, "🔍 Search chats...")
    
    def _filter_chats(self):
        """Filter chat list based on search"""
        query = self.search_var.get().lower()
        if query == "🔍 search chats...":
            query = ""
        self._refresh_chat_list(filter_text=query)
    
    def _refresh_chat_list(self, filter_text: str = ""):
        """Refresh the chat list"""
        self.chat_listbox.delete(0, tk.END)
        
        for i, thread in enumerate(self.controller.threads):
            title = thread.title[:35] + "..." if len(thread.title) > 35 else thread.title
            
            if filter_text and filter_text not in title.lower():
                continue
            
            self.chat_listbox.insert(tk.END, f"💬 {title}")
        
        if self.controller.current_index >= 0 and not filter_text:
            self.chat_listbox.selection_set(self.controller.current_index)
    
    def _on_chat_select(self, event):
        """Handle chat selection"""
        selection = self.chat_listbox.curselection()
        if not selection:
            return
        
        self.controller.current_index = selection[0]
        self._render_current_chat()
    
    def _show_context_menu(self, event):
        """Show right-click context menu"""
        try:
            index = self.chat_listbox.nearest(event.y)
            self.chat_listbox.selection_clear(0, tk.END)
            self.chat_listbox.selection_set(index)
            
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(label="🗑️ Delete Chat", command=lambda: self._delete_chat(index))
            menu.add_command(label="📝 Rename Chat", command=lambda: self._rename_chat(index))
            menu.add_separator()
            menu.add_command(label="📥 Export Chat", command=self._export_chat)
            
            menu.tk_popup(event.x_root, event.y_root)
        except:
            pass
    
    def _new_chat(self):
        """Create new chat"""
        self.controller.new_thread()
        self._refresh_chat_list()
        self._clear_messages()
        self._add_welcome_message()
        self._update_title()
    
    def _delete_chat(self, index: int):
        """Delete a chat"""
        if messagebox.askyesno("Delete Chat", "Are you sure you want to delete this chat?"):
            self.controller.delete_thread(index)
            self._refresh_chat_list()
            self._render_current_chat()
    
    def _rename_chat(self, index: int):
        """Rename a chat"""
        if 0 <= index < len(self.controller.threads):
            thread = self.controller.threads[index]
            
            dialog = tk.Toplevel(self)
            dialog.title("Rename Chat")
            dialog.geometry("400x120")
            dialog.transient(self)
            dialog.grab_set()
            
            pal = self._p()
            dialog.configure(bg=pal["bg"])
            
            tk.Label(dialog, text="New title:", font=("Helvetica", 11), bg=pal["bg"], fg=pal["text"]).pack(pady=(16, 8))
            
            entry = tk.Entry(dialog, font=("Helvetica", 11), width=40, bg=pal["bg2"], fg=pal["text"])
            entry.pack(pady=4)
            entry.insert(0, thread.title)
            entry.select_range(0, tk.END)
            entry.focus()
            
            def save():
                thread.title = entry.get() or "Untitled"
                self._refresh_chat_list()
                self._update_title()
                dialog.destroy()
            
            tk.Button(
                dialog, 
                text="Save", 
                command=save, 
                bg=pal["button_bg"], 
                fg=pal["button_fg"],
                activebackground=pal["button_active"],
                activeforeground="#000000"
            ).pack(pady=12)
            
            entry.bind("<Return>", lambda e: save())
    
    def _render_current_chat(self):
        """Render current chat messages"""
        self._clear_messages()
        
        thread = self.controller.current
        if not thread:
            return
        
        self._update_title()
        
        for msg in thread.messages:
            self._add_message_widget(msg.role, msg.content, msg.thinking, animate=False)
        
        self._scroll_to_bottom()
    
    def _clear_messages(self):
        """Clear all messages from display"""
        for child in self.messages_frame.winfo_children():
            child.destroy()
    
    def _add_welcome_message(self):
        """Add welcome message"""
        welcome = """🐱 **Welcome to Cat R1!**

I'm an AI assistant powered by a custom algorithm built entirely in Python.

**What I can help with:**
• 💡 Answering questions and explaining concepts
• 💻 Programming and coding assistance  
• ✍️ Writing and creative tasks
• 🔢 Math and logical reasoning
• 🎯 Analysis and problem-solving

**Try asking:**
• "Explain how neural networks work"
• "Write a Python function to sort a list"
• "What is the meaning of life?"

Type your message below to get started!"""
        
        self._add_message_widget("assistant", welcome, "", animate=False)
    
    def _add_message_widget(self, role: str, content: str, thinking: str = "", animate: bool = True):
        """Add a message widget to the display"""
        pal = self._p()
        
        container = tk.Frame(self.messages_frame, bg=pal["bg"])
        container._role = role
        container.pack(fill="x", padx=16, pady=8)
        
        role_label = tk.Label(
            container,
            text="🧑 You" if role == "user" else "🐱 Cat R1",
            font=("Helvetica", 10, "bold"),
            bg=pal["bg"],
            fg=pal["muted"]
        )
        role_label.pack(anchor="w", padx=4)
        
        if thinking and role == "assistant":
            thinking_frame = tk.Frame(container, bg=pal["thinking"], padx=12, pady=8)
            thinking_frame._is_thinking = True
            thinking_frame.pack(fill="x", pady=(4, 4), padx=4)
            
            thinking_label = tk.Label(
                thinking_frame,
                text="💭 " + thinking,
                font=("Helvetica", 10, "italic"),
                bg=pal["thinking"],
                fg=pal["muted"],
                wraplength=700,
                justify="left",
                anchor="w"
            )
            thinking_label.pack(anchor="w")
        
        bubble_bg = pal["user_bubble"] if role == "user" else pal["ai_bubble"]
        
        bubble = tk.Frame(container, bg=bubble_bg, padx=16, pady=12)
        bubble._is_bubble = True
        bubble.pack(fill="x", pady=4, padx=4)
        
        content_label = tk.Label(
            bubble,
            text=content,
            font=("Helvetica", 11),
            bg=bubble_bg,
            fg=pal["text"],
            wraplength=700,
            justify="left",
            anchor="w"
        )
        content_label.pack(anchor="w", fill="x")
        
        if role == "assistant":
            btn_frame = tk.Frame(container, bg=pal["bg"])
            btn_frame.pack(anchor="w", padx=4, pady=(0, 4))
            
            copy_btn = tk.Button(
                btn_frame,
                text="📋 Copy",
                font=("Helvetica", 9),
                bg=pal["button_bg"],
                fg=pal["button_fg"],
                relief="flat",
                cursor="hand2",
                activebackground=pal["button_active"],
                activeforeground="#000000",
                command=lambda c=content: self._copy_to_clipboard(c)
            )
            copy_btn.pack(side="left", padx=(0, 8))
        
        self._scroll_to_bottom()
    
    def _copy_to_clipboard(self, text: str):
        """Copy text to clipboard"""
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()
    
    def _update_title(self):
        """Update chat title display"""
        thread = self.controller.current
        if thread:
            self.chat_title_label.configure(text=thread.title)
    
    def _on_frame_configure(self, event):
        """Handle frame resize"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _on_canvas_configure(self, event):
        """Handle canvas resize"""
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
        else:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def _scroll_to_bottom(self):
        """Scroll to bottom of messages"""
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)
    
    def _on_enter(self, event):
        """Handle Enter key press"""
        if not (event.state & 0x1):
            self._send_message()
            return "break"
    
    def _send_message(self):
        """Send user message"""
        if self._streaming:
            return
        
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            return
        
        self.input_text.delete("1.0", tk.END)
        
        thread = self.controller.current
        if thread and not thread.messages:
            thread.title = text[:40] + ("..." if len(text) > 40 else "")
            self._refresh_chat_list()
            self._update_title()
        
        user_msg = Message(role="user", content=text)
        if thread:
            thread.messages.append(user_msg)
        
        self._add_message_widget("user", text, animate=False)
        
        self._generate_response(text)
    
    def _generate_response(self, query: str):
        """Generate AI response"""
        self._streaming = True
        self._stop_flag = False
        
        pal = self._p()
        
        container = tk.Frame(self.messages_frame, bg=pal["bg"])
        container._role = "assistant"
        container.pack(fill="x", padx=16, pady=8)
        
        role_label = tk.Label(
            container,
            text="🐱 Cat R1",
            font=("Helvetica", 10, "bold"),
            bg=pal["bg"],
            fg=pal["muted"]
        )
        role_label.pack(anchor="w", padx=4)
        
        thinking_frame = tk.Frame(container, bg=pal["thinking"], padx=12, pady=8)
        thinking_frame._is_thinking = True
        thinking_frame.pack(fill="x", pady=(4, 4), padx=4)
        
        thinking_var = tk.StringVar(value="💭 Thinking...")
        thinking_label = tk.Label(
            thinking_frame,
            textvariable=thinking_var,
            font=("Helvetica", 10, "italic"),
            bg=pal["thinking"],
            fg=pal["muted"],
            wraplength=700,
            justify="left",
            anchor="w"
        )
        thinking_label.pack(anchor="w")
        
        bubble = tk.Frame(container, bg=pal["ai_bubble"], padx=16, pady=12)
        bubble._is_bubble = True
        bubble.pack(fill="x", pady=4, padx=4)
        
        content_var = tk.StringVar(value="")
        content_label = tk.Label(
            bubble,
            textvariable=content_var,
            font=("Helvetica", 11),
            bg=pal["ai_bubble"],
            fg=pal["text"],
            wraplength=700,
            justify="left",
            anchor="w"
        )
        content_label.pack(anchor="w", fill="x")
        
        def stream_response():
            thinking_parts = []
            response_parts = []
            
            try:
                history = []
                thread = self.controller.current
                if thread:
                    for msg in thread.messages[-6:]:
                        history.append({"role": msg.role, "content": msg.content})
                
                generator = self.controller.ai.generate_response(
                    query,
                    history=history,
                    show_thinking=self.controller.show_thinking
                )
                
                for msg_type, content in generator:
                    if self._stop_flag:
                        break
                    
                    if msg_type == 'thinking':
                        thinking_parts.append(content)
                        self.after(0, lambda t="💭 " + "\n".join(thinking_parts): thinking_var.set(t))
                    else:
                        response_parts.append(content)
                        self.after(0, lambda r="".join(response_parts): content_var.set(r))
                    
                    self.after(0, self._scroll_to_bottom)
            
            except Exception as e:
                self.after(0, lambda: content_var.set(f"Error: {str(e)}"))
            
            finally:
                final_thinking = "\n".join(thinking_parts)
                final_response = "".join(response_parts)
                
                if thread and final_response:
                    ai_msg = Message(role="assistant", content=final_response, thinking=final_thinking)
                    thread.messages.append(ai_msg)
                
                def add_copy_button():
                    btn_frame = tk.Frame(container, bg=pal["bg"])
                    btn_frame.pack(anchor="w", padx=4, pady=(0, 4))
                    
                    copy_btn = tk.Button(
                        btn_frame,
                        text="📋 Copy",
                        font=("Helvetica", 9),
                        bg=pal["button_bg"],
                        fg=pal["button_fg"],
                        relief="flat",
                        cursor="hand2",
                        activebackground=pal["button_active"],
                        activeforeground="#000000",
                        command=lambda: self._copy_to_clipboard(final_response)
                    )
                    copy_btn.pack(side="left")
                
                self.after(0, add_copy_button)
                self._streaming = False
        
        self._stream_thread = threading.Thread(target=stream_response, daemon=True)
        self._stream_thread.start()
    
    def _stop_generation(self):
        """Stop AI generation"""
        self._stop_flag = True
    
    def _export_chat(self):
        """Export current chat to JSON"""
        thread = self.controller.current
        if not thread:
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"cat_r1_chat_{int(time.time())}.json"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(thread.to_dict(), f, indent=2, ensure_ascii=False)
                messagebox.showinfo("Export", f"Chat exported to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")
    
    def _open_settings(self):
        """Open settings dialog"""
        pal = self._p()
        
        dialog = tk.Toplevel(self)
        dialog.title("Settings")
        dialog.geometry("450x350")
        dialog.transient(self)
        dialog.grab_set()
        dialog.configure(bg=pal["bg"])
        
        tk.Label(
            dialog,
            text="⚙️ Settings",
            font=("Helvetica", 16, "bold"),
            bg=pal["bg"],
            fg=pal["text"]
        ).pack(pady=16)
        
        settings_frame = tk.Frame(dialog, bg=pal["bg"])
        settings_frame.pack(fill="x", padx=24)
        
        tk.Label(
            settings_frame,
            text="Temperature (creativity):",
            font=("Helvetica", 11),
            bg=pal["bg"],
            fg=pal["text"]
        ).pack(anchor="w", pady=(8, 4))
        
        temp_var = tk.DoubleVar(value=self.controller.temperature)
        temp_scale = tk.Scale(
            settings_frame,
            from_=0.1,
            to=1.5,
            resolution=0.1,
            orient="horizontal",
            variable=temp_var,
            bg=pal["bg"],
            fg=pal["text"],
            highlightthickness=0,
            length=300,
            troughcolor=pal["button_bg"],
            activebackground=pal["button_active"]
        )
        temp_scale.pack(anchor="w")
        
        thinking_var = tk.BooleanVar(value=self.controller.show_thinking)
        thinking_check = tk.Checkbutton(
            settings_frame,
            text="Show thinking process",
            variable=thinking_var,
            font=("Helvetica", 11),
            bg=pal["bg"],
            fg=pal["text"],
            selectcolor=pal["bg2"],
            activebackground=pal["bg"],
            activeforeground=pal["text"]
        )
        thinking_check.pack(anchor="w", pady=16)
        
        def save_settings():
            self.controller.set_temperature(temp_var.get())
            self.controller.show_thinking = thinking_var.get()
            dialog.destroy()
        
        btn_frame = tk.Frame(dialog, bg=pal["bg"])
        btn_frame.pack(pady=24)
        
        tk.Button(
            btn_frame,
            text="Save",
            font=("Helvetica", 11),
            bg=pal["button_bg"],
            fg=pal["button_fg"],
            relief="flat",
            padx=24,
            pady=8,
            cursor="hand2",
            activebackground=pal["button_active"],
            activeforeground="#000000",
            command=save_settings
        ).pack(side="left", padx=8)
        
        tk.Button(
            btn_frame,
            text="Cancel",
            font=("Helvetica", 11),
            bg=pal["bg2"],
            fg=pal["text"],
            relief="flat",
            padx=24,
            pady=8,
            cursor="hand2",
            command=dialog.destroy
        ).pack(side="left", padx=8)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                 CAT R1 - Pure Python Edition                 ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Custom AI Algorithm featuring:                              ║")
    print("║  • Markov Chain text generation                              ║")
    print("║  • Neural-inspired co-occurrence model                       ║")
    print("║  • Chain-of-thought reasoning engine                         ║")
    print("║  • Pattern matching knowledge base                           ║")
    print("║                                                              ║")
    print("║  100% Python/Tkinter - No external ML dependencies!          ║")
    print("║  (C) 2025 Samsoft / Flames Co.                               ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    app = CatR1ChatApp()
    app.mainloop()


if __name__ == "__main__":
    main()
