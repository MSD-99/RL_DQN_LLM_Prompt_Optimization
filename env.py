# env.py -- environment for the DQN prompt optimization project
# wraps a frozen TinyLlama and exposes a gym-like interface

import json
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer


# action
KEEP        = 0
ADD_COT     = 1
ADD_CONCISE = 2
ADD_CHECK   = 3
REMOVE_LAST = 4
RESET_ALL   = 5
NUM_ACTIONS = 6

# Actual text we append for each modifier
MODIFIER_TEXTS = {
    'COT':     'Think step by step.',
    'CONCISE': 'Be concise, max 2 sentences.',
    'CHECK':   'Double-check before final.',
}
MODIFIER_KEYS = list(MODIFIER_TEXTS.keys())
NUM_MODIFIERS = len(MODIFIER_KEYS)


class BoolQEnv:
    def __init__(self, samples, tokenizer, model, embedder,
                 device='cuda:0', max_steps=2):
        self.samples   = samples
        self.tokenizer = tokenizer
        self.model     = model
        self.embedder  = embedder
        self.device    = device
        self.max_steps = max_steps

        self.questions  = [s['question'] for s in samples]
        self.passages   = [s['passage']  for s in samples]
        self.references = [s['answer']   for s in samples]

        # Pre-compute question embeddings (N x embedding_dim)
        print(f"[env] Computing embeddings for {len(self.questions)} questions...")
        self.q_embeddings = embedder.encode(
            self.questions,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=64,
        ).astype(np.float32)

        self.embedding_dim = self.q_embeddings.shape[1]
        self.state_dim = self.embedding_dim + NUM_MODIFIERS + 2  # emb + onehot + sim + tok
        self.action_dim = NUM_ACTIONS
        print(f"[env] State dim = {self.embedding_dim} (emb) + "
              f"{NUM_MODIFIERS} (modifiers) + 2 (sim, tok) = {self.state_dim}")

        # Episode variables (reset each episode)
        self.current_idx = 0
        self.step_count = 0
        self.active_modifiers = []
        self.last_similarity = 0.0
        self.last_tokens = 0.0

    def _get_state(self):
        """Build state: [question_emb, modifier_onehot, similarity, tokens]."""
        q_emb = self.q_embeddings[self.current_idx]

        # One-hot for active modifiers
        mod_onehot = np.zeros(NUM_MODIFIERS, dtype=np.float32)
        for m in self.active_modifiers:
            mod_onehot[MODIFIER_KEYS.index(m)] = 1.0

        extra = np.array([
            self.last_similarity,
            self.last_tokens / 120.0,  # normalize
        ], dtype=np.float32)

        return np.concatenate([q_emb, mod_onehot, extra])

    def reset(self, idx=None):
        """Start a new episode. If idx is None, pick a random question."""
        if idx is not None:
            self.current_idx = idx
        else:
            self.current_idx = np.random.randint(len(self.samples))

        self.step_count = 0
        self.active_modifiers = []
        self.last_similarity = 0.0
        self.last_tokens = 0.0
        return self._get_state()

    def step(self, action):
        """apply action -> generate -> reward. returns (s', r, done, info)"""
        self._apply_action(action)

        prompt = self._build_prompt()
        answer, num_tokens = self._generate(prompt)

        reference = self.references[self.current_idx]
        similarity = self._compute_similarity(answer, reference)
        reward = self._compute_reward(similarity, num_tokens)

        self.last_similarity = similarity
        self.last_tokens = float(num_tokens)
        self.step_count += 1
        done = (self.step_count >= self.max_steps)

        info = {
            'answer':     answer,
            'reference':  reference,
            'similarity': similarity,
            'num_tokens': num_tokens,
            'modifiers':  list(self.active_modifiers),
            'reward':     reward,
            'action':     action,
        }
        return self._get_state(), reward, done, info

    def _apply_action(self, action):
        """update active_modifiers based on action"""
        if action == KEEP:
            pass
        elif action == ADD_COT:
            if 'COT' not in self.active_modifiers:
                self.active_modifiers.append('COT')
        elif action == ADD_CONCISE:
            if 'CONCISE' not in self.active_modifiers:
                self.active_modifiers.append('CONCISE')
        elif action == ADD_CHECK:
            if 'CHECK' not in self.active_modifiers:
                self.active_modifiers.append('CHECK')
        elif action == REMOVE_LAST:
            if self.active_modifiers:
                self.active_modifiers.pop()
        elif action == RESET_ALL:
            self.active_modifiers = []

    def _build_prompt(self):
        """Build the TinyLlama chat prompt with current modifiers."""
        passage = self.passages[self.current_idx]
        question = self.questions[self.current_idx]

        system_msg = "You are a helpful assistant."
        modifier_str = ' '.join(MODIFIER_TEXTS[m] for m in self.active_modifiers)
        if modifier_str:
            system_msg += ' ' + modifier_str

        user_msg = (
            "Answer the following yes/no question based on the passage.\n\n"
            f"Passage: {passage}\n\n"
            f"Question: {question}\n\n"
            "Answer with only 'Yes' or 'No'."
        )

        # TinyLlama chat template
        prompt = (
            f"<|system|>\n{system_msg}</s>\n"
            f"<|user|>\n{user_msg}</s>\n"
            f"<|assistant|>\n"
        )
        return prompt

    def _generate(self, prompt):
        """Run the frozen LLM. Returns (answer_text, num_new_tokens)."""
        inputs = self.tokenizer(
            prompt, return_tensors='pt',
            truncation=True, max_length=1024,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # skip the input tokens, only keep generated part
        new_ids = outputs[0][inputs['input_ids'].shape[1]:]
        num_tokens = len(new_ids)
        answer = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return answer, num_tokens

    def _compute_similarity(self, answer, reference):
        """Return 1.0 if predicted yes/no matches reference, else 0.0."""
        answer_lower = answer.lower().strip()
        ref = reference.lower().strip()

        # Look at the first few words for yes/no
        pred = None
        for word in answer_lower.split()[:5]:
            word = word.strip('.,!?;:\'"()')
            if word == 'yes':
                pred = 'yes'
                break
            elif word == 'no':
                pred = 'no'
                break

        # if not found yet just search the whole thing
        if pred is None:
            if 'yes' in answer_lower:
                pred = 'yes'
            elif 'no' in answer_lower:
                pred = 'no'
            else:
                pred = 'unknown'

        return 1.0 if pred == ref else 0.0

    def _compute_reward(self, similarity, num_tokens):
        # reward = similarity - length_penalty
        # length_penalty = 0.2 * min(1, tokens/120)
        return similarity - 0.2 * min(1.0, num_tokens / 120.0)



def load_data(data_path='./data/boolq_200.json'):
    """Load the prepared BoolQ dataset. Returns (train_list, test_list)."""
    data = json.loads(Path(data_path).read_text())
    return data['train'], data['test']


def load_llm(device='cuda:0'):
    """Load TinyLlama tokenizer and model. Returns (tokenizer, model)."""
    model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if device.startswith('cuda') and torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto',
        )
        print("[env] TinyLlama loaded (FP16)")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        model.to(device)
        print("[env] TinyLlama loaded on CPU (FP32)")

    model.eval()
    return tokenizer, model


def load_embedder():
    """load the sentence embedder for question encoding"""
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    dim = embedder.get_sentence_embedding_dimension()
    print(f"[env] Embedder loaded ({dim}-dim)")
    return embedder
