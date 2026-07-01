"""Oracle / judge for the "Analytical Language of John Wilkins" hometask.

The Interactor encapsulates EVERYTHING about the oracle. It owns the hidden
gold animal, validates contestant inputs against the published pools, and runs
a local LLM to decide yes/no answers to questions about that animal.

The LLM is loaded ONCE on the first Interactor instantiation (class-level
singleton) and shared across all subsequent Interactors. Default model:
Qwen/Qwen2.5-3B-Instruct (fits Colab T4).

  interactor.ask(question) -> "yes" or "no"
      Asks a yes/no question about the hidden animal.
      The question MUST be in questions_pool.txt; otherwise raises
      ValueError and does NOT consume the budget.

  interactor.guess(animal) -> "correct" or "wrong"
      Submits a candidate animal. "correct" ends the row.
      The animal MUST be in animals_pool.txt; otherwise raises ValueError
      and does NOT consume the budget.

Scoring per row:
  score = max(0, (1 if solved else 0) - 0.02 * queries_used)
"""
from __future__ import annotations

import re


QUERY_BUDGET = 15
QUERY_COST   = 0.02
DEFAULT_LLM  = "Qwen/Qwen2.5-3B-Instruct"
# The questions in questions_pool.txt refer to the animal generically ("it" /
# "this animal"). The oracle is told which animal is hidden and answers the
# question about exactly that animal.
JUDGE_PROMPT = (
    "You are answering a question about one specific animal.\n"
    "The animal is: {animal}.\n"
    "Answer with a single word, yes or no.\n"
    "Question: {question}"
)


class Interactor:
    # Class-level shared LLM (loaded once, reused for all Interactors).
    _tokenizer = None
    _model = None
    _model_name = None

    @classmethod
    def _ensure_llm(cls, model_name: str = DEFAULT_LLM):
        if cls._model is not None and cls._model_name == model_name:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"  [interactor] loading {model_name} on {device}...")
        cls._tokenizer = AutoTokenizer.from_pretrained(model_name)
        cls._model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype,
        ).to(device).eval()
        cls._model_name = model_name
        print(f"  [interactor] LLM ready.")

    def __init__(self, gold_animal: str,
                 animals_pool: set[str] | list[str],
                 questions_pool: set[str] | list[str],
                 budget: int = QUERY_BUDGET,
                 cost: float = QUERY_COST,
                 model_name: str = DEFAULT_LLM):
        self._ensure_llm(model_name)
        self.gold = gold_animal.strip().lower()
        self.animals_pool = set(a.strip().lower() for a in animals_pool)
        self.questions_pool = set(q.strip().lower() for q in questions_pool)
        self.budget = budget
        self.cost = cost
        self.queries_used = 0
        self.solved = False
        self.history: list[tuple[str, str, str]] = []

    def ask(self, question: str) -> str:
        if not isinstance(question, str):
            raise ValueError(f"ask() requires a string, got {type(question).__name__}")
        q = question.strip().lower()
        if q not in self.questions_pool:
            raise ValueError(
                f"'{question}' is not in questions_pool.txt. "
                f"ask() accepts only questions from the published question pool. "
                f"(Budget not consumed.)")
        self._check_done()
        self.queries_used += 1
        is_yes = self._llm_yes_no(q)
        response = "yes" if is_yes else "no"
        self.history.append(("ask", q, response))
        return response

    def guess(self, animal: str) -> str:
        if not isinstance(animal, str):
            raise ValueError(f"guess() requires a string, got {type(animal).__name__}")
        word = animal.strip().lower()
        if word not in self.animals_pool:
            raise ValueError(
                f"'{animal}' is not in animals_pool.txt. "
                f"guess() accepts only animals from the published animal pool. "
                f"(Budget not consumed.)")
        self._check_done()
        self.queries_used += 1
        if word == self.gold:
            self.solved = True
            response = "correct"
        else:
            response = "wrong"
        self.history.append(("guess", word, response))
        return response

    def is_done(self) -> bool:
        return self.solved or self.queries_used >= self.budget

    def remaining_budget(self) -> int:
        return max(0, self.budget - self.queries_used)

    def score(self) -> float:
        base = 1.0 if self.solved else 0.0
        return max(0.0, base - self.cost * self.queries_used)

    # ---- internals ---------------------------------------------------------

    def _llm_yes_no(self, question: str) -> bool:
        import torch
        prompt = JUDGE_PROMPT.format(animal=self.gold, question=question)
        messages = [{"role": "user", "content": prompt}]
        chat = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(chat, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=5, do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        raw = self._tokenizer.decode(
            out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        clean = re.sub(r"[^a-z]", "", raw.lower())
        if clean.startswith("yes"):
            return True
        if clean.startswith("no"):
            return False
        # Fallback: first hit wins.
        y, n = clean.find("yes"), clean.find("no")
        if y == -1: return False
        if n == -1: return True
        return y < n

    def _check_done(self):
        if self.is_done():
            raise RuntimeError(
                f"Interactor is done (solved={self.solved}, "
                f"queries_used={self.queries_used}/{self.budget}). "
                f"Use interactor.is_done() to check before calling ask() / guess().")
