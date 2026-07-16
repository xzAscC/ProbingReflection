"""Cross-family LLM-as-a-judge for hard-negative validation.

Loads a LOCAL model from a DIFFERENT family than the generator (glm-5.2) and
judges the 25 hard-negative trajectories using the paper's exact 3-label
self-reflection rubric (NO_REFLECTION / SURFACE_REFLECTION / GENUINE_SELF_REFLECTION).

This produces an independent, cross-family judge run that addresses the
independence concern (坑3): generator = glm-5.2, this judge = Qwen3 (or OLMo3).

Usage:
    uv run python scripts/crossfamily_judge.py --model Qwen/Qwen3-1.7B --run 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Protocol, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "hard_negatives"

# The paper's EXACT judge prompt (main.tex §app:prompt_sr, lines 607-643).
JUDGE_SYSTEM = (
    "You are evaluating whether a model response exhibits genuine self-reflection.\n\n"
    "Definition: Genuine self-reflection means that the model explicitly reconsiders, "
    "questions, checks, backtracks on, or revises its earlier reasoning. This must go "
    'beyond superficial discourse markers such as "wait", "hold on", or "let me think". '
    "A response counts as genuine self-reflection only if there is evidence that the "
    "model revisits prior reasoning and meaningfully changes, re-evaluates, or re-derives it.\n\n"
    "Label the response as one of:\n"
    "- NO_REFLECTION: no reflective behavior\n"
    "- SURFACE_REFLECTION: contains reflection-like phrases, but no substantive "
    "reconsideration of prior reasoning\n"
    "- GENUINE_SELF_REFLECTION: explicitly revisits prior reasoning and meaningfully "
    "re-evaluates, revises, or corrects it\n\n"
    "Instructions:\n"
    "1. Read the full question and full response.\n"
    "2. Focus on behavior, not keywords alone.\n"
    "3. Do not classify as genuine self-reflection unless the response refers back to "
    "earlier reasoning and performs a meaningful check, correction, backtrack, or revision.\n"
    "4. Prefer precision over recall.\n"
    "5. If the response is labeled as GENUINE_SELF_REFLECTION, identify all reflection "
    "occurrences in the response, not just the first one.\n"
    "6. For each reflection occurrence, return the reflection phrase, the corresponding "
    "reflection token, and a short evidence quote.\n"
    "7. If there is no genuine self-reflection, return an empty list for "
    '"reflection_instances".\n\n'
    "Return JSON only:\n"
    "{\n"
    '  "label": "NO_REFLECTION | SURFACE_REFLECTION | GENUINE_SELF_REFLECTION",\n'
    '  "confidence": 0.0-1.0,\n'
    '  "reflection_instances": [\n'
    "    {\n"
    '      "reflection_phrase": "...",\n'
    '      "reflection_token": "...",\n'
    '      "evidence_quote": "..."\n'
    "    }\n"
    "  ]\n"
    "}"
)

VALID_LABELS = {"NO_REFLECTION", "SURFACE_REFLECTION", "GENUINE_SELF_REFLECTION"}


class TokenizerOutput(Protocol):
    input_ids: torch.Tensor


class TokenizerProtocol(Protocol):
    pad_token: str | None
    eos_token: str | None

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        return_tensors: str,
    ) -> torch.Tensor: ...

    def __call__(self, text: str, *, return_tensors: str) -> TokenizerOutput: ...

    def decode(self, token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str: ...


class ModelProtocol(Protocol):
    def generate(self, **kwargs: object) -> torch.Tensor: ...

    def eval(self) -> ModelProtocol: ...

    def to(self, device: torch.device) -> ModelProtocol: ...


def build_user_message(problem: str, trajectory: str) -> str:
    return f"Question:\n{problem}\n\nResponse:\n{trajectory}"


def parse_judge_json(text: str) -> dict[str, object]:
    """Extract the judge JSON verdict from raw model output."""
    # find first { and matching last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {
            "label": "PARSE_ERROR",
            "confidence": 0.0,
            "reflection_instances": [],
            "raw": text[:500],
        }
    raw_json = text[start : end + 1]
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        # try to fix trailing commas / smart quotes
        cleaned = raw_json.replace("\u201c", '"').replace("\u201d", '"').replace("'", '"')
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "label": "PARSE_ERROR",
                "confidence": 0.0,
                "reflection_instances": [],
                "raw": raw_json[:500],
            }
    label = str(parsed.get("label", "")).strip().upper()
    # normalize common variants
    if "GENUINE" in label:
        label = "GENUINE_SELF_REFLECTION"
    elif "SURFACE" in label:
        label = "SURFACE_REFLECTION"
    elif "NO" in label or "NONE" in label:
        label = "NO_REFLECTION"
    if label not in VALID_LABELS:
        label = "PARSE_ERROR"
    return {
        "label": label,
        "confidence": float(parsed.get("confidence", 0.0)),
        "reflection_instances": parsed.get("reflection_instances", [])
        if isinstance(parsed.get("reflection_instances"), list)
        else [],
    }


def judge_one(
    model: ModelProtocol,
    tokenizer: TokenizerProtocol,
    problem: str,
    trajectory: str,
    device: torch.device,
) -> dict[str, object]:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": build_user_message(problem, trajectory)},
    ]
    # Prefer chat template; fall back to raw if unavailable
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(device)
        except Exception:
            prompt = JUDGE_SYSTEM + "\n\n" + build_user_message(problem, trajectory)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    else:
        prompt = JUDGE_SYSTEM + "\n\n" + build_user_message(problem, trajectory)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=400,
            do_sample=False,
            temperature=1.0,
        )
    gen = tokenizer.decode(out[0][input_ids.shape[1] :], skip_special_tokens=True)
    return parse_judge_json(gen)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--run", type=int, default=4)
    args = ap.parse_args()

    out_path = DATA_DIR / f"_judge_run_{args.run}_crossfamily.json"

    print(f"Loading cross-family judge model: {args.model}", flush=True)
    t0 = time.time()
    tokenizer = cast(TokenizerProtocol, AutoTokenizer.from_pretrained(args.model))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = cast(
        ModelProtocol,
        AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16),
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded in {time.time() - t0:.1f}s on {device}", flush=True)

    results: list[dict[str, object]] = []
    for i in range(25):
        traj_path = DATA_DIR / f"traj_{i:02d}.json"
        d = json.loads(traj_path.read_text())
        t1 = time.time()
        verdict = judge_one(model, tokenizer, d["problem"], d["trajectory"], device)
        verdict["problem_id"] = i
        verdict["judge_model"] = args.model
        elapsed = time.time() - t1
        results.append(verdict)
        reflection_instances = verdict.get("reflection_instances")
        instance_count = len(reflection_instances) if isinstance(reflection_instances, list) else 0
        print(
            f"[{i:02d}] {verdict['label']} (conf={verdict['confidence']:.2f}) "
            f"#inst={instance_count} [{elapsed:.1f}s]",
            flush=True,
        )

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    labels = [r["label"] for r in results]
    c = Counter(labels)
    n_non_genuine = sum(1 for label in labels if label != "GENUINE_SELF_REFLECTION")
    print(f"\nWrote {out_path}")
    print(f"Label distribution: {dict(c)}")
    print(f"Correct (non-GENUINE): {n_non_genuine}/25")
    return 0


if __name__ == "__main__":
    sys.exit(main())
