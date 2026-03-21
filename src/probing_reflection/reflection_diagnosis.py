"""Reflection token diagnosis and extraction module.

This module provides prompt templates for identifying self-reflection tokens
in model outputs. Reflection tokens indicate moments where a model exhibits
metacognitive behaviors like hesitation, self-correction, or verification.

The taxonomy is based on academic literature on self-reflection in language models.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from probing_reflection.types import (
    ReflectionAnalysisReport,
    ReflectionDiagnosisConfig,
    ReflectionToken,
    SampleWithReflection,
)

# Validated reflection token taxonomy (examples, not exhaustive)
REFLECTION_TAXONOMY: dict[str, list[str]] = {
    "hesitation": ["wait", "hmm", "ah", "oh", "umm"],
    "qualification": ["but", "however", "maybe", "actually", "although"],
    "verification": ["check", "verify", "double-check", "reconsider"],
    "redirection": ["alternatively", "on the other hand", "let me think"],
    "transition": ["therefore", "so", "thus", "hence"],
}


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Ensure output directory exists.

    Args:
        output_dir: Path to output directory.

    Returns:
        Path object for the directory.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_diagnosis_prompt(text: str) -> str:
    """Build a prompt for extracting reflection tokens from text.

    Creates a structured prompt that asks an LLM to identify self-reflection
    tokens in the provided text. The prompt emphasizes context-dependent
    judgment to avoid false positives (e.g., "wait" as a command vs. hesitation).

    Args:
        text: The text to analyze for reflection tokens.

    Returns:
        A formatted prompt string for the diagnosis model.

    Example:
        >>> prompt = build_diagnosis_prompt("Wait, let me think about this.")
        >>> # Returns prompt asking LLM to identify reflection tokens
    """
    taxonomy_examples = "\n".join(
        f"  - {category}: {', '.join(tokens)}" for category, tokens in REFLECTION_TAXONOMY.items()
    )

    return f"""Identify self-reflection tokens in the following text.

Self-reflection tokens indicate metacognitive moments where the model exhibits
hesitation, self-correction, verification, or cognitive redirection.

EXAMPLE CATEGORIES (not exhaustive - use judgment):
{taxonomy_examples}

CRITICAL: Context matters! Not every instance of these words indicates reflection.
- "Wait for the result" → NOT reflection (imperative command)
- "Wait, that doesn't seem right" → IS reflection (hesitation marker)
- "Check the box" → NOT reflection (instruction)
- "Let me check if this is correct" → IS reflection (verification)

Judge based on whether the token signals genuine metacognitive activity.

Analyze this text:
{text}

Respond in JSON format with this schema:
{{"tokens": [{{"text": "...", "category": "...", "context": "...", "confidence": 0.0-1.0}}]}}

Requirements:
- text: the exact reflection token found
- category: one of hesitation, qualification, verification, redirection, transition, or other
- context: a brief phrase showing how the token was used
- confidence: 0.0 (not reflection) to 1.0 (definitely reflection)

If no reflection tokens are found, return: {{"tokens": []}}"""


def _parse_json_response(response: str) -> dict[str, object]:
    """Parse JSON response from model output.

    Extracts and parses JSON from a response string, handling malformed
    and missing JSON gracefully.

    Args:
        response: The raw model output text.

    Returns:
        A dict with "tokens" key containing list of token dicts.
        On parse failure: {"tokens": [], "error": "parse_failed"}
        On no JSON found: {"tokens": [], "error": "no_json_found"}
    """
    try:
        # Try to find JSON in the response
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1
        if start_idx == -1 or end_idx == 0:
            return {"tokens": [], "error": "no_json_found"}

        json_str = response[start_idx:end_idx]
        data = json.loads(json_str)

        if "tokens" not in data:
            return {"tokens": []}

        # Return raw dict structure (don't convert to ReflectionToken)
        return {"tokens": data["tokens"]}
    except json.JSONDecodeError:
        return {"tokens": [], "error": "parse_failed"}
    except (KeyError, TypeError, ValueError):
        return {"tokens": [], "error": "parse_failed"}


class ReflectionJudge:
    """LLM-based judge for identifying reflection tokens in text.

    Uses a language model to detect self-reflection tokens that indicate
    metacognitive moments like hesitation, verification, or self-correction.

    Attributes:
        config: Configuration for the diagnosis pipeline.
        model_name: Name or path of the judge model.
        model: The loaded language model (None until load_model is called).
        tokenizer: The loaded tokenizer (None until load_model is called).
        device: Device the model is running on (None until load_model is called).
    """

    def __init__(self, config: ReflectionDiagnosisConfig | str) -> None:
        """Initialize the reflection judge.

        Args:
            config: Either a ReflectionDiagnosisConfig object or a model name string.
                If a string is provided, a default config is created with that model.
        """
        if isinstance(config, str):
            self.config = ReflectionDiagnosisConfig(model_name=config)
        else:
            self.config = config

        self.model_name = self.config.model_name
        self.model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.device: torch.device | None = None

    def load_model(self) -> None:
        """Load the model and tokenizer.

        Follows the inference.py pattern with bfloat16 precision on CUDA
        and float32 on CPU. Sets up pad_token to eos_token if not present.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
        )
        self.model.eval()  # type: ignore[no-untyped-call]
        self.model = self.model.to(self.device)  # type: ignore[arg-type]

    def _parse_json_response(self, response: str) -> list[ReflectionToken]:
        """Parse the JSON response from the model.

        Args:
            response: The raw model output text.

        Returns:
            List of ReflectionToken dicts, or empty list if parsing fails.
        """
        try:
            # Try to find JSON in the response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                return []

            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)

            if "tokens" not in data:
                return []

            tokens: list[ReflectionToken] = []
            for token_data in data["tokens"]:
                if all(k in token_data for k in ("text", "category", "context", "confidence")):
                    tokens.append(
                        ReflectionToken(
                            text=str(token_data["text"]),
                            category=str(token_data["category"]),
                            context=str(token_data["context"]),
                            confidence=float(token_data["confidence"]),
                        )
                    )
            return tokens
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return []

    def judge(self, text: str) -> list[ReflectionToken]:
        """Extract reflection tokens from text.

        Args:
            text: The text to analyze for reflection tokens.

        Returns:
            List of detected reflection tokens with their metadata.

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        if self.model is None or self.tokenizer is None or self.device is None:
            raise RuntimeError("Model not loaded. Call load_model() before judge().")

        prompt = build_diagnosis_prompt(text)

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(  # type: ignore[operator]
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = str(generated_text)[len(prompt) :].strip()

        return self._parse_json_response(response)


def diagnose_sample(judge: ReflectionJudge, sample: dict[str, object]) -> SampleWithReflection:
    """Diagnose a single sample for reflection tokens.

    Analyzes the generated text from a sample to detect self-reflection
    tokens using the provided judge, and computes reflection metrics.

    Args:
        judge: A ReflectionJudge instance (must have load_model() called).
        sample: A dict containing at minimum 'generated' text and other
            sample fields like 'problem_id', 'problem', 'reference_answer'.

    Returns:
        A SampleWithReflection dict with all original fields plus:
            - reflection_tokens: List of detected ReflectionToken objects
            - reflection_count: Number of reflection tokens found
            - reflection_density: Reflection tokens per 100 words

    Example:
        >>> judge = ReflectionJudge("model-name")
        >>> judge.load_model()
        >>> result = diagnose_sample(judge, {"generated": "Wait, let me think..."})
        >>> print(result["reflection_count"])
        1
    """
    generated = str(sample.get("generated", ""))

    subject_val = sample.get("subject")
    level_val = sample.get("level")
    subject: str | None = str(subject_val) if subject_val is not None else None
    level: int | None = (
        level_val
        if isinstance(level_val, int)
        else (int(str(level_val)) if level_val is not None else None)
    )

    if not generated.strip():
        return {
            "problem_id": str(sample.get("problem_id", "")),
            "problem": str(sample.get("problem", "")),
            "generated": "",
            "reference_answer": str(sample.get("reference_answer", "")),
            "subject": subject,
            "level": level,
            "reflection_tokens": [],
            "reflection_count": 0,
            "reflection_density": 0.0,
        }

    tokens = judge.judge(generated)
    token_count = len(tokens)

    words = generated.split()
    word_count = len(words)

    density = (token_count / word_count * 100) if word_count > 0 else 0.0

    result: SampleWithReflection = {
        "problem_id": str(sample.get("problem_id", "")),
        "problem": str(sample.get("problem", "")),
        "generated": generated,
        "reference_answer": str(sample.get("reference_answer", "")),
        "subject": subject,
        "level": level,
        "reflection_tokens": tokens,
        "reflection_count": token_count,
        "reflection_density": density,
    }
    return result


def diagnose_all(
    config: ReflectionDiagnosisConfig,
    judge: ReflectionJudge | None = None,
) -> tuple[list[SampleWithReflection], ReflectionAnalysisReport]:
    """Diagnose reflection tokens across all samples in a JSONL file.

    Processes each sample through the reflection diagnosis pipeline and
    aggregates statistics about detected reflection tokens.

    Args:
        config: Configuration containing input_path, output_dir, and model settings.
        judge: Optional pre-configured ReflectionJudge. If None, creates and loads
            a new judge from config.

    Returns:
        A tuple containing:
            - List of SampleWithReflection dicts with original fields plus analysis
            - ReflectionAnalysisReport with aggregated statistics

    Example:
        >>> config = ReflectionDiagnosisConfig(input_path="samples.jsonl")
        >>> samples, report = diagnose_all(config)
        >>> print(f"Found {report['total_tokens']} reflection tokens")
    """
    ensure_output_dir(config.output_dir)

    processed_samples: list[SampleWithReflection] = []
    total_samples = 0
    total_tokens = 0
    all_densities: list[float] = []
    token_frequency: dict[str, int] = defaultdict(int)
    category_distribution: dict[str, int] = defaultdict(int)
    per_subject_raw: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "total_density": 0.0}
    )
    per_level_raw: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "total_density": 0.0}
    )
    processing_errors = 0

    input_path = Path(config.input_path)
    with open(input_path) as f:
        lines = f.readlines()

    if judge is None and lines:
        judge = ReflectionJudge(config)
        try:
            judge.load_model()
        except Exception:
            judge = None

    for line in tqdm(lines, desc="Diagnosing reflection tokens"):
        total_samples += 1
        try:
            if judge is None:
                raise RuntimeError("Model not loaded")
            sample = json.loads(line)
            result = diagnose_sample(judge, sample)
            processed_samples.append(result)

            total_tokens += result["reflection_count"]
            all_densities.append(result["reflection_density"])

            for token in result["reflection_tokens"]:
                token_frequency[token["text"]] += 1
                category_distribution[token["category"]] += 1

            subject = result.get("subject")
            if subject is not None:
                per_subject_raw[subject]["count"] += 1
                per_subject_raw[subject]["total_density"] += result["reflection_density"]

            level = result.get("level")
            if level is not None:
                level_key = str(level)
                per_level_raw[level_key]["count"] += 1
                per_level_raw[level_key]["total_density"] += result["reflection_density"]

        except (json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError):
            processing_errors += 1

    avg_tokens_per_sample = total_tokens / total_samples if total_samples > 0 else 0.0
    overall_density = sum(all_densities) / len(all_densities) if all_densities else 0.0

    per_subject_stats: dict[str, dict[str, float | int]] = {}
    for subject, stats in per_subject_raw.items():
        count = int(stats["count"])
        total_density = float(stats["total_density"])
        avg_density = total_density / count if count > 0 else 0.0
        per_subject_stats[subject] = {"count": count, "avg_density": avg_density}

    per_level_stats: dict[str, dict[str, float | int]] = {}
    for level_key, stats in per_level_raw.items():
        count = int(stats["count"])
        total_density = float(stats["total_density"])
        avg_density = total_density / count if count > 0 else 0.0
        per_level_stats[level_key] = {"count": count, "avg_density": avg_density}

    report = ReflectionAnalysisReport(
        total_samples=total_samples,
        total_tokens=total_tokens,
        avg_tokens_per_sample=avg_tokens_per_sample,
        overall_density=overall_density,
        token_frequency=dict(token_frequency),
        category_distribution=dict(category_distribution),
        per_subject_stats=per_subject_stats,
        per_level_stats=per_level_stats,
        processing_errors=processing_errors,
    )

    return processed_samples, report


def write_analysis_report(report: ReflectionAnalysisReport, output_path: Path | str) -> None:
    """Write a reflection analysis report to a JSON file.

    Args:
        report: The ReflectionAnalysisReport to write.
        output_path: Path to the output JSON file. Parent directories
            will be created if they don't exist.

    Example:
        >>> report = diagnose_all(config)
        >>> write_analysis_report(report, "outputs/report.json")
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def write_analyzed_jsonl(
    samples: list[SampleWithReflection],
    output_path: Path | str,
) -> None:
    """Write analyzed samples to a JSONL file.

    Args:
        samples: List of samples with reflection analysis results.
        output_path: Path to the output JSONL file.

    Example:
        >>> samples = [{"problem_id": "1", "reflection_count": 2, ...}]
        >>> write_analyzed_jsonl(samples, "analyzed.jsonl")
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
