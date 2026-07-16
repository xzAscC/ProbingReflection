"""Prompt template functions for various inference and evaluation tasks.

This module centralizes all prompt formatting functions to ensure
consistency across different modules and make prompt modifications
easier to manage.
"""

from __future__ import annotations


def format_cot_prompt(problem: str) -> str:
    """Format a math problem with chain-of-thought prompting.

    Args:
        problem: The math problem text.

    Returns:
        Formatted prompt with CoT instructions.
    """
    return (
        f"Please reason step by step, and put your final answer within \\boxed{{}}."
        f"\n\nProblem: {problem}\n\nSolution:"
    )


def build_comparison_prompt(answer_a: str, answer_b: str) -> str:
    """Create a comparison prompt for judging answer equivalence.

    The prompt is designed to mitigate verbosity bias by explicitly
    instructing not to reward longer answers.

    Args:
        answer_a: The first answer to compare (reference).
        answer_b: The second answer to compare (candidate).

    Returns:
        Formatted prompt string for the judge model.
    """
    return f"""Compare these two mathematical answers. Are they semantically equivalent?

Reference: {answer_a}
Candidate: {answer_b}

CRITICAL: Do NOT reward longer answers. Conciseness is equally valuable.

First explain your reasoning step by step.
Then provide your verdict.

Respond in JSON format:
{{"explanation": "...", "equivalent": true/false, "confidence": 0.0-1.0}}"""


# Reflection token taxonomy for diagnosis
REFLECTION_TAXONOMY: dict[str, list[str]] = {
    "hesitation": ["wait", "hmm", "ah", "oh", "umm"],
    "qualification": ["but", "however", "maybe", "actually", "although"],
    "verification": ["check", "verify", "double-check", "reconsider"],
    "redirection": ["alternatively", "on the other hand", "let me think"],
    "transition": ["therefore", "so", "thus", "hence"],
}


def build_diagnosis_prompt(text: str) -> str:
    """Build a prompt for extracting reflection tokens from text.

    Creates a structured prompt that asks an LLM to identify self-reflection
    tokens in the provided text. The prompt emphasizes context-dependent
    judgment to avoid false positives.

    Args:
        text: The text to analyze for reflection tokens.

    Returns:
        A formatted prompt string for the diagnosis model.
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


def build_roscoe_prompt(reflection_text: str) -> str:
    """Build a prompt for ROSCOE-based reasoning quality evaluation.

    Creates a structured prompt that asks an LLM to evaluate step-by-step
    reasoning quality using 5 core metrics on a 1-5 scale.

    Args:
        reflection_text: The reasoning text to evaluate.

    Returns:
        A formatted prompt string for the evaluation model.
    """
    return f"""Evaluate the quality of this reasoning chain using 5 criteria.

1. FAITHFULNESS (1-5): Is each step grounded in the problem context?
   - 1: Contains hallucinations or fabricated facts
   - 3: Mostly grounded with minor misinterpretations
   - 5: Every step is directly traceable to source

2. COHERENCE (1-5): Do steps logically follow without contradictions?
   - 1: Major logical contradictions exist
   - 3: Minor inconsistencies but overall logical
   - 5: Flawless logical progression

3. INFORMATIVENESS (1-5): Does each step add new relevant information?
   - 1: No new information or trivial restatements
   - 3: Adequate progress toward solution
   - 5: Each step optimally advances reasoning

4. REPETITION (1-5): Are there redundant or circular steps?
   - 1: Significant repetition or circular reasoning
   - 3: Some redundancy present
   - 5: Each step is distinct and novel

5. COMPLETENESS (1-5): Are all essential reasoning steps included?
   - 1: Missing critical steps
   - 3: Basic coverage but gaps exist
   - 5: Complete reasoning path to conclusion

Reasoning to evaluate:
{reflection_text}

Respond in JSON format:
{{"faithfulness": 1-5, "coherence": 1-5, "informativeness": 1-5,
 "repetition": 1-5, "completeness": 1-5}}"""
