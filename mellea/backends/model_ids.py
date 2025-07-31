"""Utilities for model identifiers."""

import dataclasses
from typing import Optional


@dataclasses.dataclass(frozen=True)
class ModelIdentifier:
    """The `ModelIdentifier` class wraps around model identification strings.

    Using model strings is messy:
        1. Different platforms use variations on model id strings.
        2. Using raw strings is annoying because: no autocomplete, typos, hallucinated names, mismatched model and tokenizer names, etc.
    """

    hf_model_name: str
    ollama_name: str
    watsonx_name: str | None = None
    mlx_name: str | None = None

    hf_tokenizer_name: str | None = None  # if None, is the same as hf_model_name


# IBM models

IBM_GRANITE_3_2_8B = ModelIdentifier(
    hf_model_name="ibm-granite/granite-3.2-8b-instruct",
    ollama_name="granite3.2:8b",
    watsonx_name="ibm/granite-3-2b-instruct",
)

IBM_GRANITE_3_3_8B = ModelIdentifier(
    hf_model_name="ibm-granite/granite-3.3-8b-instruct",
    ollama_name="granite3.3:8b",
    watsonx_name="ibm/granite-3-3-8b-instruct",
)

IBM_GRANITE_GUARDIAN_3_0_2B = ModelIdentifier(
    hf_model_name="ibm-granite/granite-guardian-3.0-2b",
    ollama_name="granite3-guardian:2b",
)


# Meta models

META_LLAMA_3_3_70B = ModelIdentifier(
    hf_model_name="meta-llama/Llama-3.3-70B-Instruct",
    ollama_name="llama3.3:70b",
    watsonx_name="meta-llama/llama-3-3-70b-instruct",
    hf_tokenizer_name="unsloth/Llama-3.3-70B-Instruct",
)

META_LLAMA_3_2_3B = ModelIdentifier(
    hf_model_name="unsloth/Llama-3.2-3B-Instruct", ollama_name="llama3.2:3b"
)

META_LLAMA_GUARD3_1B = ModelIdentifier(
    ollama_name="llama-guard3:1b", hf_model_name="meta-llama/Llama-Guard-3-1B"
)

GOOGLE_GEMMA_3N_E4B = ModelIdentifier(
    hf_model_name="google/gemma-3n-e4b-it", ollama_name="gemma3n:e4b"
)

MS_PHI_4_14B = ModelIdentifier(hf_model_name="microsoft/phi-4", ollama_name="phi4:14b")

MS_PHI_4_MINI_REASONING_4B = ModelIdentifier(
    hf_model_name="microsoft/Phi-4-mini-flash-reasoning",
    ollama_name="phi4-mini-reasoning:3.8b",
)


DEEPSEEK_R1_8B = ModelIdentifier(
    hf_model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    ollama_name="deepseek-r1:8b",
)


QWEN3_8B = ModelIdentifier(hf_model_name="Qwen/Qwen3-8B", ollama_name="qwen3:8b")

MISTRALAI_MISTRAL_0_3_7b = ModelIdentifier(
    hf_model_name="mistralai/Mistral-7B-Instruct-v0.3", ollama_name="mistral:7b"
)

MISTRALAI_MISTRAL_SMALL_24B = ModelIdentifier(
    hf_model_name="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    ollama_name="mistral-small:latest",
)

HF_SMOLLM2_2B = ModelIdentifier(
    ollama_name="smollm2:1.7b",
    hf_model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    mlx_name="mlx-community/SmolLM2-1.7B-Instruct",
)

HF_SMOLLM3_3B_no_ollama = ModelIdentifier(
    hf_model_name="HuggingFaceTB/SmolLM3-3B", ollama_name=""
)
