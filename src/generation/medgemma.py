"""MEDgemma generation layer — the fixed LLM for all experiments."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.retrieval.base import RetrievedChunk


class MedGemmaGenerator:
    """Generate answers using Google's MEDgemma model.

    MEDgemma is kept as the FIXED component across all experiments.
    Only the RAG pipeline (extraction, chunking, embedding, retrieval)
    changes between experiments.
    """

    def __init__(
        self,
        model_name: str = "google/medgemma-4b-it",
        device: str = "auto",
        quantization: str | None = "4bit",
        max_new_tokens: int = 1024,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model with optional quantization
        load_kwargs = {"device_map": device, "torch_dtype": torch.bfloat16}
        if quantization == "4bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "8bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    def generate(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        prompt_template: str = "default",
        temperature: float = 0.0,
    ) -> dict:
        """Generate an answer using retrieved context + MEDgemma.

        Returns a dict with the answer and metadata for evaluation.
        """
        # Build the prompt
        prompt = self._build_prompt(query, retrieved_chunks, prompt_template)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
            )

        # Decode only the generated part
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return {
            "answer": answer.strip(),
            "query": query,
            "prompt_template": prompt_template,
            "num_context_chunks": len(retrieved_chunks),
            "context_sources": [c.source_file for c in retrieved_chunks],
            "input_tokens": inputs["input_ids"].shape[1],
            "output_tokens": len(generated_tokens),
        }

    def _build_prompt(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        template: str,
    ) -> str:
        context = "\n\n---\n\n".join(
            f"[Source: {c.source_file}, Page: {c.page_number}]\n{c.text}" for c in chunks
        )

        if template == "zero_shot":
            return PROMPT_ZERO_SHOT.format(context=context, question=query)
        elif template == "few_shot":
            return PROMPT_FEW_SHOT.format(context=context, question=query)
        elif template == "cot":
            return PROMPT_COT.format(context=context, question=query)
        else:
            return PROMPT_DEFAULT.format(context=context, question=query)


# ── Prompt Templates ──────────────────────────────────────────────────

PROMPT_DEFAULT = """You are a medical AI assistant. Answer the following medical question
using ONLY the provided context. If the context does not contain enough information,
say "I cannot answer this based on the provided context."

## Context
{context}

## Question
{question}

## Answer"""

PROMPT_ZERO_SHOT = """Based on the following medical reference material, answer the question.
Cite the source when possible. If uncertain, state your confidence level.

Reference Material:
{context}

Question: {question}

Answer:"""

PROMPT_FEW_SHOT = """You are a medical expert. Use the provided context to answer questions.

Example 1:
Q: What is the first-line treatment for hypertension?
A: According to current guidelines, first-line treatment for hypertension includes
thiazide diuretics, ACE inhibitors, ARBs, or calcium channel blockers.

Example 2:
Q: What are the warning signs of a stroke?
A: The warning signs of stroke can be remembered using FAST: Face drooping,
Arm weakness, Speech difficulty, Time to call emergency services.

Now answer the following using the provided context:

Context:
{context}

Question: {question}

Answer:"""

PROMPT_COT = """You are a medical expert. Use the provided context to answer the question.
Think through your reasoning step by step before giving your final answer.

Context:
{context}

Question: {question}

Let me reason through this step by step:
1."""
