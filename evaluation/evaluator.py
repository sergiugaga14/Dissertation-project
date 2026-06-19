"""Evaluation framework for comparing RAG configurations."""

import json
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.pipeline import MedRAGPipeline

console = Console()


@dataclass
class EvalResult:
    """Result of evaluating a single query."""

    query: str
    reference_answer: str
    generated_answer: str
    retrieved_contexts: list[str]
    metrics: dict
    latency_seconds: float


from dataclasses import dataclass


class Evaluator:
    """Evaluate a MedRAG pipeline on a benchmark dataset.

    Computes retrieval metrics (context precision/recall) and
    generation metrics (accuracy, F1, faithfulness).
    """

    def __init__(self, pipeline: MedRAGPipeline):
        self.pipeline = pipeline

    def evaluate_dataset(
        self,
        questions: list[str],
        reference_answers: list[str],
        top_k: int = 5,
        prompt_template: str = "default",
    ) -> dict:
        """Run evaluation on a full dataset."""
        results = []

        for i, (q, ref) in enumerate(zip(questions, reference_answers)):
            console.print(f"  Evaluating [{i+1}/{len(questions)}]: {q[:80]}...")

            start = time.time()
            output = self.pipeline.query(q, top_k=top_k, prompt_template=prompt_template)
            latency = time.time() - start

            contexts = [c["text"] for c in output["retrieved_chunks"]]
            metrics = self._compute_metrics(q, ref, output["answer"], contexts)

            results.append(
                EvalResult(
                    query=q,
                    reference_answer=ref,
                    generated_answer=output["answer"],
                    retrieved_contexts=contexts,
                    metrics=metrics,
                    latency_seconds=latency,
                )
            )

        # Aggregate metrics
        aggregate = self._aggregate_metrics(results)
        return {
            "config": self.pipeline.config_summary(),
            "num_queries": len(questions),
            "aggregate_metrics": aggregate,
            "per_query_results": [
                {
                    "query": r.query,
                    "metrics": r.metrics,
                    "latency": r.latency_seconds,
                }
                for r in results
            ],
        }

    def _compute_metrics(
        self,
        query: str,
        reference: str,
        generated: str,
        contexts: list[str],
    ) -> dict:
        """Compute all metrics for a single query."""
        metrics = {}

        # Exact match (for MCQ benchmarks)
        metrics["exact_match"] = self._exact_match(reference, generated)

        # Token-level F1
        metrics["f1"] = self._token_f1(reference, generated)

        # ROUGE-L
        metrics["rouge_l"] = self._rouge_l(reference, generated)

        return metrics

    def _exact_match(self, reference: str, generated: str) -> float:
        """Check if the answer matches (for MCQ: check if correct letter appears)."""
        ref_clean = reference.strip().lower()
        gen_clean = generated.strip().lower()

        # For MCQ: check if reference answer letter appears at start
        if len(ref_clean) == 1 and ref_clean.isalpha():
            return 1.0 if gen_clean.startswith(ref_clean) else 0.0

        return 1.0 if ref_clean == gen_clean else 0.0

    def _token_f1(self, reference: str, generated: str) -> float:
        """Token-level F1 score."""
        ref_tokens = set(reference.lower().split())
        gen_tokens = set(generated.lower().split())

        if not ref_tokens or not gen_tokens:
            return 0.0

        common = ref_tokens & gen_tokens
        precision = len(common) / len(gen_tokens) if gen_tokens else 0
        recall = len(common) / len(ref_tokens) if ref_tokens else 0

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _rouge_l(self, reference: str, generated: str) -> float:
        """ROUGE-L score (longest common subsequence)."""
        ref_tokens = reference.lower().split()
        gen_tokens = generated.lower().split()

        if not ref_tokens or not gen_tokens:
            return 0.0

        # LCS length
        m, n = len(ref_tokens), len(gen_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i - 1] == gen_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_len = dp[m][n]
        precision = lcs_len / n if n else 0
        recall = lcs_len / m if m else 0

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _aggregate_metrics(self, results: list[EvalResult]) -> dict:
        """Compute mean metrics across all results."""
        if not results:
            return {}

        metric_keys = results[0].metrics.keys()
        aggregate = {}
        for key in metric_keys:
            values = [r.metrics[key] for r in results]
            aggregate[f"mean_{key}"] = sum(values) / len(values)

        latencies = [r.latency_seconds for r in results]
        aggregate["mean_latency"] = sum(latencies) / len(latencies)
        aggregate["p95_latency"] = sorted(latencies)[int(0.95 * len(latencies))]

        return aggregate

    def save_results(self, results: dict, output_path: str | Path) -> None:
        """Save evaluation results to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"[green]Results saved to {path}[/green]")

    def print_summary(self, results: dict) -> None:
        """Print a nice summary table of results."""
        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in results["aggregate_metrics"].items():
            table.add_row(key, f"{value:.4f}")

        console.print(table)
        console.print(f"\nPipeline config: {results['config']}")
